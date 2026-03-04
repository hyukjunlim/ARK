#!/usr/bin/env python3
"""
Slab Relaxation Script

Standalone script for relaxing slabs only.

Usage:
    python slab_relaxation.py --slab_dir ../1_data_prep/data_slabs/precomputed_slabs \
                              --output_dir ./slabs_relaxed \
                              --fmax 0.05 \
                              --steps 300
"""

import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.optimize import LBFGS
from ase import Atoms
import torch
from datetime import datetime
from tqdm import tqdm

# Add utility path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utility'))

# Add Open-Catalyst-Dataset path for loading PKL files
sys.path.insert(0, str(Path(__file__).parent.parent / '1_data_prep' / 'Open-Catalyst-Dataset'))

from model_loading import CrackCalculator
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

# Import ocdata for unpickling slab files
try:
    from ocdata.core import Bulk, Slab, Adsorbate, AdsorbateSlabConfig
except ImportError:
    print("Warning: ocdata module not found. Slab PKL files may not load properly.")


def prepare_atoms_for_relaxation(atoms):
    """
    Prepare atoms object for relaxation.

    Args:
        atoms: ASE Atoms object

    Returns:
        Prepared atoms object
    """
    atoms_copy = atoms.copy()

    # Ensure PBC
    if not atoms_copy.pbc.any():
        atoms_copy.set_pbc([True, True, True])

    # For gas molecules in small cells, expand box
    if atoms_copy.cell.volume < 100:
        atoms_copy.set_cell([20, 20, 20])
        atoms_copy.center()

    return atoms_copy


def relax_structure(atoms, calculator, model_name, fmax=0.05, steps=300, log_path=None):
    """
    Relax structure using LBFGS optimizer.

    Args:
        atoms: ASE Atoms object
        calculator: Calculator instance
        model_name: Name of the model (for logging)
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum number of optimization steps
        log_path: Path to save optimization log

    Returns:
        dict with 'energy', 'time', 'steps', 'converged', 'atoms'
    """
    try:
        atoms_prepared = prepare_atoms_for_relaxation(atoms)
        atoms_prepared.calc = calculator

        # Force initial energy/force calculation
        initial_energy = atoms_prepared.get_potential_energy()
        initial_forces = atoms_prepared.get_forces()

        # Create optimizer
        optimizer = LBFGS(atoms_prepared, trajectory=None, logfile=log_path)

        # Time the relaxation
        start_time = time.time()
        optimizer.run(fmax=fmax, steps=steps)
        end_time = time.time()

        relaxation_time = end_time - start_time
        num_steps = optimizer.nsteps

        # Get final energy
        try:
            energy = atoms_prepared.get_potential_energy()
        except Exception as e:
            print(f"    ⚠️  {model_name} failed to get final energy, using initial")
            energy = initial_energy

        # Check convergence
        converged = (optimizer.nsteps < steps)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'energy': float(energy),
            'time': relaxation_time,
            'steps': num_steps,
            'converged': converged,
            'atoms': atoms_prepared
        }

    except Exception as e:
        print(f"    ❌ {model_name} relaxation failed: {str(e)}")
        return None


def process_slab_file(slab_path, crack_calc, gemnet_calc, fmax, steps, log_dir):
    """
    Process a single slab PKL file.

    Args:
        slab_path: Path to slab PKL file
        crack_calc: Crack calculator
        gemnet_calc: GemNet-OC calculator
        fmax: Force convergence criterion
        steps: Maximum optimization steps
        log_dir: Directory for optimization logs

    Returns:
        tuple: (slab_key, result_dict)
    """
    try:
        # Load slab data
        with open(slab_path, 'rb') as f:
            slab_data = pickle.load(f)

        bulk_id = slab_data['bulk_src_id']
        formula = slab_data['formula']
        slabs = slab_data['slabs']
        slab_metadata = slab_data['slab_metadata']

        if len(slabs) == 0:
            return None

        # Use most stable slab (rank 0)
        slab = slabs[0]
        metadata = slab_metadata[0]

        # Convert Slab object to ASE Atoms if needed
        if hasattr(slab, 'atoms'):
            # Slab object has .atoms attribute
            slab_atoms = slab.atoms
        elif isinstance(slab, Atoms):
            # Already an Atoms object
            slab_atoms = slab
        else:
            # Try to convert
            slab_atoms = Atoms(slab)

        miller_str = f"{metadata['miller'][0]}{metadata['miller'][1]}{metadata['miller'][2]}"
        slab_key = f"{bulk_id}_{miller_str}"

        result = {
            'bulk_id': bulk_id,
            'formula': formula,
            'miller': metadata['miller'],
            'surface_energy': metadata['surface_energy'],
            'num_atoms': len(slab_atoms),
            'initial_atoms': slab_atoms  # Store initial structure
        }

        # Relax with Crack
        if crack_calc is not None:
            log_path = os.path.join(log_dir, f"{slab_key}_crack.log")
            crack_result = relax_structure(slab_atoms, crack_calc, "Crack", fmax, steps, log_path)
            if crack_result:
                result['crack'] = {
                    'energy': crack_result['energy'],
                    'time': crack_result['time'],
                    'steps': crack_result['steps'],
                    'converged': crack_result['converged'],
                    'relaxed_atoms': crack_result['atoms']
                }

        # Relax with GemNet-OC
        if gemnet_calc is not None:
            log_path = os.path.join(log_dir, f"{slab_key}_gemnet.log")
            gemnet_result = relax_structure(slab_atoms, gemnet_calc, "GemNet-OC", fmax, steps, log_path)
            if gemnet_result:
                result['gemnet'] = {
                    'energy': gemnet_result['energy'],
                    'time': gemnet_result['time'],
                    'steps': gemnet_result['steps'],
                    'converged': gemnet_result['converged'],
                    'relaxed_atoms': gemnet_result['atoms']
                }

        return slab_key, result

    except Exception as e:
        print(f"    Error processing slab file {slab_path}: {e}")
        return None


def save_results(slab_results, output_dir):
    """
    Save slab relaxation results to JSON and trajectory files.

    Args:
        slab_results: Dictionary of slab relaxation results
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for trajectories
    traj_crack_dir = output_path / 'trajectories' / 'crack'
    traj_gemnet_dir = output_path / 'trajectories' / 'gemnet'
    traj_crack_dir.mkdir(parents=True, exist_ok=True)
    traj_gemnet_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, Atoms):
            # Don't serialize Atoms objects in JSON
            return None
        return obj

    # Prepare JSON data (without Atoms objects)
    json_results = {}
    for slab_key, slab_data in slab_results.items():
        json_data = {
            'bulk_id': slab_data['bulk_id'],
            'formula': slab_data['formula'],
            'miller': slab_data['miller'],
            'surface_energy': slab_data['surface_energy'],
            'num_atoms': slab_data['num_atoms']
        }

        # Add Crack results (without atoms)
        if 'crack' in slab_data:
            json_data['crack'] = {
                'energy': slab_data['crack']['energy'],
                'time': slab_data['crack']['time'],
                'steps': slab_data['crack']['steps'],
                'converged': slab_data['crack']['converged']
            }
            # Save relaxed structure as trajectory
            traj_path = traj_crack_dir / f"{slab_key}_relaxed.traj"
            write(str(traj_path), slab_data['crack']['relaxed_atoms'])

        # Add GemNet results (without atoms)
        if 'gemnet' in slab_data:
            json_data['gemnet'] = {
                'energy': slab_data['gemnet']['energy'],
                'time': slab_data['gemnet']['time'],
                'steps': slab_data['gemnet']['steps'],
                'converged': slab_data['gemnet']['converged']
            }
            # Save relaxed structure as trajectory
            traj_path = traj_gemnet_dir / f"{slab_key}_relaxed.traj"
            write(str(traj_path), slab_data['gemnet']['relaxed_atoms'])

        json_results[slab_key] = json_data

    # Save JSON results
    json_file = output_path / "slab_results.json"
    with open(json_file, 'w') as f:
        json.dump(convert_types(json_results), f, indent=2)
    print(f"\n✅ Slab results saved to {json_file}")
    print(f"✅ Trajectories saved to {output_path / 'trajectories'}")


def initialize_models():
    """Initialize Crack and GemNet-OC models."""
    teacher_model = "/home/jsh9967/10_crack/4_re_on_0721/model/gnoc_oc22_oc20_all_s2ef.pt"
    student_model = "/home/jsh9967/10_crack/4_re_on_0721/model/crack.pt"

    print("\nInitializing models...")
    print("=" * 70)

    crack_calc = None
    try:
        print("Loading Crack model...")
        crack_calc = CrackCalculator(checkpoint_path=student_model, device="cuda")
        print("✅ Crack model loaded")
    except Exception as e:
        print(f"❌ Crack model failed: {e}")

    gemnet_calc = None
    try:
        print("Loading GemNet-OC model...")
        gemnet_calc = OCPCalculator(checkpoint_path=teacher_model, cpu=False)
        print("✅ GemNet-OC model loaded")
    except Exception as e:
        print(f"❌ GemNet-OC model failed: {e}")

    print("=" * 70)

    return crack_calc, gemnet_calc


def main():
    """Main execution function for slab relaxation."""
    parser = argparse.ArgumentParser(
        description='Slab relaxation script'
    )
    parser.add_argument('--slab_dir', type=str, required=True,
                       help='Directory containing slab PKL files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--fmax', type=float, default=0.05,
                       help='Force convergence criterion (eV/Å)')
    parser.add_argument('--steps', type=int, default=300,
                       help='Maximum optimization steps')
    parser.add_argument('--max_slabs', type=int, default=None,
                       help='Maximum number of slabs to process (for testing)')

    args = parser.parse_args()

    print("=" * 70)
    print("SLAB RELAXATION SCRIPT")
    print("=" * 70)
    print(f"\nSlab directory: {args.slab_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Relaxation: fmax={args.fmax}, steps={args.steps}")

    # Create log directory
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Initialize models
    crack_calc, gemnet_calc = initialize_models()

    if crack_calc is None and gemnet_calc is None:
        print("❌ No models loaded. Exiting.")
        return

    # Find slab PKL files
    slab_files = sorted(Path(args.slab_dir).glob('*.pkl'))

    if args.max_slabs:
        slab_files = slab_files[:args.max_slabs]

    print(f"\nFound {len(slab_files)} slab files to process")

    # Process slabs
    print("\n" + "=" * 70)
    print("RELAXING SLABS")
    print("=" * 70)

    slab_results = {}
    processed_slabs = 0

    for slab_file in tqdm(slab_files, desc="Processing slabs"):
        result = process_slab_file(slab_file, crack_calc, gemnet_calc, args.fmax, args.steps, log_dir)
        if result:
            slab_key, slab_data = result
            slab_results[slab_key] = slab_data
            processed_slabs += 1

    print(f"\n✅ Processed {processed_slabs} slabs")

    # Save results
    save_results(slab_results, args.output_dir)

    print(f"\n✅ Slab relaxation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()