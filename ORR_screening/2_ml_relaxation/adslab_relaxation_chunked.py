#!/usr/bin/env python3
"""
Adslab Relaxation Script (Chunked Version)

Processes a subset of adslabs based on chunk_id and total_chunks.
Implements improved logging strategy - one log file per slab.

Usage:
    python adslab_relaxation_chunked.py --adsorbate CO \
                                        --adslab_dir ../1_data_prep/data_co_adslab \
                                        --slab_results ./slabs_relaxed/slab_results.json \
                                        --gas_data ./gas/gas_data.json \
                                        --output_dir ./CO/chunk_0 \
                                        --chunk_id 0 \
                                        --total_chunks 4
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from ase.io import read, write, Trajectory
from ase.optimize import LBFGS
import torch
from datetime import datetime
from tqdm import tqdm

# Add utility path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utility'))

from model_loading import CrackCalculator
from fairchem.core.common.relaxation.ase_utils import OCPCalculator


def load_gas_energies(gas_data_path):
    """
    Load gas molecule energies from JSON file.

    Args:
        gas_data_path: Path to gas_data.json

    Returns:
        dict: {molecule_name: {'crack': energy, 'gemnet': energy}}
    """
    with open(gas_data_path, 'r') as f:
        gas_data = json.load(f)

    gas_energies = {}
    for mol in gas_data['molecules']:
        gas_energies[mol['name']] = {
            'crack': mol['crack_energy'],
            'gemnet': mol['gemnet_energy'],
            'formula': mol['formula']
        }

    return gas_energies


def load_slab_results(slab_results_path):
    """
    Load pre-computed slab relaxation results.

    Args:
        slab_results_path: Path to slab_results.json

    Returns:
        dict: Slab relaxation results
    """
    with open(slab_results_path, 'r') as f:
        return json.load(f)


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


def relax_structure(atoms, calculator, model_name, fmax=0.05, steps=300, log_file=None):
    """
    Relax structure using LBFGS optimizer.

    Args:
        atoms: ASE Atoms object
        calculator: Calculator instance
        model_name: Name of the model (for logging)
        fmax: Force convergence criterion (eV/Å)
        steps: Maximum number of optimization steps
        log_file: File handle for logging (append mode)

    Returns:
        dict with 'energy', 'time', 'steps', 'converged', 'atoms'
    """
    try:
        atoms_prepared = prepare_atoms_for_relaxation(atoms)
        atoms_prepared.calc = calculator

        # Force initial energy/force calculation
        initial_energy = atoms_prepared.get_potential_energy()
        initial_forces = atoms_prepared.get_forces()

        # Custom logging to the shared log file
        if log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Starting {model_name} relaxation at {datetime.now()}\n")
            log_file.write(f"Initial energy: {initial_energy:.6f} eV\n")
            log_file.write(f"Max force: {np.max(np.abs(initial_forces)):.6f} eV/Å\n")
            log_file.flush()

        # Create optimizer with logfile for proper logging
        # Use a temporary logfile path if log_file is provided
        temp_log_path = None
        if log_file:
            temp_log_path = f"/tmp/{model_name}_{datetime.now().timestamp()}.log"

        optimizer = LBFGS(atoms_prepared, trajectory=None, logfile=temp_log_path)

        # Time the relaxation
        start_time = time.time()

        # Run the optimizer properly
        optimizer.run(fmax=fmax, steps=steps)

        end_time = time.time()
        relaxation_time = end_time - start_time

        # Copy optimizer log to main log file and clean up
        if temp_log_path and log_file and os.path.exists(temp_log_path):
            with open(temp_log_path, 'r') as temp_log:
                log_file.write(temp_log.read())
                log_file.flush()
            os.remove(temp_log_path)

        # Get final energy
        try:
            energy = atoms_prepared.get_potential_energy()
            final_forces = atoms_prepared.get_forces()
        except Exception as e:
            if log_file:
                log_file.write(f"⚠️  {model_name} failed to get final energy: {e}\n")
                log_file.flush()
            energy = initial_energy
            final_forces = initial_forces

        # Check convergence
        converged = (optimizer.nsteps < steps)
        num_steps = optimizer.nsteps

        # Final logging
        if log_file:
            log_file.write(f"Relaxation completed in {relaxation_time:.2f} seconds\n")
            log_file.write(f"Steps taken: {num_steps}/{steps}\n")
            log_file.write(f"Final energy: {energy:.6f} eV\n")
            log_file.write(f"Final max force: {np.max(np.abs(final_forces)):.6f} eV/Å\n")
            log_file.write(f"Converged: {converged}\n")
            log_file.flush()

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
        if log_file:
            log_file.write(f"❌ {model_name} relaxation failed: {str(e)}\n")
            log_file.flush()
        print(f"    ❌ {model_name} relaxation failed: {str(e)}")
        return None


def process_adslab_file(adslab_path, crack_calc, gemnet_calc, fmax, steps, log_dir, last_frame_dir):
    """
    Process a single adslab trajectory file.
    Uses improved logging strategy - one log file per slab.

    Args:
        adslab_path: Path to adslab trajectory file
        crack_calc: Crack calculator
        gemnet_calc: GemNet-OC calculator
        fmax: Force convergence criterion
        steps: Maximum optimization steps
        log_dir: Directory for optimization logs
        last_frame_dir: Directory for last frame trajectories

    Returns:
        tuple: (adslab_key, result_dict)
    """
    try:
        # Load all configurations from trajectory
        adslabs = read(adslab_path, index=':')

        if len(adslabs) == 0:
            return None

        # Get metadata from first structure
        first_atoms = adslabs[0]
        bulk_id = first_atoms.info['bulk_id']
        miller = first_atoms.info['miller']
        adsorbate = first_atoms.info['adsorbate']

        miller_str = f"{miller[0]}{miller[1]}{miller[2]}"
        adslab_key = f"{bulk_id}_{miller_str}"

        # Create/open single log file for this slab
        crack_log_path = os.path.join(log_dir, f"{adslab_key}_{adsorbate}_crack.log")
        gemnet_log_path = os.path.join(log_dir, f"{adslab_key}_{adsorbate}_gemnet.log")

        results = []

        # Open log files in append mode
        with open(crack_log_path, 'a') as crack_log, open(gemnet_log_path, 'a') as gemnet_log:

            # Write header for this file
            crack_log.write(f"\n{'#'*70}\n")
            crack_log.write(f"# Processing {adslab_path.name}\n")
            crack_log.write(f"# Bulk ID: {bulk_id}, Miller: {miller}, Adsorbate: {adsorbate}\n")
            crack_log.write(f"# Total configurations: {len(adslabs)}\n")
            crack_log.write(f"# Timestamp: {datetime.now()}\n")
            crack_log.write(f"{'#'*70}\n\n")

            gemnet_log.write(f"\n{'#'*70}\n")
            gemnet_log.write(f"# Processing {adslab_path.name}\n")
            gemnet_log.write(f"# Bulk ID: {bulk_id}, Miller: {miller}, Adsorbate: {adsorbate}\n")
            gemnet_log.write(f"# Total configurations: {len(adslabs)}\n")
            gemnet_log.write(f"# Timestamp: {datetime.now()}\n")
            gemnet_log.write(f"{'#'*70}\n\n")

            # Process each configuration
            for config_idx, adslab in enumerate(adslabs):
                config_result = {
                    'config_idx': config_idx,
                    'num_atoms': len(adslab)
                }

                config_key = f"{adslab_key}_config{config_idx}"

                # Write config header
                crack_log.write(f"\n--- Configuration {config_idx} ---\n")
                gemnet_log.write(f"\n--- Configuration {config_idx} ---\n")

                # Relax with Crack
                if crack_calc is not None:
                    crack_result = relax_structure(adslab, crack_calc, "Crack", fmax, steps, crack_log)
                    if crack_result:
                        config_result['crack'] = {
                            'energy': crack_result['energy'],
                            'time': crack_result['time'],
                            'steps': crack_result['steps'],
                            'converged': crack_result['converged']
                        }
                        # Save the last frame
                        last_frame_path = os.path.join(last_frame_dir, 'crack', f"{config_key}_crack_last.traj")
                        write(last_frame_path, crack_result['atoms'])

                # Relax with GemNet-OC
                if gemnet_calc is not None:
                    gemnet_result = relax_structure(adslab, gemnet_calc, "GemNet-OC", fmax, steps, gemnet_log)
                    if gemnet_result:
                        config_result['gemnet'] = {
                            'energy': gemnet_result['energy'],
                            'time': gemnet_result['time'],
                            'steps': gemnet_result['steps'],
                            'converged': gemnet_result['converged']
                        }
                        # Save the last frame
                        last_frame_path = os.path.join(last_frame_dir, 'gemnet', f"{config_key}_gemnet_last.traj")
                        write(last_frame_path, gemnet_result['atoms'])

                results.append(config_result)

        summary = {
            'bulk_id': bulk_id,
            'miller': miller,
            'adsorbate': adsorbate,
            'total_configs': len(results),
            'configurations': results
        }

        return adslab_key, summary

    except Exception as e:
        print(f"    Error processing adslab file {adslab_path}: {e}")
        return None


def calculate_adsorption_energies(slab_results, adslab_results, gas_energies, adsorbate):
    """
    Calculate adsorption energies.

    E_ads = E_adslab - E_slab - E_gas

    Args:
        slab_results: Dict of slab relaxation results
        adslab_results: Dict of adslab relaxation results
        gas_energies: Dict of gas molecule energies
        adsorbate: Adsorbate molecule name (e.g., 'CO')

    Returns:
        Dict with adsorption energy results
    """
    adsorption_results = {}

    if adsorbate not in gas_energies:
        print(f"Warning: Gas energy for {adsorbate} not found!")
        return adsorption_results

    gas_crack = gas_energies[adsorbate]['crack']
    gas_gemnet = gas_energies[adsorbate]['gemnet']

    for adslab_key, adslab_data in adslab_results.items():
        # Find matching slab
        if adslab_key not in slab_results:
            print(f"Warning: No matching slab for {adslab_key}")
            continue

        slab_data = slab_results[adslab_key]

        # Calculate for each configuration
        config_ads_energies = []

        for config in adslab_data['configurations']:
            config_result = {'config_idx': config['config_idx']}

            # Crack adsorption energy
            if 'crack' in slab_data and 'crack' in config:
                E_adslab = config['crack']['energy']
                E_slab = slab_data['crack']['energy']
                E_gas = gas_crack
                E_ads = E_adslab - E_slab - E_gas

                config_result['crack_ads_energy'] = E_ads
                config_result['crack_adslab_energy'] = E_adslab
                config_result['crack_slab_energy'] = E_slab

            # GemNet adsorption energy
            if 'gemnet' in slab_data and 'gemnet' in config:
                E_adslab = config['gemnet']['energy']
                E_slab = slab_data['gemnet']['energy']
                E_gas = gas_gemnet
                E_ads = E_adslab - E_slab - E_gas

                config_result['gemnet_ads_energy'] = E_ads
                config_result['gemnet_adslab_energy'] = E_adslab
                config_result['gemnet_slab_energy'] = E_slab

            config_ads_energies.append(config_result)

        adsorption_results[adslab_key] = {
            'bulk_id': adslab_data['bulk_id'],
            'miller': adslab_data['miller'],
            'adsorbate': adsorbate,
            'configurations': config_ads_energies
        }

    return adsorption_results


def save_chunk_results(adslab_results, adsorption_results, output_dir, adsorbate, chunk_id):
    """
    Save chunk results to JSON files.

    Args:
        adslab_results: Adslab relaxation results for this chunk
        adsorption_results: Adsorption energy results for this chunk
        output_dir: Output directory
        adsorbate: Adsorbate name
        chunk_id: Chunk identifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    # Save adslab results
    adslab_file = output_path / f"{adsorbate}_adslab_results_chunk{chunk_id}.json"
    with open(adslab_file, 'w') as f:
        json.dump(convert_types(adslab_results), f, indent=2)
    print(f"✅ Adslab results saved to {adslab_file}")

    # Save adsorption energy results
    if adsorption_results:
        ads_file = output_path / f"{adsorbate}_adsorption_energies_chunk{chunk_id}.json"
        with open(ads_file, 'w') as f:
            json.dump(convert_types(adsorption_results), f, indent=2)
        print(f"✅ Adsorption energies saved to {ads_file}")


def get_chunk_files(all_files, chunk_id, total_chunks):
    """
    Get the subset of files for this chunk.

    Args:
        all_files: List of all files
        chunk_id: Current chunk ID (0-indexed)
        total_chunks: Total number of chunks

    Returns:
        List of files for this chunk
    """
    chunk_size = len(all_files) // total_chunks
    remainder = len(all_files) % total_chunks

    # Calculate start and end indices for this chunk
    if chunk_id < remainder:
        start_idx = chunk_id * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = chunk_id * chunk_size + remainder
        end_idx = start_idx + chunk_size

    return all_files[start_idx:end_idx]


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
    """Main execution function for chunked adslab relaxation."""
    parser = argparse.ArgumentParser(
        description='Chunked adslab relaxation script'
    )
    parser.add_argument('--adsorbate', type=str, required=True,
                       help='Adsorbate molecule name (e.g., CO, NO)')
    parser.add_argument('--adslab_dir', type=str, required=True,
                       help='Directory containing adslab trajectory files')
    parser.add_argument('--slab_results', type=str, required=False, default=None,
                       help='Path to pre-computed slab_results.json (optional, for adsorption energy calculation)')
    parser.add_argument('--gas_data', type=str, required=False, default=None,
                       help='Path to gas molecule energies JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--fmax', type=float, default=0.05,
                       help='Force convergence criterion (eV/Å)')
    parser.add_argument('--steps', type=int, default=300,
                       help='Maximum optimization steps')
    parser.add_argument('--chunk_id', type=int, required=True,
                       help='Chunk ID (0-indexed)')
    parser.add_argument('--total_chunks', type=int, required=True,
                       help='Total number of chunks')

    args = parser.parse_args()

    print("=" * 70)
    print(f"CHUNKED ADSLAB RELAXATION - {args.adsorbate}")
    print(f"Chunk {args.chunk_id}/{args.total_chunks}")
    print("=" * 70)
    print(f"\nAdsorbate: {args.adsorbate}")
    print(f"Adslab directory: {args.adslab_dir}")
    print(f"Slab results: {args.slab_results}")
    print(f"Gas data: {args.gas_data if args.gas_data else 'Not provided'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Relaxation: fmax={args.fmax}, steps={args.steps}")
    print(f"Processing chunk {args.chunk_id} of {args.total_chunks}")

    # Create output directories
    log_dir = os.path.join(args.output_dir, 'logs')
    last_frame_dir = os.path.join(args.output_dir, 'last_frames')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(last_frame_dir, 'crack'), exist_ok=True)
    os.makedirs(os.path.join(last_frame_dir, 'gemnet'), exist_ok=True)

    # Load pre-computed slab results (optional)
    slab_results = {}
    if args.slab_results:
        print("\nLoading slab results...")
        try:
            slab_results = load_slab_results(args.slab_results)
            print(f"✅ Loaded results for {len(slab_results)} slabs")
        except Exception as e:
            print(f"⚠️  Could not load slab results: {e}. Skipping adsorption energy calculation.")
            slab_results = {}
    else:
        print("\n⚠️  No slab results provided. Skipping adsorption energy calculation.")

    # Load gas energies (optional)
    gas_energies = {}
    calculate_ads_energy = False
    if args.gas_data and slab_results:
        print("\nLoading gas molecule energies...")
        try:
            gas_energies = load_gas_energies(args.gas_data)
            if args.adsorbate in gas_energies:
                print(f"✅ {args.adsorbate}: Crack={gas_energies[args.adsorbate]['crack']:.4f} eV, "
                      f"GemNet={gas_energies[args.adsorbate]['gemnet']:.4f} eV")
                calculate_ads_energy = True
            else:
                print(f"⚠️  Gas energy for {args.adsorbate} not found.")
        except Exception as e:
            print(f"⚠️  Could not load gas data: {e}")
    elif args.gas_data and not slab_results:
        print("\n⚠️  Gas data provided but no slab results. Skipping adsorption energy calculation.")
    else:
        print("\n⚠️  No gas data provided. Skipping adsorption energy calculation.")

    # Initialize models
    crack_calc, gemnet_calc = initialize_models()

    if crack_calc is None and gemnet_calc is None:
        print("❌ No models loaded. Exiting.")
        return

    # Find all adslab files
    adslab_pattern = f"*_{args.adsorbate}.traj"
    all_adslab_files = sorted(Path(args.adslab_dir).glob(adslab_pattern))

    # Get chunk files
    chunk_files = get_chunk_files(all_adslab_files, args.chunk_id, args.total_chunks)

    print(f"\nTotal adslab files: {len(all_adslab_files)}")
    print(f"Files in this chunk: {len(chunk_files)}")

    if len(chunk_files) == 0:
        print("⚠️  No files to process in this chunk")
        return

    # Process adslabs
    print("\n" + "=" * 70)
    print(f"RELAXING ADSLABS - CHUNK {args.chunk_id}")
    print("=" * 70)

    adslab_results = {}
    processed_adslabs = 0

    for adslab_file in tqdm(chunk_files, desc=f"Processing chunk {args.chunk_id}"):
        result = process_adslab_file(adslab_file, crack_calc, gemnet_calc,
                                    args.fmax, args.steps, log_dir, last_frame_dir)
        if result:
            adslab_key, adslab_data = result
            adslab_results[adslab_key] = adslab_data
            processed_adslabs += 1

    print(f"\n✅ Processed {processed_adslabs} adslabs in chunk {args.chunk_id}")

    # Calculate adsorption energies (optional)
    adsorption_results = {}
    if calculate_ads_energy:
        print("\n" + "=" * 70)
        print("CALCULATING ADSORPTION ENERGIES")
        print("=" * 70)

        adsorption_results = calculate_adsorption_energies(
            slab_results, adslab_results, gas_energies, args.adsorbate
        )

        print(f"✅ Calculated adsorption energies for {len(adsorption_results)} systems")

    # Save chunk results
    save_chunk_results(adslab_results, adsorption_results, args.output_dir,
                      args.adsorbate, args.chunk_id)

    print(f"\n✅ Chunk {args.chunk_id} completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()