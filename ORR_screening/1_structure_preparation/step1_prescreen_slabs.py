#!/usr/bin/env python3
"""
Step 1: Slab Prescreening with GemNet-OC (SINGLE WORKER PER GPU)

NO MULTIPROCESSING - Each GPU processes bulks sequentially with ONE model loaded.
This is the simplest, most stable approach for GPU workloads.

Key difference: No Pool, no workers - just load model once and process bulks in a loop.
"""

import os
import sys
import pickle
import logging
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from ase import Atoms
import torch

# Add the Open-Catalyst-Dataset directory to the path
sys.path.insert(0, 'Open-Catalyst-Dataset')

from ocdata.core import Bulk, Slab
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('slab_prescreening.log'),
        logging.StreamHandler()
    ]
)


def prepare_atoms_simple(atoms):
    """Simple atom preparation for energy calculation"""
    atoms_copy = atoms.copy()
    atoms_copy.set_tags([1] * len(atoms_copy))

    if not atoms_copy.pbc.any():
        atoms_copy.set_pbc([True, True, True])

    return atoms_copy


def calculate_slab_surface_energy(slab_atoms, bulk_energy_per_atom, calculator):
    """
    Calculate surface energy for a slab.

    Surface energy = (E_slab - N_atoms * E_bulk_per_atom) / (2 * Area)
    """
    import time

    try:
        atoms_prepared = prepare_atoms_simple(slab_atoms)
        atoms_prepared.calc = calculator

        start_time = time.time()
        slab_energy = atoms_prepared.get_potential_energy()
        end_time = time.time()

        calculation_time = end_time - start_time

        # Surface energy calculation
        n_atoms = len(slab_atoms)
        cell = slab_atoms.get_cell()
        area = np.linalg.norm(np.cross(cell[0], cell[1]))  # Surface area

        # Surface energy (two surfaces)
        surface_energy = (slab_energy - n_atoms * bulk_energy_per_atom) / (2 * area)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return surface_energy, slab_energy, calculation_time

    except Exception as e:
        logging.warning(f"Surface energy calculation failed: {e}")
        return None, None, None


def calculate_bulk_energy_per_atom(bulk_atoms, calculator):
    """Calculate bulk energy per atom using GemNet-OC"""
    try:
        atoms_prepared = prepare_atoms_simple(bulk_atoms)
        atoms_prepared.calc = calculator

        bulk_energy = atoms_prepared.get_potential_energy()
        energy_per_atom = bulk_energy / len(bulk_atoms)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return energy_per_atom

    except Exception as e:
        logging.warning(f"Bulk energy calculation failed: {e}")
        return None


def process_single_bulk_prescreen(bulk_idx, bulk_data, output_dir, max_miller,
                                  top_n_slabs, nelement_cutoff, calculator):
    """
    Process a single bulk to generate and prescreen all slabs.

    NO MULTIPROCESSING - uses pre-loaded calculator.
    """

    try:
        # Check element count cutoff
        unique_elements = set(bulk_data['atoms'].get_chemical_symbols())
        if len(unique_elements) > nelement_cutoff:
            logging.info(f"Skipping bulk {bulk_idx}: {bulk_data['src_id']} "
                        f"- has {len(unique_elements)} elements (cutoff={nelement_cutoff})")
            return {
                'bulk_idx': bulk_idx,
                'bulk_id': bulk_data['src_id'],
                'status': 'skipped_elements',
                'reason': f'Too many elements ({len(unique_elements)} > {nelement_cutoff})',
                'slabs_generated': 0,
                'slabs_saved': 0
            }

        # Create Bulk object
        bulk = Bulk(
            bulk_atoms=bulk_data['atoms'],
            bulk_id_from_db=bulk_idx
        )
        bulk.src_id = bulk_data['src_id']

        formula = bulk.atoms.get_chemical_formula()
        logging.info(f"Processing bulk {bulk_idx}: {bulk.src_id} ({formula})")

        # Calculate bulk energy per atom
        bulk_energy_per_atom = calculate_bulk_energy_per_atom(bulk.atoms, calculator)
        if bulk_energy_per_atom is None:
            logging.warning(f"Failed to calculate bulk energy for {bulk_idx}")
            return {
                'bulk_idx': bulk_idx,
                'bulk_id': bulk.src_id,
                'status': 'bulk_energy_failed',
                'slabs_generated': 0,
                'slabs_saved': 0
            }

        # Generate all slabs with specified max_miller
        try:
            slabs = bulk.get_slabs(max_miller=max_miller)
        except Exception as e:
            logging.warning(f"Failed to generate slabs for bulk {bulk_idx}: {e}")
            return {
                'bulk_idx': bulk_idx,
                'bulk_id': bulk.src_id,
                'status': 'slab_generation_failed',
                'error': str(e),
                'slabs_generated': 0,
                'slabs_saved': 0
            }

        num_slabs_generated = len(slabs)
        logging.info(f"Generated {num_slabs_generated} slabs for bulk {bulk_idx}")

        # Evaluate surface energy for all slabs
        slab_energies = []
        for slab_idx, slab in enumerate(slabs):
            surface_energy, slab_energy, calc_time = calculate_slab_surface_energy(
                slab.atoms, bulk_energy_per_atom, calculator
            )

            if surface_energy is not None:
                slab_energies.append({
                    'slab_idx': slab_idx,
                    'slab': slab,
                    'surface_energy': surface_energy,
                    'slab_energy': slab_energy,
                    'calc_time': calc_time,
                    'miller': slab.millers,
                    'shift': slab.shift,
                    'top': slab.top
                })

        if len(slab_energies) == 0:
            logging.warning(f"No valid slab energies for bulk {bulk_idx}")
            return {
                'bulk_idx': bulk_idx,
                'bulk_id': bulk.src_id,
                'status': 'no_valid_slabs',
                'slabs_generated': num_slabs_generated,
                'slabs_saved': 0
            }

        # Sort by surface energy (lowest = most stable)
        slab_energies.sort(key=lambda x: x['surface_energy'])

        # Select top N most stable slabs
        top_slabs = slab_energies[:top_n_slabs]
        num_slabs_saved = len(top_slabs)

        # Save to precomputed pkl format
        precomp_dir = Path(output_dir) / "precomputed_slabs"
        precomp_dir.mkdir(parents=True, exist_ok=True)

        precomp_pkl_path = precomp_dir / f"{bulk_idx}.pkl"

        # Prepare data structure for pkl
        slabs_to_save = [entry['slab'] for entry in top_slabs]

        # Save with metadata
        slab_data = {
            'bulk_id': bulk_idx,
            'bulk_src_id': bulk.src_id,
            'formula': formula,
            'slabs': slabs_to_save,
            'bulk_energy_per_atom': bulk_energy_per_atom,
            'slab_metadata': [
                {
                    'miller': entry['miller'],
                    'shift': entry['shift'],
                    'top': entry['top'],
                    'surface_energy': entry['surface_energy'],
                    'slab_energy': entry['slab_energy']
                }
                for entry in top_slabs
            ],
            'prescreening_info': {
                'total_slabs_generated': num_slabs_generated,
                'top_n_selected': num_slabs_saved,
                'max_miller': max_miller,
                'timestamp': datetime.now().isoformat()
            }
        }

        with open(precomp_pkl_path, 'wb') as f:
            pickle.dump(slab_data, f)

        logging.info(f"Saved top {num_slabs_saved}/{num_slabs_generated} slabs "
                    f"for bulk {bulk_idx} to {precomp_pkl_path}")

        # Log top 3 most stable slabs
        for i, entry in enumerate(top_slabs[:3]):
            logging.debug(f"  Rank {i+1}: Miller {entry['miller']}, "
                         f"Surface Energy: {entry['surface_energy']:.4f} eV/Å²")

        return {
            'bulk_idx': bulk_idx,
            'bulk_id': bulk.src_id,
            'status': 'success',
            'slabs_generated': num_slabs_generated,
            'slabs_saved': num_slabs_saved,
            'bulk_energy_per_atom': bulk_energy_per_atom,
            'best_surface_energy': top_slabs[0]['surface_energy'],
            'worst_surface_energy': top_slabs[-1]['surface_energy']
        }

    except Exception as e:
        logging.error(f"Error processing bulk {bulk_idx}: {e}")
        logging.error(traceback.format_exc())
        return {
            'bulk_idx': bulk_idx,
            'bulk_id': bulk_data.get('src_id', 'unknown'),
            'status': 'error',
            'error': str(e),
            'slabs_generated': 0,
            'slabs_saved': 0
        }


def sequential_prescreen_slabs(
    bulk_db_path='Open-Catalyst-Dataset/ocdata/databases/pkls/bulks.pkl',
    output_dir='co_structures_prescreened',
    max_miller=1,
    top_n_slabs=5,
    bulk_indices=None,
    nelement_cutoff=3,
    gemnet_checkpoint='/home/jsh9967/10_crack/4_re_on_0721/model/gnoc_oc22_oc20_all_s2ef.pt'
):
    """
    Sequential slab prescreening - NO MULTIPROCESSING.

    Load model ONCE, process all bulks sequentially.
    This is the most stable approach for GPU workloads.
    """

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load bulk database
    logging.info(f"Loading bulk database from {bulk_db_path}")
    with open(bulk_db_path, 'rb') as f:
        bulk_db = pickle.load(f)

    total_bulks = len(bulk_db)
    logging.info(f"Total bulks in database: {total_bulks}")

    # Determine which bulks to process
    if bulk_indices is None:
        bulk_indices = range(total_bulks)
    else:
        logging.info(f"Processing specific bulks: {bulk_indices}")

    num_bulks_to_process = len(bulk_indices)

    # Load GemNet-OC calculator ONCE
    logging.info(f"Loading GemNet-OC model from {gemnet_checkpoint}")
    try:
        calculator = OCPCalculator(checkpoint_path=gemnet_checkpoint, cpu=False)
        logging.info("✓ Model loaded successfully")
    except Exception as e:
        logging.error(f"✗ Failed to load model: {e}")
        return []

    # Process bulks sequentially
    results = []
    for idx_count, bulk_idx in enumerate(bulk_indices):
        bulk_data = bulk_db[bulk_idx]

        logging.info(f"Progress: {idx_count+1}/{num_bulks_to_process} bulks")

        result = process_single_bulk_prescreen(
            bulk_idx, bulk_data, output_dir, max_miller,
            top_n_slabs, nelement_cutoff, calculator
        )

        results.append(result)

        # Periodic progress update
        if (idx_count + 1) % 10 == 0:
            successful = len([r for r in results if r['status'] == 'success'])
            logging.info(f"Checkpoint: {idx_count+1}/{num_bulks_to_process} processed, "
                        f"{successful} successful")

    return results


def summarize_prescreening_results(results, output_file='prescreening_summary.txt'):
    """Summarize the prescreening results."""

    successful = [r for r in results if r['status'] == 'success']
    skipped = [r for r in results if r['status'] == 'skipped_elements']
    failed = [r for r in results if r['status'] not in ['success', 'skipped_elements']]

    total_slabs_generated = sum(r.get('slabs_generated', 0) for r in successful)
    total_slabs_saved = sum(r.get('slabs_saved', 0) for r in successful)

    summary = f"""
Slab Prescreening Summary (Single Worker)
================================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Total bulks processed: {len(results)}
Successful: {len(successful)}
Skipped (too many elements): {len(skipped)}
Failed: {len(failed)}

Total slabs generated: {total_slabs_generated}
Total slabs saved (prescreened): {total_slabs_saved}
Reduction rate: {100*(1 - total_slabs_saved/total_slabs_generated):.1f}%

Prescreening Statistics:
"""

    if successful:
        avg_slabs_gen = np.mean([r['slabs_generated'] for r in successful])
        avg_slabs_saved = np.mean([r['slabs_saved'] for r in successful])

        summary += f"  Average slabs generated per bulk: {avg_slabs_gen:.1f}\n"
        summary += f"  Average slabs saved per bulk: {avg_slabs_saved:.1f}\n"

        best_energies = [r['best_surface_energy'] for r in successful
                        if 'best_surface_energy' in r]
        if best_energies:
            summary += f"  Best surface energy range: {min(best_energies):.4f} to {max(best_energies):.4f} eV/Å²\n"

    summary += "\nSkipped bulks (element cutoff):\n"
    for r in skipped[:10]:
        summary += f"  - Bulk {r['bulk_idx']} ({r['bulk_id']}): {r.get('reason', '')}\n"
    if len(skipped) > 10:
        summary += f"  ... and {len(skipped)-10} more\n"

    summary += "\nFailed bulks:\n"
    for r in failed:
        summary += f"  - Bulk {r['bulk_idx']} ({r['bulk_id']}): {r.get('error', 'Unknown error')}\n"

    # Save summary to file
    with open(output_file, 'w') as f:
        f.write(summary)

    print(summary)

    # Save detailed results
    with open('prescreening_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"Summary saved to {output_file}")
    logging.info(f"Detailed results saved to prescreening_results.pkl")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Step 1: Slab Prescreening (SINGLE WORKER)')
    parser.add_argument('--start_idx', type=int, default=None,
                       help='Start bulk index (for GPU splitting)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End bulk index (exclusive, for GPU splitting)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID for identification')
    parser.add_argument('--max_miller', type=int, default=1,
                       help='Maximum Miller index')
    parser.add_argument('--top_n_slabs', type=int, default=5,
                       help='Number of best slabs to keep per bulk')
    parser.add_argument('--nelement_cutoff', type=int, default=3,
                       help='Maximum number of unique elements')

    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Slab Prescreening (SINGLE WORKER - NO MULTIPROCESSING)")
    print("=" * 60)

    # Configuration
    config = {
        'bulk_db_path': 'Open-Catalyst-Dataset/ocdata/databases/pkls/bulks.pkl',
        'output_dir': 'co_structures_prescreened',
        'max_miller': args.max_miller,
        'top_n_slabs': args.top_n_slabs,
        'bulk_indices': None,
        'nelement_cutoff': args.nelement_cutoff,
        'gemnet_checkpoint': '/home/jsh9967/10_crack/4_re_on_0721/model/gnoc_oc22_oc20_all_s2ef.pt'
    }

    # Set bulk indices based on start/end arguments
    if args.start_idx is not None and args.end_idx is not None:
        config['bulk_indices'] = range(args.start_idx, args.end_idx)
        print(f"\nGPU {args.gpu_id}: Processing bulks {args.start_idx} to {args.end_idx-1}")
        print(f"Total bulks to process: {args.end_idx - args.start_idx}")

    print("\nConfiguration:")
    for key, value in config.items():
        if key == 'bulk_indices':
            if value is not None:
                print(f"  {key}: range({value.start}, {value.stop})")
            else:
                print(f"  {key}: ALL")
        else:
            print(f"  {key}: {value}")

    print("\nNO MULTIPROCESSING: Model loaded once, bulks processed sequentially")
    print("Starting slab prescreening...")
    start_time = datetime.now()

    # Run sequential prescreening
    results = sequential_prescreen_slabs(**config)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\nPrescreening completed in {duration}")

    # Summarize results with GPU-specific filename
    if args.start_idx is not None:
        summary_file = f'prescreening_summary_gpu{args.gpu_id}.txt'
    else:
        summary_file = 'prescreening_summary.txt'

    summarize_prescreening_results(results, output_file=summary_file)


if __name__ == "__main__":
    main()
