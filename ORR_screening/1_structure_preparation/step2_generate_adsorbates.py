#!/usr/bin/env python3
"""
Step 2: Generate CO Adsorbate Structures from Prescreened Slabs

Uses the most stable slab (rank 0) from each precomputed bulk pkl file.
Generates CO adsorption configurations for each top stable surface.
"""

import os
import sys
import pickle
import logging
import traceback
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
from datetime import datetime
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import write, Trajectory

# Add the Open-Catalyst-Dataset directory to the path
sys.path.insert(0, 'Open-Catalyst-Dataset')

from ocdata.core import Bulk, Slab, Adsorbate, AdsorbateSlabConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('co_generation_prescreened.log'),
        logging.StreamHandler()
    ]
)


def create_co_adsorbate():
    """Create a CO molecule adsorbate."""
    co_molecule = molecule('CO')
    co_adsorbate = Adsorbate(
        adsorbate_atoms=co_molecule,
        adsorbate_binding_indices=[0]  # Carbon atom binds to surface
    )
    return co_adsorbate


def extract_adsorbate_info(atoms_config, num_slab_atoms):
    """
    Extract CO adsorbate information from the configuration.

    Args:
        atoms_config: ASE Atoms object with slab + adsorbate
        num_slab_atoms: Number of atoms in the slab (to identify adsorbate atoms)

    Returns:
        dict with C and O positions and indices
    """
    # Adsorbate atoms are the last atoms added after the slab
    adsorbate_indices = list(range(num_slab_atoms, len(atoms_config)))

    # CO molecule: first adsorbate atom is C, second is O
    c_idx = adsorbate_indices[0]
    o_idx = adsorbate_indices[1]

    c_position = atoms_config.positions[c_idx].tolist()
    o_position = atoms_config.positions[o_idx].tolist()

    return {
        'C': {
            'index': int(c_idx),
            'position': c_position,
            'symbol': atoms_config.symbols[c_idx]
        },
        'O': {
            'index': int(o_idx),
            'position': o_position,
            'symbol': atoms_config.symbols[o_idx]
        }
    }


def process_single_precomputed_slab(args):
    """
    Process a single precomputed bulk to generate CO adsorbate structures.
    Uses ONLY the most stable slab (index 0) from the prescreened pkl file.

    Args:
        args: tuple of (pkl_path, output_dir, num_sites, mode)

    Returns:
        dict with processing results
    """
    pkl_path, output_dir, num_sites, mode = args

    process_id = current_process().name
    bulk_idx = pkl_path.stem  # Extract bulk_idx from filename (e.g., "0" from "0.pkl")

    try:
        # Load precomputed slab data
        with open(pkl_path, 'rb') as f:
            slab_data = pickle.load(f)

        bulk_id = slab_data['bulk_src_id']
        formula = slab_data['formula']
        slabs = slab_data['slabs']
        slab_metadata = slab_data['slab_metadata']

        # Validate that we have slabs
        if len(slabs) == 0:
            logging.warning(f"[{process_id}] No slabs found in {pkl_path}")
            return {
                'bulk_idx': bulk_idx,
                'bulk_id': bulk_id,
                'status': 'no_slabs',
                'traj_files': []
            }

        # Take ONLY the most stable slab (index 0)
        most_stable_slab = slabs[0]
        most_stable_metadata = slab_metadata[0]

        logging.info(f"[{process_id}] Processing bulk {bulk_idx}: {bulk_id} ({formula})")
        logging.info(f"[{process_id}]   Most stable slab - Miller: {most_stable_metadata['miller']}, "
                    f"Surface Energy: {most_stable_metadata['surface_energy']:.4f} eV/Å²")

        # Create CO adsorbate
        co_adsorbate = create_co_adsorbate()

        # Process the most stable slab
        miller_str = f"{most_stable_metadata['miller'][0]}{most_stable_metadata['miller'][1]}{most_stable_metadata['miller'][2]}"

        # Create filenames for this slab-miller-adsorbate combination
        # Format: bulkID_miller_CO.traj and bulkID_miller_CO.json
        base_filename = f"{bulk_id}_{miller_str}_CO"
        traj_filename = f"{base_filename}.traj"
        json_filename = f"{base_filename}.json"
        traj_path = Path(output_dir) / traj_filename
        json_path = Path(output_dir) / json_filename

        try:
            # Generate adsorbate-slab configurations (no limit on num_sites)
            config = AdsorbateSlabConfig(
                slab=most_stable_slab,
                adsorbate=co_adsorbate,
                num_sites=num_sites,  # This is max sites for heuristic to try
                num_augmentations_per_site=1,
                interstitial_gap=0.1,
                mode=mode
            )

            # Use all generated configurations (no artificial limit)
            configs_to_save = config.atoms_list
            num_slab_atoms = len(most_stable_slab)

            # Prepare JSON data structure
            json_data = {
                'bulk_id': bulk_id,
                'bulk_idx': bulk_idx,
                'formula': formula,
                'miller': most_stable_metadata['miller'],
                'shift': most_stable_metadata['shift'],
                'top': most_stable_metadata['top'],
                'surface_energy': most_stable_metadata['surface_energy'],
                'slab_energy': most_stable_metadata['slab_energy'],
                'num_slab_atoms': num_slab_atoms,
                'total_configs': len(configs_to_save),
                'configurations': []
            }

            # Open trajectory file for this slab-miller combination
            with Trajectory(str(traj_path), 'w') as traj:
                # Write all configurations to the trajectory
                for config_idx, atoms_config in enumerate(configs_to_save):
                    # Add metadata as info
                    atoms_config.info = {
                        'bulk_id': bulk_id,
                        'bulk_idx': bulk_idx,
                        'miller': most_stable_metadata['miller'],
                        'shift': most_stable_metadata['shift'],
                        'top': most_stable_metadata['top'],
                        'adsorbate': 'CO',
                        'surface_energy': most_stable_metadata['surface_energy'],
                        'slab_energy': most_stable_metadata['slab_energy'],
                        'config_idx': config_idx
                    }
                    traj.write(atoms_config)

                    # Extract adsorbate information for JSON
                    adsorbate_info = extract_adsorbate_info(atoms_config, num_slab_atoms)
                    json_data['configurations'].append({
                        'config_idx': config_idx,
                        'adsorbate': adsorbate_info
                    })

            # Save JSON file with adsorbate information
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            num_configs = len(configs_to_save)

            logging.info(f"[{process_id}] Completed bulk {bulk_idx}: {num_configs} configs "
                        f"saved to {traj_filename} and {json_filename}")

            return {
                'bulk_idx': bulk_idx,
                'bulk_id': bulk_id,
                'status': 'success',
                'total_configs': num_configs,
                'traj_files': [traj_filename],
                'json_files': [json_filename],
                'miller': most_stable_metadata['miller'],
                'surface_energy': most_stable_metadata['surface_energy']
            }

        except Exception as e:
            logging.warning(f"[{process_id}] Failed to generate configs for bulk {bulk_idx}: {e}")
            return {
                'bulk_idx': bulk_idx,
                'bulk_id': bulk_id,
                'status': 'config_generation_failed',
                'error': str(e),
                'traj_files': []
            }

    except Exception as e:
        logging.error(f"[{process_id}] Error processing {pkl_path}: {e}")
        logging.error(traceback.format_exc())
        return {
            'bulk_idx': bulk_idx,
            'bulk_id': 'unknown',
            'status': 'error',
            'error': str(e),
            'traj_files': []
        }


def parallel_generate_co_from_prescreened(
    precomputed_dir='co_structures_prescreened/precomputed_slabs',
    output_dir='co_structures_from_prescreened',
    num_sites=10,
    mode='heuristic',
    num_workers=None,
    chunk_size=1,
    bulk_indices=None,
    max_bulks=None
):
    """
    Generate CO structures from prescreened slabs in parallel.
    Uses ONLY the most stable slab from each bulk.

    Args:
        precomputed_dir: Directory containing precomputed slab pkl files
        output_dir: Directory to save trajectory files
        num_sites: Maximum number of adsorption sites per slab
        mode: Adsorption mode ('random', 'heuristic', 'random_site_heuristic_placement')
        num_workers: Number of parallel workers (default: number of CPU cores)
        chunk_size: Number of bulks to process per worker at once
        bulk_indices: List of specific bulk indices to process (None = all)
        max_bulks: Maximum number of bulks to process (processes first N files)

    Returns:
        List of result dictionaries from processing
    """

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all precomputed pkl files
    precomputed_path = Path(precomputed_dir)
    if not precomputed_path.exists():
        logging.error(f"Precomputed directory not found: {precomputed_dir}")
        return []

    pkl_files = sorted(precomputed_path.glob('*.pkl'))

    # Filter by bulk_indices if specified
    if bulk_indices is not None:
        bulk_indices_set = set(str(idx) for idx in bulk_indices)
        pkl_files = [f for f in pkl_files if f.stem in bulk_indices_set]

    # Limit to max_bulks if specified
    if max_bulks is not None and max_bulks > 0:
        pkl_files = pkl_files[:max_bulks]
        logging.info(f"Limiting to first {max_bulks} bulks")

    total_files = len(pkl_files)
    logging.info(f"Found {total_files} precomputed slab files to process")

    if total_files == 0:
        logging.warning("No precomputed slab files found!")
        return []

    # Prepare arguments for parallel processing
    args_list = [
        (pkl_path, output_dir, num_sites, mode)
        for pkl_path in pkl_files
    ]

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    logging.info(f"Using {num_workers} parallel workers")

    # Process bulks in parallel
    results = []
    with Pool(processes=num_workers) as pool:
        # Use imap for better progress tracking
        for result in pool.imap_unordered(process_single_precomputed_slab, args_list, chunksize=chunk_size):
            results.append(result)

            # Progress update
            completed = len(results)
            if completed % 10 == 0:
                logging.info(f"Progress: {completed}/{len(args_list)} bulks processed")

    return results


def summarize_results(results, output_file='generation_summary_prescreened.txt'):
    """
    Summarize the generation results.

    Args:
        results: List of result dictionaries
        output_file: File to save summary
    """

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    total_configs = sum(r.get('total_configs', 0) for r in successful)
    total_traj_files = sum(len(r.get('traj_files', [])) for r in successful)
    total_json_files = sum(len(r.get('json_files', [])) for r in successful)

    summary = f"""
CO Structure Generation Summary (From Prescreened Slabs)
========================================================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Strategy: Most stable slab only (rank 0) from each bulk
Note: All adsorption sites generated (no artificial limit)

Total bulks processed: {len(results)}
Successful: {len(successful)}
Failed: {len(failed)}

Total configurations generated: {total_configs}
Total trajectory files created: {total_traj_files}
Total JSON files created: {total_json_files}
Average configs per bulk: {total_configs/len(successful) if successful else 0:.1f}

"""

    if successful:
        surface_energies = [r['surface_energy'] for r in successful if 'surface_energy' in r]
        if surface_energies:
            summary += f"\nSurface Energy Statistics:\n"
            summary += f"  Range: {min(surface_energies):.4f} to {max(surface_energies):.4f} eV/Å²\n"
            summary += f"  Mean: {np.mean(surface_energies):.4f} eV/Å²\n"
            summary += f"  Median: {np.median(surface_energies):.4f} eV/Å²\n"

    summary += "\nFailed bulks:\n"
    for r in failed:
        summary += f"  - Bulk {r['bulk_idx']} ({r['bulk_id']}): {r.get('error', 'Unknown error')}\n"

    # Save summary to file
    with open(output_file, 'w') as f:
        f.write(summary)

    print(summary)

    # Save detailed results
    with open('generation_results_prescreened.pkl', 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"Summary saved to {output_file}")
    logging.info(f"Detailed results saved to generation_results_prescreened.pkl")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Step 2: Generate CO from Prescreened Slabs')
    parser.add_argument('--precomputed_dir', type=str,
                       default='co_structures_prescreened/precomputed_slabs',
                       help='Directory containing precomputed slab pkl files')
    parser.add_argument('--output_dir', type=str,
                       default='co_structures_from_prescreened',
                       help='Output directory for trajectory files')
    parser.add_argument('--num_sites', type=int, default=200,
                       help='Maximum number of adsorption sites to attempt (heuristic will find optimal sites)')
    parser.add_argument('--mode', type=str, default='heuristic',
                       choices=['random', 'heuristic', 'random_site_heuristic_placement'],
                       help='Adsorption site selection mode')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPU cores)')
    parser.add_argument('--chunk_size', type=int, default=1,
                       help='Chunk size for parallel processing')
    parser.add_argument('--bulk_indices', type=int, nargs='+', default=None,
                       help='Specific bulk indices to process (optional)')
    parser.add_argument('--max_bulks', type=int, default=None,
                       help='Maximum number of bulks to process (processes first N files)')

    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: CO Adsorbate Generation from Prescreened Slabs")
    print("=" * 60)
    print("\nStrategy: Use ONLY the most stable slab (rank 0) per bulk")

    # Configuration
    config = {
        'precomputed_dir': args.precomputed_dir,
        'output_dir': args.output_dir,
        'num_sites': args.num_sites,
        'mode': args.mode,
        'num_workers': args.num_workers if args.num_workers else cpu_count(),
        'chunk_size': args.chunk_size,
        'bulk_indices': args.bulk_indices,
        'max_bulks': args.max_bulks
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nStarting parallel generation...")
    start_time = datetime.now()

    # Run parallel generation
    results = parallel_generate_co_from_prescreened(**config)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\nGeneration completed in {duration}")

    # Summarize results
    summarize_results(results)


if __name__ == "__main__":
    main()
