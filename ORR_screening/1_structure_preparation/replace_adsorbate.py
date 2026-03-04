#!/usr/bin/env python3
"""
General-purpose script to replace CO adsorbate with any other adsorbate.
Supports single atoms (e.g., H), diatomic molecules (e.g., NO, H2), and more complex adsorbates.
"""

from pathlib import Path
from ase.io import read, write
from ase import Atoms
from tqdm import tqdm
import argparse

# Predefined adsorbate configurations
ADSORBATE_CONFIGS = {
    # Diatomic molecules (replace C and O)
    'NO': {'symbols': ['N', 'O'], 'replace_indices': [0, 1]},
    'CO': {'symbols': ['C', 'O'], 'replace_indices': [0, 1]},
    'OH': {'symbols': ['O', 'H'], 'replace_indices': [0, 1]},
    'H2': {'symbols': ['H', 'H'], 'replace_indices': [0, 1]},
    'N2': {'symbols': ['N', 'N'], 'replace_indices': [0, 1]},
    'O2': {'symbols': ['O', 'O'], 'replace_indices': [0, 1]},

    # Single atoms (replace C, remove O)
    'H': {'symbols': ['H'], 'replace_indices': [0], 'remove_indices': [1]},
    'N': {'symbols': ['N'], 'replace_indices': [0], 'remove_indices': [1]},
    'O': {'symbols': ['O'], 'replace_indices': [0], 'remove_indices': [1]},
    'C': {'symbols': ['C'], 'replace_indices': [0], 'remove_indices': [1]},
    'S': {'symbols': ['S'], 'replace_indices': [0], 'remove_indices': [1]},

    # Triatomic molecules (replace C and O, add third atom)
    'H2O': {'symbols': ['O', 'H', 'H'], 'replace_indices': [0, 1], 'add_atoms': [('H', [0, 0, 1.0])]},
    'CO2': {'symbols': ['C', 'O', 'O'], 'replace_indices': [0, 1], 'add_atoms': [('O', [0, 0, -1.16])]},
    'NO2': {'symbols': ['N', 'O', 'O'], 'replace_indices': [0, 1], 'add_atoms': [('O', [0, 0, -1.2])]},
}

def replace_adsorbate(input_dir, co_indices_file, output_dir, target_adsorbate, config=None):
    """
    Replace CO adsorbate with target adsorbate in all trajectory files.

    Args:
        input_dir: Directory with downsized CO trajectories
        co_indices_file: Text file with C and O indices
        output_dir: Directory to save modified trajectories
        target_adsorbate: Name of target adsorbate (e.g., 'NO', 'H2O')
        config: Custom configuration dict (optional, uses ADSORBATE_CONFIGS if None)
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get adsorbate configuration
    if config is None:
        if target_adsorbate not in ADSORBATE_CONFIGS:
            raise ValueError(f"Unknown adsorbate: {target_adsorbate}. "
                           f"Available: {list(ADSORBATE_CONFIGS.keys())}")
        config = ADSORBATE_CONFIGS[target_adsorbate]

    # Read CO indices
    with open(co_indices_file, 'r') as f:
        co_indices_list = [tuple(map(int, line.strip().split())) for line in f]

    # Get all trajectory files (sorted for consistency with indices)
    traj_files = sorted(input_path.glob("*.traj"))

    print(f"Replacing CO with {target_adsorbate}")
    print(f"Configuration: {config}")
    print(f"Found {len(traj_files)} trajectory files")
    print(f"Loaded {len(co_indices_list)} CO index pairs")

    if len(traj_files) != len(co_indices_list):
        print(f"Warning: Number of trajectories ({len(traj_files)}) != number of index pairs ({len(co_indices_list)})")
        n_files = min(len(traj_files), len(co_indices_list))
    else:
        n_files = len(traj_files)

    # Process each trajectory
    successful = 0
    failed = 0

    for traj_file, (c_idx, o_idx) in tqdm(
        zip(traj_files[:n_files], co_indices_list[:n_files]),
        total=n_files,
        desc=f"Processing"
    ):
        try:
            # Read all frames from trajectory
            atoms_list = read(str(traj_file), ":")

            # Process each frame
            modified_frames = []
            for atoms in atoms_list:
                # Determine actual C and O positions (JSON labels may be swapped)
                atom_at_c_idx = atoms[c_idx].symbol
                atom_at_o_idx = atoms[o_idx].symbol

                # Find actual C and O indices
                if atom_at_c_idx == 'C' and atom_at_o_idx == 'O':
                    # JSON is correct
                    actual_c_idx, actual_o_idx = c_idx, o_idx
                elif atom_at_c_idx == 'O' and atom_at_o_idx == 'C':
                    # JSON labels are swapped
                    actual_c_idx, actual_o_idx = o_idx, c_idx
                else:
                    print(f"\nWarning: {traj_file.name} - Unexpected atoms: idx {c_idx}={atom_at_c_idx}, idx {o_idx}={atom_at_o_idx}")
                    # Try to proceed anyway
                    actual_c_idx, actual_o_idx = c_idx, o_idx

                # Get CO positions
                c_pos = atoms[actual_c_idx].position.copy()
                o_pos = atoms[actual_o_idx].position.copy()

                # Use actual indices for replacements
                c_idx_temp, o_idx_temp = actual_c_idx, actual_o_idx

                # Handle different replacement strategies
                if 'remove_indices' in config:
                    # Remove specified atoms (for single-atom adsorbates)
                    indices_to_remove = []
                    if 0 in config['remove_indices']:
                        indices_to_remove.append(c_idx_temp)
                    if 1 in config['remove_indices']:
                        indices_to_remove.append(o_idx_temp)

                    # Replace atoms that are not being removed
                    for i, replace_idx in enumerate(config['replace_indices']):
                        if replace_idx == 0 and 0 not in config.get('remove_indices', []):
                            atoms[c_idx_temp].symbol = config['symbols'][i]
                        elif replace_idx == 1 and 1 not in config.get('remove_indices', []):
                            atoms[o_idx_temp].symbol = config['symbols'][i]

                    # Remove atoms (in reverse order to maintain indices)
                    for idx in sorted(indices_to_remove, reverse=True):
                        del atoms[idx]
                else:
                    # Simple replacement (most common case)
                    for i, replace_idx in enumerate(config['replace_indices']):
                        if replace_idx == 0:
                            atoms[c_idx_temp].symbol = config['symbols'][i]
                        elif replace_idx == 1:
                            atoms[o_idx_temp].symbol = config['symbols'][i]

                # Add additional atoms if specified
                if 'add_atoms' in config:
                    for symbol, offset in config['add_atoms']:
                        new_pos = o_pos + offset
                        atoms.append(Atoms(symbol, positions=[new_pos]))

                modified_frames.append(atoms)

            # Generate output filename
            output_name = traj_file.name.replace("_CO.traj", f"_{target_adsorbate}.traj")
            output_file = output_path / output_name

            # Write modified trajectory
            write(str(output_file), modified_frames)
            successful += 1

        except Exception as e:
            print(f"\nError processing {traj_file.name}: {e}")
            failed += 1
            continue

    print(f"\nReplacement complete!")
    print(f"Output directory: {output_path}")
    print(f"Successful: {successful} files")
    print(f"Failed: {failed} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace CO adsorbate with another adsorbate')
    parser.add_argument('--adsorbate', type=str, default='NO',
                       help=f'Target adsorbate (choices: {list(ADSORBATE_CONFIGS.keys())})')
    parser.add_argument('--input-dir', type=str, default='data_co_adslab_downsized',
                       help='Input directory with CO trajectories')
    parser.add_argument('--indices-file', type=str, default='co_indices.txt',
                       help='File with CO indices')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: data_{adsorbate}_adslab)')

    args = parser.parse_args()

    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"data_{args.adsorbate.lower()}_adslab"

    # Replace adsorbate
    replace_adsorbate(
        input_dir=args.input_dir,
        co_indices_file=args.indices_file,
        output_dir=args.output_dir,
        target_adsorbate=args.adsorbate
    )
