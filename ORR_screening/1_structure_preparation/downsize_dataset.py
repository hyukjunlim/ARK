#!/usr/bin/env python3
"""
Script to downsize CO adslab dataset:
1. Randomly select 10 frames from each trajectory file
2. Extract C and O indices from JSON files to a single text file
"""

import json
import random
from pathlib import Path
from ase.io import read, write
from tqdm import tqdm

def process_dataset(data_dir, output_dir, co_indices_file, num_frames=10, seed=42):
    """
    Process dataset by downsizing trajectories and extracting CO indices.

    Args:
        data_dir: Path to input directory with traj and json files
        output_dir: Path to output directory for downsized trajs
        co_indices_file: Path to output text file for C and O indices
        num_frames: Number of frames to randomly select per trajectory
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get all trajectory files
    traj_files = sorted(data_path.glob("*.traj"))
    print(f"Found {len(traj_files)} trajectory files")

    # Lists to store C and O indices
    co_indices_list = []

    # Process each trajectory file
    for traj_file in tqdm(traj_files, desc="Processing trajectories"):
        base_name = traj_file.stem  # e.g., "mp-1194450_111_CO"
        json_file = data_path / f"{base_name}.json"

        # Check if corresponding JSON exists
        if not json_file.exists():
            print(f"Warning: JSON file not found for {traj_file.name}, skipping...")
            continue

        # Read trajectory
        try:
            atoms_list = read(str(traj_file), ":")
            total_frames = len(atoms_list)

            # Read JSON to get C and O indices
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract C and O indices from first configuration
            # Note: In the JSON, "C" key has index for C, "O" key has index for O
            c_index = data["configurations"][0]["adsorbate"]["C"]["index"]
            o_index = data["configurations"][0]["adsorbate"]["O"]["index"]

            # Store indices: format "C_index O_index"
            co_indices_list.append(f"{c_index} {o_index}")

            # Randomly select frames
            if total_frames <= num_frames:
                # If fewer frames than requested, use all
                selected_frames = list(range(total_frames))
            else:
                # Randomly select num_frames
                selected_frames = sorted(random.sample(range(total_frames), num_frames))

            # Extract selected frames
            selected_atoms = [atoms_list[i] for i in selected_frames]

            # Write downsized trajectory
            output_traj = output_path / traj_file.name
            write(str(output_traj), selected_atoms)

        except Exception as e:
            print(f"Error processing {traj_file.name}: {e}")
            continue

    # Write CO indices to text file
    with open(co_indices_file, 'w') as f:
        for line in co_indices_list:
            f.write(line + '\n')

    print(f"\nProcessing complete!")
    print(f"Downsized trajectories saved to: {output_path}")
    print(f"CO indices saved to: {co_indices_file}")
    print(f"Total processed: {len(co_indices_list)} files")

if __name__ == "__main__":
    # Configuration
    data_dir = "data_co_adslab"
    output_dir = "data_co_adslab_downsized"
    co_indices_file = "co_indices.txt"
    num_frames = 10

    # Process dataset
    process_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        co_indices_file=co_indices_file,
        num_frames=num_frames,
        seed=42
    )
