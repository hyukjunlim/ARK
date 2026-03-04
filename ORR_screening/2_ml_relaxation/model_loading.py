import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from ase.io import read
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase import Atoms
from tqdm import tqdm
import sys
from ase.calculators.calculator import Calculator
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import data_list_collater

class CrackCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, checkpoint_path, device="cuda"):
        Calculator.__init__(self)
        
        # Load checkpoint and get the model config
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']['model']
        
        print(f"Model name in config: {config.get('name', 'unknown')}")
        
        # Try to temporarily register the crack model as gemnet_oc
        try:
            from fairchem.core.common.relaxation.ase_utils import OCPCalculator
            
            # Temporarily modify the config to use gemnet_oc
            modified_checkpoint = checkpoint.copy()
            modified_config = config.copy()
            modified_config['name'] = 'gemnet_oc'  # Trick OCPCalculator
            modified_config.pop('scale_file', None)  # Remove scale file requirement
            modified_config.pop('teacher_model_args', None)  # Remove teacher args
            modified_checkpoint['config']['model'] = modified_config
            
            print("Attempting to load with modified config (no scale file)...")
            
            # Save temporary checkpoint
            temp_path = checkpoint_path.replace('.pt', '_temp_gemnet.pt')
            torch.save(modified_checkpoint, temp_path)
            
            # Try loading with OCPCalculator using modified checkpoint
            self._calc = OCPCalculator(checkpoint_path=temp_path, cpu=False)
            
            # Clean up temp file
            import os
            os.remove(temp_path)
            
            print("✅ Successfully loaded crack model as gemnet_oc")
            
            # Check parameter count
            if hasattr(self._calc, 'trainer') and hasattr(self._calc.trainer, 'model'):
                total_params = sum(p.numel() for p in self._calc.trainer.model.parameters())
                print(f"Model loaded with {total_params:,} parameters")
                
                # Check if we got close to the expected parameter count
                expected_params = 1157068
                if abs(total_params - expected_params) < 100000:  # Within 100k
                    print(f"✅ Parameter count reasonable (expected ~{expected_params:,})")
                else:
                    print(f"⚠️ Parameter count mismatch: got {total_params:,}, expected ~{expected_params:,}")
            
            return
            
        except Exception as e:
            print(f"Modified OCPCalculator loading failed: {e}")
            import traceback
            traceback.print_exc()
        
        # If that fails, mark calculator as unavailable
        print("❌ Crack model loading failed - calculator unavailable")
        self._calc = None
    
    def calculate(self, atoms, properties=None, system_changes=None):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        if self._calc is not None:
            try:
                atoms.calc = self._calc
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                
                self.results = {
                    'energy': float(energy),
                    'forces': forces
                }
                return self.results
                
            except Exception as e:
                # Log the actual error for debugging
                print(f"Crack calculation failed for {atoms.get_chemical_formula()}: {str(e)}")
                # Instead of dummy values, raise the exception to indicate failure
                raise RuntimeError(f"Crack model calculation failed: {str(e)}")
        
        # If no calculator is available, raise an exception
        raise RuntimeError("Crack calculator not available - model failed to load")

def extract_dft_energy(atoms):
    """Extract DFT energy from atoms object"""
    if hasattr(atoms, 'info'):
        # Common keys for DFT energy
        for key in ['energy', 'dft_energy', 'total_energy', 'Energy', 'DFT_energy']:
            if key in atoms.info:
                return float(atoms.info[key])
    
    # Try from calculator results
    if hasattr(atoms, 'get_potential_energy'):
        try:
            return float(atoms.get_potential_energy())
        except:
            pass
    
    return None


def setup_atom_tags(atoms):
    """Setup atom tags for OCP models"""
    if not hasattr(atoms, 'tags') or atoms.tags is None:
        atoms.tags = np.zeros(len(atoms), dtype=int)
    
    # Set surface tags (assuming slab geometry)
    if len(atoms) > 10:  # Only for larger systems
        z_coords = atoms.positions[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_range = z_max - z_min
        
        if z_range > 5.0:  # If significant z-variation
            # Bottom 20% and top 20% are surface atoms (tag=1)
            # Middle 60% are subsurface atoms (tag=0)
            surface_threshold_low = z_min + 0.2 * z_range
            surface_threshold_high = z_max - 0.2 * z_range
            
            atoms.tags = np.where(
                (z_coords <= surface_threshold_low) | (z_coords >= surface_threshold_high),
                1, 0  # 1 for surface, 0 for subsurface
            )
    
    return atoms


def compare_teacher_student_models():
    """Compare teacher (GemNet-OC) vs student (Crack) models on 100 samples"""
    
    # GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.6)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model paths
    teacher_model = "/home/jsh9967/10_crack/4_re_on_0721/model/gnoc_oc22_oc20_all_s2ef.pt"
    student_model = "/home/jsh9967/10_crack/4_re_on_0721/model/crack.pt"
    
    # Test data path - use the 10-sample trajectory
    trajectory_path = "/home/jsh9967/10_crack/4_re_on_0721/test_data/test_structures_oads_rand100.traj"
    
    # Create results directory
    results_dir = "/home/jsh9967/10_crack/4_re_on_0721/inference_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading test trajectory from: {trajectory_path}")
    
    # Load test trajectory
    try:
        trajectory = read(trajectory_path, index=':')
        print(f"✓ Loaded {len(trajectory)} test structures")
    except Exception as e:
        print(f"✗ Error loading trajectory: {e}")
        return
    
    # Initialize calculators
    print("\n" + "="*80)
    print("INITIALIZING MODELS")
    print("="*80)
    torch.set_default_dtype(torch.float32)

    # Load student model (Crack) - using robust loader
    student_calc = None
    print("\n1. Loading Student Model (Crack)...")
    try:
        student_calc = CrackCalculator(
            checkpoint_path=student_model,
            device="cuda"
        )
        print("✅ Student model (Crack) loaded successfully")
    except Exception as e:
        print(f"❌ Error loading student model: {e}")
        import traceback
        traceback.print_exc()
        print("\nCannot proceed without student model. Exiting.")
        return

    # Load teacher model (GemNet-OC) - using OCPCalculator
    teacher_calc = None
    print("\n2. Loading Teacher Model (GemNet-OC)...")
    try:
        teacher_calc = OCPCalculator(
            checkpoint_path=teacher_model,
            cpu=False
        )
        print("✅ Teacher model (GemNet-OC) loaded successfully")
    except Exception as e:
        print(f"❌ Error loading teacher model: {e}")
        print("Will continue with student model only...")
        import traceback
        traceback.print_exc()
    
    # Quick test both models
    print("\n" + "="*80)
    print("TESTING MODELS")
    print("="*80)

    # Use first structure from trajectory instead of H2O
    test_atoms = trajectory[0].copy()  # Use actual structure from trajectory
    test_atoms = setup_atom_tags(test_atoms)

    # Ensure proper setup
    if not test_atoms.pbc.any():
        test_atoms.set_pbc([True, True, True])
    if test_atoms.cell.volume < 100:
        test_atoms.set_cell([20, 20, 20])

    print(f"Test structure: {test_atoms.get_chemical_formula()} ({len(test_atoms)} atoms)")

    # Test student model
    if student_calc is not None:
        try:
            test_atoms.calc = student_calc
            student_energy = test_atoms.get_potential_energy()
            student_forces = test_atoms.get_forces()
            print(f"✅ Student test: E={student_energy:.4f} eV, max_F={np.abs(student_forces).max():.4f}")
        except Exception as e:
            print(f"❌ Student test failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Test teacher model
    if teacher_calc is not None:
        try:
            test_atoms.calc = teacher_calc
            teacher_energy = test_atoms.get_potential_energy()
            teacher_forces = test_atoms.get_forces()
            print(f"✅ Teacher test: E={teacher_energy:.4f} eV, max_F={np.abs(teacher_forces).max():.4f}")
        except Exception as e:
            print(f"❌ Teacher test failed: {e}")
            teacher_calc = None
    
    # Storage for results
    results = {
        "dft_energies": [],
        "teacher_energies": [],  # GemNet-OC
        "student_energies": [],  # Crack
        "teacher_forces": [],
        "student_forces": [],
        "teacher_times": [],
        "student_times": [],
        "system_ids": [],
        "formulas": [],
        "num_atoms": [],
        "teacher_errors": [],
        "student_errors": []
    }
    
    # Process all structures
    print(f"\n" + "="*80)
    print(f"RUNNING INFERENCE ON {len(trajectory)} STRUCTURES")
    print("="*80)
    
    successful_count = 0
    teacher_failures = 0
    student_failures = 0
    
    pbar = tqdm(enumerate(trajectory), total=len(trajectory))
    
    for idx, atoms in pbar:
        pbar.set_description(f"Structure {idx}")
        
        try:
            # Setup atoms properly
            atoms = setup_atom_tags(atoms)
            
            # Ensure PBC and cell
            if not atoms.pbc.any() or atoms.cell.volume < 100:
                atoms.set_cell([20, 20, 20])
                atoms.set_pbc([True, True, True])
            
            # System info
            system_id = f"test_structure_{idx}"
            num_atoms = len(atoms)
            formula = atoms.get_chemical_formula()
            
            # Extract DFT energy if available
            dft_energy = extract_dft_energy(atoms)
            
            # Store basic info
            results["system_ids"].append(system_id)
            results["formulas"].append(formula)
            results["num_atoms"].append(num_atoms)
            results["dft_energies"].append(dft_energy if dft_energy is not None else np.nan)
            
            # Initialize success flags
            teacher_success = False
            student_success = False
            
            # Calculate with teacher model (GemNet-OC)
            if teacher_calc is not None:
                try:
                    atoms.calc = teacher_calc
                    start_time = time.time()
                    teacher_energy = float(atoms.get_potential_energy())
                    teacher_forces = atoms.get_forces()
                    teacher_time = time.time() - start_time
                    
                    results["teacher_energies"].append(teacher_energy)
                    results["teacher_forces"].append(teacher_forces.flatten())
                    results["teacher_times"].append(teacher_time)
                    results["teacher_errors"].append("")
                    
                    teacher_success = True
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    teacher_failures += 1
                    error_msg = str(e)[:100]
                    results["teacher_energies"].append(np.nan)
                    results["teacher_forces"].append(np.array([]))
                    results["teacher_times"].append(np.nan)
                    results["teacher_errors"].append(error_msg)
                    if teacher_failures <= 3:
                        print(f"\nTeacher error on structure {idx}: {error_msg}")
            else:
                results["teacher_energies"].append(np.nan)
                results["teacher_forces"].append(np.array([]))
                results["teacher_times"].append(np.nan)
                results["teacher_errors"].append("No teacher calculator")
            
            # Calculate with student model (Crack)
            try:
                atoms.calc = student_calc
                start_time = time.time()
                student_energy = float(atoms.get_potential_energy())
                student_forces = atoms.get_forces()
                student_time = time.time() - start_time
                
                results["student_energies"].append(student_energy)
                results["student_forces"].append(student_forces.flatten())
                results["student_times"].append(student_time)
                results["student_errors"].append("")
                
                student_success = True
                torch.cuda.empty_cache()
                
            except Exception as e:
                student_failures += 1
                error_msg = str(e)[:100]
                results["student_energies"].append(np.nan)
                results["student_forces"].append(np.array([]))
                results["student_times"].append(np.nan)
                results["student_errors"].append(error_msg)
                if student_failures <= 3:
                    print(f"\nStudent error on structure {idx}: {error_msg}")
            
            if teacher_success or student_success:
                successful_count += 1
                
            # Update progress
            postfix_dict = {
                "success": successful_count,
                "T_fail": teacher_failures,
                "S_fail": student_failures
            }
            if teacher_success:
                postfix_dict["T_time"] = f"{teacher_time:.3f}s"
            if student_success:
                postfix_dict["S_time"] = f"{student_time:.3f}s"
            
            pbar.set_postfix(postfix_dict)
            
        except Exception as e:
            print(f"\nGeneral error processing structure {idx}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"INFERENCE SUMMARY:")
    print(f"  Total structures: {len(trajectory)}")
    print(f"  Successful: {successful_count}")
    print(f"  Teacher (GemNet-OC) failures: {teacher_failures}")
    print(f"  Student (Crack) failures: {student_failures}")
    print("="*80)
    
    if successful_count == 0:
        print("No structures were processed successfully!")
        return
    
    # Calculate metrics and create plots
    analyze_results(results, results_dir)
    
    return results


def analyze_results(results, results_dir):
    """Analyze and visualize teacher vs student model results"""
    
    # Convert to numpy arrays
    teacher_energies = np.array(results["teacher_energies"])
    student_energies = np.array(results["student_energies"])
    dft_energies = np.array(results["dft_energies"])
    teacher_times = np.array(results["teacher_times"])
    student_times = np.array(results["student_times"])
    num_atoms = np.array(results["num_atoms"])
    
    # Find valid comparisons
    valid_both = ~np.isnan(teacher_energies) & ~np.isnan(student_energies)
    valid_teacher = ~np.isnan(teacher_energies)
    valid_student = ~np.isnan(student_energies)
    valid_dft = ~np.isnan(dft_energies)
    
    print(f"\nVALID RESULTS:")
    print(f"  Both models: {valid_both.sum()}")
    print(f"  Teacher only: {valid_teacher.sum()}")
    print(f"  Student only: {valid_student.sum()}")
    print(f"  With DFT energies: {valid_dft.sum()}")
    
    # Calculate metrics where both models succeeded
    if valid_both.sum() > 0:
        teacher_valid = teacher_energies[valid_both]
        student_valid = student_energies[valid_both]
        
        # Teacher vs Student comparison
        energy_diff = teacher_valid - student_valid
        mae = np.mean(np.abs(energy_diff))
        rmse = np.sqrt(np.mean(energy_diff**2))
        corr = np.corrcoef(teacher_valid, student_valid)[0, 1]
        
        print(f"\nTEACHER vs STUDENT COMPARISON ({valid_both.sum()} samples):")
        print(f"  Energy MAE: {mae:.4f} eV")
        print(f"  Energy RMSE: {rmse:.4f} eV")
        print(f"  Correlation: {corr:.4f}")
        
        # Per-atom metrics
        atoms_valid = num_atoms[valid_both]
        teacher_epa = teacher_valid / atoms_valid
        student_epa = student_valid / atoms_valid
        
        epa_diff = teacher_epa - student_epa
        epa_mae = np.mean(np.abs(epa_diff))
        epa_rmse = np.sqrt(np.mean(epa_diff**2))
        
        print(f"  Energy/Atom MAE: {epa_mae:.4f} eV/atom")
        print(f"  Energy/Atom RMSE: {epa_rmse:.4f} eV/atom")
    
    # Timing comparison
    if valid_teacher.sum() > 0 and valid_student.sum() > 0:
        teacher_avg_time = np.nanmean(teacher_times[valid_teacher])
        student_avg_time = np.nanmean(student_times[valid_student])
        
        print(f"\nTIMING COMPARISON:")
        print(f"  Teacher avg time: {teacher_avg_time:.4f} s")
        print(f"  Student avg time: {student_avg_time:.4f} s")
        print(f"  Speedup: {teacher_avg_time/student_avg_time:.2f}x")
    
    # Create plots
    create_comparison_plots(results, results_dir)
    
    # Save results
    save_detailed_results(results, results_dir)


def create_comparison_plots(results, results_dir):
    """Create comparison plots"""
    
    teacher_energies = np.array(results["teacher_energies"])
    student_energies = np.array(results["student_energies"])
    valid_both = ~np.isnan(teacher_energies) & ~np.isnan(student_energies)
    
    if valid_both.sum() < 2:
        print("Not enough valid data for plots")
        return
    
    teacher_valid = teacher_energies[valid_both]
    student_valid = student_energies[valid_both]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Teacher vs Student scatter
    ax1.scatter(teacher_valid, student_valid, alpha=0.6, s=30)
    min_e = min(teacher_valid.min(), student_valid.min())
    max_e = max(teacher_valid.max(), student_valid.max())
    ax1.plot([min_e, max_e], [min_e, max_e], 'r--', alpha=0.8)
    ax1.set_xlabel('Teacher Energy (eV)')
    ax1.set_ylabel('Student Energy (eV)')
    ax1.set_title('Teacher vs Student Energy')
    ax1.grid(alpha=0.3)
    
    # 2. Energy difference histogram
    energy_diff = teacher_valid - student_valid
    ax2.hist(energy_diff, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Energy Difference (Teacher - Student) [eV]')
    ax2.set_ylabel('Count')
    ax2.set_title('Energy Difference Distribution')
    ax2.grid(alpha=0.3)
    
    # 3. Error vs system size
    num_atoms_valid = np.array(results["num_atoms"])[valid_both]
    ax3.scatter(num_atoms_valid, energy_diff, alpha=0.6, s=30)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Number of Atoms')
    ax3.set_ylabel('Energy Difference (eV)')
    ax3.set_title('Error vs System Size')
    ax3.grid(alpha=0.3)
    
    # 4. Timing comparison
    teacher_times = np.array(results["teacher_times"])
    student_times = np.array(results["student_times"])
    
    valid_times = ~np.isnan(teacher_times) & ~np.isnan(student_times)
    if valid_times.sum() > 0:
        ax4.scatter(teacher_times[valid_times], student_times[valid_times], alpha=0.6, s=30)
        min_t = min(teacher_times[valid_times].min(), student_times[valid_times].min())
        max_t = max(teacher_times[valid_times].max(), student_times[valid_times].max())
        ax4.plot([min_t, max_t], [min_t, max_t], 'r--', alpha=0.8)
        ax4.set_xlabel('Teacher Time (s)')
        ax4.set_ylabel('Student Time (s)')
        ax4.set_title('Inference Time Comparison')
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'teacher_vs_student_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")


def save_detailed_results(results, results_dir):
    """Save detailed results to JSON"""
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    results_json = {k: convert_for_json(v) for k, v in results.items()}
    
    results_path = os.path.join(results_dir, 'teacher_vs_student_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to: {results_path}")


def main():
    """Main function"""
    print("Testing Teacher (GemNet-OC) vs Student (Crack) models on 100 samples")
    print("ROBUST VERSION with improved crack model loading")
    print("="*80)
    
    results = compare_teacher_student_models()
    
    if results:
        print("\n✅ Testing completed successfully!")
    else:
        print("\n❌ Testing failed!")


if __name__ == "__main__":
    main()