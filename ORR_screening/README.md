# ORR Catalyst Screening via Multi-Fidelity MLIP Workflow

This repository contains the high-throughput computational screening pipeline for oxygen reduction reaction (ORR) catalyst discovery described in:

> **Angular Relational Knowledge Distillation of Machine Learning Interatomic Potentials for Scalable Catalyst Exploration**
> Hyukjun Lim\*, Seokhyun Choung\*, Jinuk Moon, Jeong Woo Han
> *npj Computational Materials* (under review)

Starting from 11,379 bulk compounds in the Materials Project database, the pipeline generates 580,329 adsorbate–slab configurations, screens them through a three-stage MLIP→DFT funnel, and identifies 30 DFT-confirmed ORR catalyst candidates.

---

## Pipeline Overview

```
1_structure_preparation/   — slab generation & adsorbate placement
        ↓
2_ml_relaxation/           — structural relaxation (ARK student + GemNet-OC teacher)
        ↓
3_adsorption_energy/       — O* and OH* adsorption energies (OC20 reference)
        ↓
4_orr_screening/           — volcano plot screening & DFT validation
```

### Three-stage screening funnel

| Stage | Model | Threshold | Candidates |
|-------|-------|-----------|-----------|
| 1 | ARK student (1M params) | U_lim > 0 V | 2,981 |
| 2 | GemNet-OC teacher (39M params) | U_lim > 0.5 V | 913 |
| 3 | DFT (VASP) | U_lim > 0.5 V | **30** |

---

## Repository Structure

```
.
├── 1_structure_preparation/
│   ├── step1_prescreen_slabs.py       # Filter stable slabs from Materials Project bulks
│   ├── step2_generate_adsorbates.py   # Place CO adsorbates on high-symmetry sites
│   ├── replace_adsorbate.py           # Replace CO with O*, OH* (and other species)
│   └── downsize_dataset.py            # Sample 10 frames per trajectory
│
├── 2_ml_relaxation/
│   ├── model_loading.py               # ARK student / GemNet-OC model loader
│   ├── slab_relaxation.py             # Relax clean slabs
│   └── adslab_relaxation_chunked.py   # Relax adsorbate–slab configs (chunked for GPU memory)
│
├── 3_adsorption_energy/
│   ├── adsorption_calculator_oc20.py  # Core energy calculation module (OC20 atomic references)
│   ├── o_complete_analysis_oc20.py    # Full O* analysis pipeline
│   └── oh_complete_analysis_oc20.py   # Full OH* analysis pipeline
│
├── 4_orr_screening/
│   ├── screening_overpotential.py     # Multi-fidelity ORR screening (overpotential-based)
│   ├── create_layer_volcano.py        # Layered 2D volcano plots (paper figures)
│   └── visualize_final_survivors.py   # Summary plots for DFT-validated candidates
│
└── data/
    ├── adsorption_energies/           # Per-surface minimum adsorption energies (screening input)
    │   ├── o_min_energy_oc20.csv      # O* minimum ΔE per surface (~11K surfaces)
    │   └── oh_min_energy_oc20.csv     # OH* minimum ΔE per surface
    │
    ├── screening_results/             # Screening outputs at each funnel stage
    │   ├── FORK_ORR_active_eta.csv    # Stage 1: ARK student survivors (η < 0.73 V)
    │   ├── GemNet_ORR_validated_eta.csv  # Stage 2: GemNet teacher survivors
    │   ├── DFT_ORR_validated_final.csv   # Stage 3: DFT-evaluated candidates with energies
    │   ├── ORR_final_survivors.csv       # Final survivors (U_DFT > 0.8 V)
    │   ├── ORR_survivors_eta_0.5V.csv    # Survivors at η < 0.5 V threshold
    │   └── ORR_final_survivors_with_formulas.csv  # Above + chemical formulas
    │
    ├── dft_structures/                # VASP input (POSCAR) and relaxed (CONTCAR) structures
    │   └── {mp-id}_{miller}/          # e.g., mp-4876_111/ → SbSe3Tl3 (111) surface
    │       ├── slab/  {POSCAR, CONTCAR}
    │       ├── O/     {POSCAR, CONTCAR}
    │       └── OH/    {POSCAR, CONTCAR}
    │
    └── figures/                       # Volcano plot layers (paper figures)
        ├── layer_0_v1.png             # Empty volcano (background)
        ├── layer_1_v1.png             # All ARK student predictions
        ├── layer_2_v1.png             # ARK active region (η < 0.73 V)
        ├── layer_3_v1.png             # GemNet teacher validation
        ├── layer_4_v1.png             # GemNet high-performance (η < 0.65 V)
        ├── layer_5_v1.png             # DFT-confirmed candidates
        ├── screening_funnel.png       # Funnel statistics
        └── ORR_survivors_summary.png  # Final candidate summary
```

---

## ORR Thermodynamics

The 4-electron ORR pathway (acidic conditions):

```
O2 + 4H+ + 4e- → 2H2O
```

Free energies of intermediates are computed from DFT adsorption energies using the computational hydrogen electrode (CHE) framework. The OOH* free energy is estimated via the universal scaling relation:

```
G_OOH = G_OH + 3.2 eV
```

Limiting potential and overpotential:
```
U_lim = min(4.92 - G_OOH, G_OOH - G_O, G_O - G_OH, G_OH)
η = 1.23 - U_lim
```

Adsorption energies use OC20 atomic reference energies: E_O = −7.204 eV, E_H = −3.477 eV.

---

## Requirements

```
Python >= 3.8
ase
ocdata          # Open Catalyst Dataset structure generation
fairchem        # GemNet-OC model (fair-chem.github.io)
pandas
numpy
matplotlib
pymatgen
```

The ARK-distilled student model and GemNet-OC teacher checkpoint are available at the [ARK repository](https://github.com/hyukjunlim/ARK).

---

## Usage

```bash
# 1. Prescreen stable slabs
python 1_structure_preparation/step1_prescreen_slabs.py

# 2. Generate O* and OH* adsorbate configurations
python 1_structure_preparation/step2_generate_adsorbates.py
python 1_structure_preparation/replace_adsorbate.py --adsorbate O
python 1_structure_preparation/replace_adsorbate.py --adsorbate OH

# 3. Relax all structures
python 2_ml_relaxation/slab_relaxation.py
python 2_ml_relaxation/adslab_relaxation_chunked.py

# 4. Calculate adsorption energies
python 3_adsorption_energy/o_complete_analysis_oc20.py  --sim_path ./2_ml_relaxation --output_dir ./output
python 3_adsorption_energy/oh_complete_analysis_oc20.py --sim_path ./2_ml_relaxation --output_dir ./output

# 5. Run ORR screening
python 4_orr_screening/screening_overpotential.py
python 4_orr_screening/create_layer_volcano.py
python 4_orr_screening/visualize_final_survivors.py
```

---

## Citation

```bibtex
@article{lim2025ark,
  title   = {Angular Relational Knowledge Distillation of Machine Learning Interatomic Potentials for Scalable Catalyst Exploration},
  author  = {Lim, Hyukjun and Choung, Seokhyun and Moon, Jinuk and Han, Jeong Woo},
  journal = {npj Computational Materials},
  year    = {2025}
}
```
