"""
Complete OH Adsorption Analysis Pipeline with OC20 Energies (Phases 1-3)

OH adsorption energy calculation using OC20 predefined atomic energies:
 ΔE_OH = E_adslab - E_slab - (E_O + E_H)

Where:
 - E_O = -7.204 eV (OC20 atomic energy)
 - E_H = -3.477 eV (OC20 atomic energy)
 - Reference = E_O + E_H = -10.681 eV

Optimal value: ΔE_OH = 1.1 eV (±0.2 eV discovery range)

This script:
- Phase 1: Load OH data and calculate adsorption energies using OC20
- Phase 2: Generate all comparison plots (min and mean versions)
- Phase 3: Advanced statistical analysis
- All plots in SVG and PNG formats
- Energy limits: optimal ±2eV
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from pathlib import Path
from datetime import datetime
import glob

sys.path.insert(0, str(Path(__file__).parent))

from util import fs, fss, fsss, fsl, subscript, colors, edge_color


class OHAnalysisOC20:
    """Complete OH adsorption analysis pipeline using OC20 atomic energies."""

    # OC20 adsorbate atom energies (eV)
    ADSORBATE_ENERGIES = {
        'H': -3.477,
        'O': -7.204,
        'C': -7.282,
        'N': -8.083
    }

    def __init__(self, base_sim_path: str, output_dir: str, arial_font_path: str = None):
        """Initialize OH analysis pipeline."""
        self.base_sim_path = Path(base_sim_path)
        self.output_dir = Path(output_dir)
        self.arial_font_path = arial_font_path

        # Create output directories
        self.results_dir = self.output_dir / "results_oh"
        self.svg_dir = self.output_dir / "plot_svg_oh"
        self.png_dir = self.output_dir / "plot_png_oh"
        self.metrics_dir = self.output_dir / "metrics_oh"
        self.dft_dir = self.output_dir / "materials_for_dft_oh"

        for d in [self.results_dir, self.svg_dir, self.png_dir, self.metrics_dir, self.dft_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load Arial font
        if arial_font_path and Path(arial_font_path).exists():
            self.arial_font = fm.FontProperties(fname=arial_font_path)
        else:
            self.arial_font = fm.FontProperties(family='DejaVu Sans')

        # Colors
        self.color_crack = colors[0]
        self.color_gemnet = colors[1]
        self.color_both = colors[2]
        self.color_neutral = colors[3]

        # OH optimal target
        self.OH_OPT = 1.1
        self.RANGE = 0.2
        self.OH_XLIM = (self.OH_OPT - 2, self.OH_OPT + 2)
        self.OH_YLIM = (self.OH_OPT - 2, self.OH_OPT + 2)

        # OC20 reference energy for OH: E_O + E_H
        self.ref_energy_oc20 = self.ADSORBATE_ENERGIES['O'] + self.ADSORBATE_ENERGIES['H']

    # ==================== PHASE 1: DATA LOADING ====================

    def phase1_load_data(self):
        """Phase 1: Load and process OH data."""
        print("\n" + "=" * 80)
        print("PHASE 1: OH DATA LOADING AND PREPROCESSING (OC20 METHOD)")
        print("=" * 80)
        print(f"\nUsing OC20 predefined atomic energies:")
        print(f"  O: {self.ADSORBATE_ENERGIES['O']} eV")
        print(f"  H: {self.ADSORBATE_ENERGIES['H']} eV")
        print(f"  OH reference (O + H): {self.ref_energy_oc20:.3f} eV")
        print("=" * 80)

        # Load slab data
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading slab data...")
        slab_file = self.base_sim_path / "slabs_relaxed" / "slab_results.json"
        with open(slab_file) as f:
            self.slab_data = json.load(f)
        print(f"  Loaded {len(self.slab_data)} slab surfaces")

        # Load OH adslab data
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading OH adslab data...")
        oh_dir = self.base_sim_path / "OH"
        chunk_files = sorted(glob.glob(str(oh_dir / "chunk_*" / "OH_adslab_results_chunk*.json")))
        print(f"  Found {len(chunk_files)} chunk files")

        self.oh_adslab_data = {}
        for chunk_file in chunk_files:
            with open(chunk_file) as f:
                chunk_data = json.load(f)
                self.oh_adslab_data.update(chunk_data)

        print(f"  Loaded {len(self.oh_adslab_data)} surfaces")

        # Calculate adsorption energies
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Calculating OH adsorption energies (OC20 method)...")
        self.oh_results = self.calculate_oh_adsorption_energies()
        print(f"  Processed {len(self.oh_results)} configurations")

        # Save to CSV
        self.oh_df = pd.DataFrame(self.oh_results)
        self.oh_df.to_csv(self.results_dir / "oh_adsorption_data_oc20.csv", index=False)
        print(f"  Saved: oh_adsorption_data_oc20.csv")

        # Process min and mean
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing min and mean energies...")
        self.oh_min = self.get_min_energy_data()
        self.oh_mean = self.get_mean_energy_data()

        self.oh_min.to_csv(self.results_dir / "oh_min_energy_oc20.csv", index=False)
        self.oh_mean.to_csv(self.results_dir / "oh_mean_energy_oc20.csv", index=False)
        print(f"  Saved: oh_min_energy_oc20.csv ({len(self.oh_min)} surfaces)")
        print(f"  Saved: oh_mean_energy_oc20.csv ({len(self.oh_mean)} surfaces)")

        # Statistics
        self.calculate_statistics()

        print("\n" + "=" * 80)
        print("PHASE 1 COMPLETE!")
        print("=" * 80)

    def calculate_oh_adsorption_energies(self):
        """Calculate OH adsorption energies using OC20: ΔE = E_adslab - E_slab - (E_O + E_H)."""
        results = []

        for surface_id, surface_data in self.oh_adslab_data.items():
            bulk_id = surface_data['bulk_id']
            miller = tuple(surface_data['miller'])

            for config in surface_data['configurations']:
                config_idx = config['config_idx']
                num_atoms = config['num_atoms']

                # Get slab energy
                if surface_id not in self.slab_data:
                    continue

                slab_crack = self.slab_data[surface_id]['crack']['energy']
                slab_gemnet = self.slab_data[surface_id]['gemnet']['energy']

                # Crack results
                crack_data = config.get('crack', {})
                if crack_data.get('converged', False):
                    # ΔE_OH = E_adslab - E_slab - (E_O + E_H)
                    crack_e_ads = crack_data['energy'] - slab_crack - self.ref_energy_oc20
                    crack_time = crack_data['time']
                    crack_steps = crack_data['steps']
                else:
                    crack_e_ads = None
                    crack_time = None
                    crack_steps = None

                # GemNet results
                gemnet_data = config.get('gemnet', {})
                if gemnet_data.get('converged', False):
                    # ΔE_OH = E_adslab - E_slab - (E_O + E_H)
                    gemnet_e_ads = gemnet_data['energy'] - slab_gemnet - self.ref_energy_oc20
                    gemnet_time = gemnet_data['time']
                    gemnet_steps = gemnet_data['steps']
                else:
                    gemnet_e_ads = None
                    gemnet_time = None
                    gemnet_steps = None

                results.append({
                    'surface_id': surface_id,
                    'bulk_id': bulk_id,
                    'miller': miller,
                    'config_idx': config_idx,
                    'num_atoms': num_atoms,
                    'crack_e_ads': crack_e_ads,
                    'crack_time': crack_time,
                    'crack_steps': crack_steps,
                    'crack_converged': crack_data.get('converged', False),
                    'gemnet_e_ads': gemnet_e_ads,
                    'gemnet_time': gemnet_time,
                    'gemnet_steps': gemnet_steps,
                    'gemnet_converged': gemnet_data.get('converged', False),
                })

        return results

    def get_min_energy_data(self):
        """Get minimum energy configuration per surface."""
        min_data = []

        for surface_id in self.oh_df['surface_id'].unique():
            surf_data = self.oh_df[self.oh_df['surface_id'] == surface_id]

            row = {'surface_id': surface_id, 'bulk_id': surf_data.iloc[0]['bulk_id']}

            # Crack min
            crack_conv = surf_data[surf_data['crack_converged']]
            if len(crack_conv) > 0:
                min_idx = crack_conv['crack_e_ads'].idxmin()
                min_crack = crack_conv.loc[min_idx]
                row['crack_e_ads'] = min_crack['crack_e_ads']
                row['crack_time'] = min_crack['crack_time']
                row['crack_steps'] = min_crack['crack_steps']
                row['crack_converged'] = True
            else:
                row['crack_e_ads'] = np.nan
                row['crack_time'] = np.nan
                row['crack_steps'] = np.nan
                row['crack_converged'] = False

            # GemNet min
            gemnet_conv = surf_data[surf_data['gemnet_converged']]
            if len(gemnet_conv) > 0:
                min_idx = gemnet_conv['gemnet_e_ads'].idxmin()
                min_gemnet = gemnet_conv.loc[min_idx]
                row['gemnet_e_ads'] = min_gemnet['gemnet_e_ads']
                row['gemnet_time'] = min_gemnet['gemnet_time']
                row['gemnet_steps'] = min_gemnet['gemnet_steps']
                row['gemnet_converged'] = True
            else:
                row['gemnet_e_ads'] = np.nan
                row['gemnet_time'] = np.nan
                row['gemnet_steps'] = np.nan
                row['gemnet_converged'] = False

            min_data.append(row)

        return pd.DataFrame(min_data)

    def get_mean_energy_data(self):
        """Get mean energy across configurations per surface."""
        mean_data = []

        for surface_id in self.oh_df['surface_id'].unique():
            surf_data = self.oh_df[self.oh_df['surface_id'] == surface_id]

            row = {'surface_id': surface_id, 'bulk_id': surf_data.iloc[0]['bulk_id']}

            # Crack mean
            crack_conv = surf_data[surf_data['crack_converged']]
            if len(crack_conv) > 0:
                row['crack_e_ads'] = crack_conv['crack_e_ads'].mean()
                row['crack_e_ads_std'] = crack_conv['crack_e_ads'].std()
                row['crack_time'] = crack_conv['crack_time'].mean()
                row['crack_steps'] = crack_conv['crack_steps'].mean()
                row['crack_converged'] = True
                row['crack_n_configs'] = len(crack_conv)
            else:
                row['crack_e_ads'] = np.nan
                row['crack_e_ads_std'] = np.nan
                row['crack_time'] = np.nan
                row['crack_steps'] = np.nan
                row['crack_converged'] = False
                row['crack_n_configs'] = 0

            # GemNet mean
            gemnet_conv = surf_data[surf_data['gemnet_converged']]
            if len(gemnet_conv) > 0:
                row['gemnet_e_ads'] = gemnet_conv['gemnet_e_ads'].mean()
                row['gemnet_e_ads_std'] = gemnet_conv['gemnet_e_ads'].std()
                row['gemnet_time'] = gemnet_conv['gemnet_time'].mean()
                row['gemnet_steps'] = gemnet_conv['gemnet_steps'].mean()
                row['gemnet_converged'] = True
                row['gemnet_n_configs'] = len(gemnet_conv)
            else:
                row['gemnet_e_ads'] = np.nan
                row['gemnet_e_ads_std'] = np.nan
                row['gemnet_time'] = np.nan
                row['gemnet_steps'] = np.nan
                row['gemnet_converged'] = False
                row['gemnet_n_configs'] = 0

            mean_data.append(row)

        return pd.DataFrame(mean_data)

    def calculate_statistics(self):
        """Calculate and print statistics."""
        print("\n" + "=" * 60)
        print("OH ADSORPTION STATISTICS (OC20 METHOD)")
        print("=" * 60)

        total = len(self.oh_df)
        crack_conv = self.oh_df['crack_converged'].sum()
        gemnet_conv = self.oh_df['gemnet_converged'].sum()
        both_conv = (self.oh_df['crack_converged'] & self.oh_df['gemnet_converged']).sum()

        print(f"\nTotal configurations: {total}")
        print(f"Crack converged: {crack_conv} ({crack_conv/total*100:.1f}%)")
        print(f"GemNet converged: {gemnet_conv} ({gemnet_conv/total*100:.1f}%)")
        print(f"Both converged: {both_conv} ({both_conv/total*100:.1f}%)")

        # Error statistics
        both = self.oh_df[self.oh_df['crack_converged'] & self.oh_df['gemnet_converged']].copy()
        both['error'] = both['crack_e_ads'] - both['gemnet_e_ads']

        if len(both) > 0:
            print(f"\nEnergy Comparison (eV):")
            print(f"  MAE: {np.abs(both['error']).mean():.4f}")
            print(f"  RMSE: {np.sqrt((both['error']**2).mean()):.4f}")
            print(f"  Correlation: {both[['crack_e_ads', 'gemnet_e_ads']].corr().iloc[0,1]:.4f}")

        # Active catalysts
        for calc in ['crack', 'gemnet']:
            conv_df = self.oh_df[self.oh_df[f'{calc}_converged']]
            active = conv_df[
                (conv_df[f'{calc}_e_ads'] >= self.OH_OPT - self.RANGE) &
                (conv_df[f'{calc}_e_ads'] <= self.OH_OPT + self.RANGE)
            ]
            print(f"\n{calc.upper()} active catalysts: {len(active)}")

        print("=" * 60)

    # ==================== PHASE 2: PLOTTING ====================

    def save_plot(self, fig, name: str):
        """Save plot in SVG and PNG."""
        fig.savefig(self.svg_dir / f"{name}.svg", dpi=300, bbox_inches='tight')
        fig.savefig(self.png_dir / f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def style_ax(self, ax):
        """Apply styling to axis."""
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(self.arial_font)
            label.set_fontsize(fs)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontproperties=self.arial_font, fontsize=fsl)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontproperties=self.arial_font, fontsize=fsl)
        if ax.get_title():
            ax.set_title(ax.get_title(), fontproperties=self.arial_font, fontsize=fsl)
        for spine in ax.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(1.5)

    def phase2_generate_plots(self):
        """Phase 2: Generate basic plots."""
        print("\n" + "=" * 80)
        print("PHASE 2: GENERATING BASIC PLOTS")
        print("=" * 80)

        self.plot_01_histogram_min()
        self.plot_02_histogram_mean()
        self.plot_03_parity_min()
        self.plot_04_parity_mean()
        self.plot_05_error_analysis_min()
        self.plot_06_error_analysis_mean()
        self.plot_07_discovery_timeline()
        self.plot_08_active_catalysts()

        print("\n" + "=" * 80)
        print("PHASE 2 COMPLETE!")
        print("=" * 80)

    def plot_01_histogram_min(self):
        """OH histogram with min energy."""
        fig, ax = plt.subplots(figsize=(10, 7))

        crack_data = self.oh_min[self.oh_min['crack_converged']]['crack_e_ads']
        gemnet_data = self.oh_min[self.oh_min['gemnet_converged']]['gemnet_e_ads']

        ax.hist(crack_data, bins=80, range=self.OH_XLIM, color=self.color_crack, alpha=0.5,
                edgecolor=edge_color, linewidth=0.5, label='Crack')
        ax.hist(gemnet_data, bins=80, range=self.OH_XLIM, color=self.color_gemnet, alpha=0.5,
                edgecolor=edge_color, linewidth=0.5, label='GemNet')
        ax.axvline(self.OH_OPT, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax.axvspan(self.OH_OPT - self.RANGE, self.OH_OPT + self.RANGE, alpha=0.2, color='red')

        ax.set_xlim(self.OH_XLIM)
        ax.set_xlabel(f'ΔE{subscript("OH")} (eV)')
        ax.set_ylabel('Count')
        ax.set_title('OH Adsorption Energy Distribution (Min per Slab, OC20)')
        ax.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_01_histogram_min")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_01_histogram_min")

    def plot_02_histogram_mean(self):
        """OH histogram with mean energy."""
        fig, ax = plt.subplots(figsize=(10, 7))

        crack_data = self.oh_mean[self.oh_mean['crack_converged']]['crack_e_ads']
        gemnet_data = self.oh_mean[self.oh_mean['gemnet_converged']]['gemnet_e_ads']

        ax.hist(crack_data, bins=80, range=self.OH_XLIM, color=self.color_crack, alpha=0.5,
                edgecolor=edge_color, linewidth=0.5, label='Crack')
        ax.hist(gemnet_data, bins=80, range=self.OH_XLIM, color=self.color_gemnet, alpha=0.5,
                edgecolor=edge_color, linewidth=0.5, label='GemNet')
        ax.axvline(self.OH_OPT, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax.axvspan(self.OH_OPT - self.RANGE, self.OH_OPT + self.RANGE, alpha=0.2, color='red')

        ax.set_xlim(self.OH_XLIM)
        ax.set_xlabel(f'ΔE{subscript("OH")} (eV)')
        ax.set_ylabel('Count')
        ax.set_title('OH Adsorption Energy Distribution (Mean per Slab, OC20)')
        ax.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_02_histogram_mean")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_02_histogram_mean")

    def plot_03_parity_min(self):
        """Parity plot with min energy."""
        fig, ax = plt.subplots(figsize=(9, 9))

        both = self.oh_min[self.oh_min['crack_converged'] & self.oh_min['gemnet_converged']]

        ax.scatter(both['crack_e_ads'], both['gemnet_e_ads'], c=self.color_crack,
                   s=20, alpha=0.5, edgecolors='none')
        ax.plot(self.OH_XLIM, self.OH_YLIM, 'k--', linewidth=2, label='Perfect agreement')

        ax.set_xlim(self.OH_XLIM)
        ax.set_ylim(self.OH_YLIM)
        ax.set_xlabel(f'Crack ΔE{subscript("OH")} (eV)')
        ax.set_ylabel(f'GemNet ΔE{subscript("OH")} (eV)')
        ax.set_title('OH: Crack vs GemNet (Min per Slab, OC20)')
        ax.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_03_parity_min")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_03_parity_min")

    def plot_04_parity_mean(self):
        """Parity plot with mean energy."""
        fig, ax = plt.subplots(figsize=(9, 9))

        both = self.oh_mean[self.oh_mean['crack_converged'] & self.oh_mean['gemnet_converged']]

        # Plot with error bars
        ax.errorbar(both['crack_e_ads'], both['gemnet_e_ads'],
                    xerr=both['crack_e_ads_std'], yerr=both['gemnet_e_ads_std'],
                    fmt='o', color=self.color_crack, markersize=4, alpha=0.3,
                    ecolor=self.color_neutral, elinewidth=0.5, capsize=0)

        ax.plot(self.OH_XLIM, self.OH_YLIM, 'k--', linewidth=2, label='Perfect agreement')

        ax.set_xlim(self.OH_XLIM)
        ax.set_ylim(self.OH_YLIM)
        ax.set_xlabel(f'Crack ΔE{subscript("OH")} (eV)')
        ax.set_ylabel(f'GemNet ΔE{subscript("OH")} (eV)')
        ax.set_title('OH: Crack vs GemNet (Mean per Slab, OC20)')
        ax.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_04_parity_mean")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_04_parity_mean")

    def plot_05_error_analysis_min(self):
        """Error analysis with min energy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        both = self.oh_min[self.oh_min['crack_converged'] & self.oh_min['gemnet_converged']].copy()
        both['error'] = np.abs(both['crack_e_ads'] - both['gemnet_e_ads'])

        # Error vs energy
        scatter = ax1.scatter(both['crack_e_ads'], both['error'], c=both['gemnet_e_ads'],
                             cmap='viridis', s=20, alpha=0.5, edgecolors='none')
        ax1.axvline(self.OH_OPT, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax1.set_xlim(self.OH_XLIM)
        ax1.set_xlabel(f'Crack ΔE{subscript("OH")} (eV)')
        ax1.set_ylabel('|Crack - GemNet| (eV)')
        ax1.set_title('Prediction Error vs Energy (Min, OC20)')
        ax1.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax1.grid(True, alpha=0.3, linestyle='--')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('GemNet Energy (eV)', fontproperties=self.arial_font, fontsize=fss)

        # Error histogram
        ax2.hist(both['error'], bins=50, color=self.color_crack, alpha=0.6,
                edgecolor=edge_color, linewidth=0.5)
        ax2.set_xlabel('|Crack - GemNet| (eV)')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution (Min, OC20)')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        for ax in [ax1, ax2]:
            self.style_ax(ax)

        plt.tight_layout()
        self.save_plot(fig, "oh_05_error_analysis_min")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_05_error_analysis_min")

    def plot_06_error_analysis_mean(self):
        """Error analysis with mean energy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        both = self.oh_mean[self.oh_mean['crack_converged'] & self.oh_mean['gemnet_converged']].copy()
        both['error'] = np.abs(both['crack_e_ads'] - both['gemnet_e_ads'])

        # Error vs energy
        scatter = ax1.scatter(both['crack_e_ads'], both['error'], c=both['gemnet_e_ads'],
                             cmap='viridis', s=20, alpha=0.5, edgecolors='none')
        ax1.axvline(self.OH_OPT, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax1.set_xlim(self.OH_XLIM)
        ax1.set_xlabel(f'Crack ΔE{subscript("OH")} (eV)')
        ax1.set_ylabel('|Crack - GemNet| (eV)')
        ax1.set_title('Prediction Error vs Energy (Mean, OC20)')
        ax1.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax1.grid(True, alpha=0.3, linestyle='--')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('GemNet Energy (eV)', fontproperties=self.arial_font, fontsize=fss)

        # Error histogram
        ax2.hist(both['error'], bins=50, color=self.color_crack, alpha=0.6,
                edgecolor=edge_color, linewidth=0.5)
        ax2.set_xlabel('|Crack - GemNet| (eV)')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution (Mean, OC20)')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        for ax in [ax1, ax2]:
            self.style_ax(ax)

        plt.tight_layout()
        self.save_plot(fig, "oh_06_error_analysis_mean")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_06_error_analysis_mean")

    def plot_07_discovery_timeline(self):
        """Discovery timeline (min energy) - both methods on one plot."""
        fig, ax = plt.subplots(figsize=(10, 7))

        for calc, color in [('crack', self.color_crack), ('gemnet', self.color_gemnet)]:
            conv_df = self.oh_min[self.oh_min[f'{calc}_converged']].copy()
            active = conv_df[
                (conv_df[f'{calc}_e_ads'] >= self.OH_OPT - self.RANGE) &
                (conv_df[f'{calc}_e_ads'] <= self.OH_OPT + self.RANGE)
            ]

            if len(active) > 0:
                active = active.sort_values(f'{calc}_time')
                active['cumulative_time'] = active[f'{calc}_time'].cumsum() / 3600
                active['discovery_number'] = range(1, len(active) + 1)

                ax.plot(active['cumulative_time'], active['discovery_number'],
                       color=color, linewidth=2, label=calc.capitalize())

        ax.set_xlabel('Cumulative Time (hours)')
        ax.set_ylabel('Active OH Catalysts Discovered')
        ax.set_title('OH Discovery Timeline (OC20)')
        ax.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_07_discovery_timeline")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_07_discovery_timeline")

    def plot_08_active_catalysts(self):
        """Active catalyst comparison."""
        fig, ax = plt.subplots(figsize=(10, 7))

        counts = []
        labels = []

        for calc in ['crack', 'gemnet']:
            conv_df = self.oh_min[self.oh_min[f'{calc}_converged']]
            active = conv_df[
                (conv_df[f'{calc}_e_ads'] >= self.OH_OPT - self.RANGE) &
                (conv_df[f'{calc}_e_ads'] <= self.OH_OPT + self.RANGE)
            ]
            counts.append(len(active))
            labels.append(calc.capitalize())

        x = np.arange(len(labels))
        bars = ax.bar(x, counts, color=[self.color_crack, self.color_gemnet],
                     edgecolor=edge_color, linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Number of Active Catalysts')
        ax.set_title(f'OH Active Catalysts (ΔE = {self.OH_OPT} ± {self.RANGE} eV, OC20)')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{count}', ha='center', va='bottom',
                   fontproperties=self.arial_font, fontsize=fsl)

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_08_active_catalysts")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_08_active_catalysts")

    # ==================== PHASE 3: ADVANCED PLOTS ====================

    def phase3_advanced_plots(self):
        """Phase 3: Generate advanced analysis plots."""
        print("\n" + "=" * 80)
        print("PHASE 3: ADVANCED ANALYSIS PLOTS")
        print("=" * 80)

        self.plot_09_classification_performance()
        self.plot_10_energy_landscape()
        self.plot_11_cost_benefit()
        self.plot_12_material_stratification()

        print("\n" + "=" * 80)
        print("PHASE 3 COMPLETE!")
        print("=" * 80)

    def plot_09_classification_performance(self):
        """Classification performance - active catalyst identification."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Energy distribution comparison
        for calc, color in [('crack', self.color_crack), ('gemnet', self.color_gemnet)]:
            conv_df = self.oh_min[self.oh_min[f'{calc}_converged']].copy()
            active = conv_df[
                (conv_df[f'{calc}_e_ads'] >= self.OH_OPT - self.RANGE) &
                (conv_df[f'{calc}_e_ads'] <= self.OH_OPT + self.RANGE)
            ]

            # Plot energy distribution
            ax1.hist(conv_df[f'{calc}_e_ads'], bins=100, range=self.OH_XLIM,
                    color=color, alpha=0.3, edgecolor=edge_color, linewidth=0.5,
                    label=f'{calc.capitalize()} (all)')
            ax1.hist(active[f'{calc}_e_ads'], bins=100, range=self.OH_XLIM,
                    color=color, alpha=0.7, edgecolor=edge_color, linewidth=0.5,
                    label=f'{calc.capitalize()} (active)')

        ax1.axvline(self.OH_OPT, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax1.axvspan(self.OH_OPT - self.RANGE, self.OH_OPT + self.RANGE, alpha=0.2, color='red')
        ax1.set_xlim(self.OH_XLIM)
        ax1.set_xlabel(f'ΔE{subscript("OH")} (eV)')
        ax1.set_ylabel('Count')
        ax1.set_title('OH: Active vs All Catalysts')
        ax1.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Confusion matrix-style bar chart
        categories = []
        crack_counts = []
        gemnet_counts = []

        for calc in ['crack', 'gemnet']:
            conv_df = self.oh_min[self.oh_min[f'{calc}_converged']].copy()
            active = (
                (conv_df[f'{calc}_e_ads'] >= self.OH_OPT - self.RANGE) &
                (conv_df[f'{calc}_e_ads'] <= self.OH_OPT + self.RANGE)
            ).sum()
            inactive = len(conv_df) - active

            if calc == 'crack':
                crack_counts = [active, inactive]
            else:
                gemnet_counts = [active, inactive]

        categories = ['Active', 'Inactive']
        x = np.arange(len(categories))
        width = 0.35

        ax2.bar(x - width/2, crack_counts, width, color=self.color_crack,
               edgecolor=edge_color, linewidth=1.5, label='Crack')
        ax2.bar(x + width/2, gemnet_counts, width, color=self.color_gemnet,
               edgecolor=edge_color, linewidth=1.5, label='GemNet')

        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Count')
        ax2.set_title('OH: Catalyst Classification')
        ax2.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add value labels
        for i, (crack_val, gemnet_val) in enumerate(zip(crack_counts, gemnet_counts)):
            ax2.text(i - width/2, crack_val, f'{crack_val}', ha='center', va='bottom',
                    fontproperties=self.arial_font, fontsize=fsss)
            ax2.text(i + width/2, gemnet_val, f'{gemnet_val}', ha='center', va='bottom',
                    fontproperties=self.arial_font, fontsize=fsss)

        for ax in [ax1, ax2]:
            self.style_ax(ax)

        plt.tight_layout()
        self.save_plot(fig, "oh_09_classification_performance")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_09_classification_performance")

    def plot_10_energy_landscape(self):
        """Energy landscape topology (hexbin density plot)."""
        fig, ax = plt.subplots(figsize=(10, 9))

        both = self.oh_min[self.oh_min['crack_converged'] & self.oh_min['gemnet_converged']]

        # Filter to plot limits
        plot_data = both[
            (both['crack_e_ads'] >= self.OH_XLIM[0]) &
            (both['crack_e_ads'] <= self.OH_XLIM[1]) &
            (both['gemnet_e_ads'] >= self.OH_YLIM[0]) &
            (both['gemnet_e_ads'] <= self.OH_YLIM[1])
        ]

        h = ax.hexbin(plot_data['crack_e_ads'], plot_data['gemnet_e_ads'],
                     gridsize=50, cmap='YlOrRd', mincnt=1, edgecolors='none')
        ax.plot(self.OH_XLIM, self.OH_YLIM, 'k--', linewidth=2, label='Perfect agreement')

        # Optimal region box
        from matplotlib.patches import Rectangle
        rect = Rectangle((self.OH_OPT - self.RANGE, self.OH_OPT - self.RANGE),
                         2 * self.RANGE, 2 * self.RANGE,
                         linewidth=2, edgecolor='red', facecolor='none',
                         linestyle='--', label='Optimal region')
        ax.add_patch(rect)

        ax.set_xlim(self.OH_XLIM)
        ax.set_ylim(self.OH_YLIM)
        ax.set_xlabel(f'Crack ΔE{subscript("OH")} (eV)')
        ax.set_ylabel(f'GemNet ΔE{subscript("OH")} (eV)')
        ax.set_title('OH Energy Landscape Topology (OC20)')
        ax.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)

        cbar = plt.colorbar(h, ax=ax)
        cbar.set_label('Density', fontproperties=self.arial_font, fontsize=fss)
        cbar.ax.tick_params(labelsize=fs)

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_10_energy_landscape")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_10_energy_landscape")

    def plot_11_cost_benefit(self):
        """Cost-benefit analysis (discovery yield vs time)."""
        fig, ax = plt.subplots(figsize=(10, 7))

        for calc, color in [('crack', self.color_crack), ('gemnet', self.color_gemnet)]:
            conv_df = self.oh_min[self.oh_min[f'{calc}_converged']].copy()
            active = conv_df[
                (conv_df[f'{calc}_e_ads'] >= self.OH_OPT - self.RANGE) &
                (conv_df[f'{calc}_e_ads'] <= self.OH_OPT + self.RANGE)
            ]

            if len(active) > 0:
                active = active.sort_values(f'{calc}_time')
                active['cumulative_time'] = active[f'{calc}_time'].cumsum() / 3600
                active['cumulative_discoveries'] = range(1, len(active) + 1)

                ax.plot(active['cumulative_time'], active['cumulative_discoveries'],
                       color=color, linewidth=2, label=calc.capitalize())

        ax.set_xlabel('Cumulative Time (hours)')
        ax.set_ylabel('Cumulative Discoveries')
        ax.set_title('OH: Discovery Yield vs Computational Cost (OC20)')
        ax.legend(prop=self.arial_font, fontsize=fss, frameon=True, edgecolor=edge_color)
        ax.grid(True, alpha=0.3, linestyle='--')

        self.style_ax(ax)
        plt.tight_layout()
        self.save_plot(fig, "oh_11_cost_benefit")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_11_cost_benefit")

    def plot_12_material_stratification(self):
        """Material class stratification (error by material)."""
        # Calculate error for each surface
        both = self.oh_min[self.oh_min['crack_converged'] & self.oh_min['gemnet_converged']].copy()
        both['error'] = np.abs(both['crack_e_ads'] - both['gemnet_e_ads'])

        # Group by bulk_id
        material_stats = both.groupby('bulk_id').agg({
            'error': ['mean', 'std', 'count']
        }).reset_index()
        material_stats.columns = ['bulk_id', 'mae', 'std', 'count']
        material_stats = material_stats.sort_values('mae', ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Scatter: error vs sample size
        scatter = ax1.scatter(material_stats['count'], material_stats['mae'],
                            c=material_stats['mae'], cmap='RdYlGn_r',
                            s=100, alpha=0.6, edgecolors=edge_color, linewidth=1)
        ax1.set_xlabel('Number of Surfaces')
        ax1.set_ylabel('Mean Absolute Error (eV)')
        ax1.set_title('OH: Error vs Sample Size by Material')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('MAE (eV)', fontproperties=self.arial_font, fontsize=fss)
        cbar.ax.tick_params(labelsize=fs)

        # Bar: top 20 worst materials
        worst = material_stats.nlargest(20, 'mae')
        y_pos = np.arange(len(worst))
        ax2.barh(y_pos, worst['mae'].values,
                color=self.color_crack, edgecolor=edge_color, linewidth=1)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(worst['bulk_id'].values, fontsize=8)
        ax2.set_xlabel('Mean Absolute Error (eV)')
        ax2.set_title('OH: Top 20 Worst Materials')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='x')

        for ax in [ax1, ax2]:
            self.style_ax(ax)

        plt.tight_layout()
        self.save_plot(fig, "oh_12_material_stratification")
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved: oh_12_material_stratification")

    # ==================== MAIN EXECUTION ====================

    def run_complete_analysis(self):
        """Run complete OH analysis pipeline."""
        print("\n" + "=" * 80)
        print("OH ADSORPTION ANALYSIS WITH OC20 ENERGIES")
        print("=" * 80)

        self.phase1_load_data()
        self.phase2_generate_plots()
        self.phase3_advanced_plots()

        print("\n" + "=" * 80)
        print("COMPLETE OH ANALYSIS FINISHED!")
        print("=" * 80)
        print(f"\nResults saved to:")
        print(f"  CSV: {self.results_dir}")
        print(f"  SVG: {self.svg_dir}")
        print(f"  PNG: {self.png_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OH adsorption energy analysis with OC20 reference energies')
    parser.add_argument('--sim_path', required=True, help='Path to ML relaxation results (2_ml_relaxation/)')
    parser.add_argument('--output_dir', default='./output', help='Output directory for results and plots')
    args = parser.parse_args()

    pipeline = OHAnalysisOC20(args.sim_path, args.output_dir)
    pipeline.run_complete_analysis()
