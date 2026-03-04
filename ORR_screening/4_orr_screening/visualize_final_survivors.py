#!/usr/bin/env python3
"""
Visualize the final DFT-validated ORR catalyst survivors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
from pathlib import Path

# Set mathtext to regular
matplotlib.rcParams['mathtext.default'] = 'regular'

# File paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
SURVIVORS_FILE = _REPO_ROOT / "data" / "screening_results" / "ORR_final_survivors.csv"
DFT_VALIDATED_FILE = _REPO_ROOT / "data" / "screening_results" / "DFT_ORR_validated_final.csv"
OUTPUT_DIR = _REPO_ROOT / "data" / "figures"

# Color scheme
colors = {
    'excellent': '#2ecc71',  # Green
    'good': '#3498db',       # Blue
    'acceptable': '#f39c12',  # Orange
    'poor': '#e74c3c'        # Red
}


def create_survivors_summary():
    """Create a comprehensive summary plot of the final survivors"""

    # Load data
    survivors = pd.read_csv(SURVIVORS_FILE)
    all_dft = pd.read_csv(DFT_VALIDATED_FILE)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Volcano plot with survivors highlighted
    ax1 = plt.subplot(2, 3, 1)
    create_volcano_survivors(ax1, survivors, all_dft)

    # 2. Bar plot of overpotentials
    ax2 = plt.subplot(2, 3, 2)
    create_overpotential_bars(ax2, survivors)

    # 3. GemNet vs DFT correlation
    ax3 = plt.subplot(2, 3, 3)
    create_correlation_plot(ax3, all_dft)

    # 4. Performance distribution pie chart
    ax4 = plt.subplot(2, 3, 4)
    create_performance_pie(ax4, all_dft)

    # 5. Error distribution histogram
    ax5 = plt.subplot(2, 3, 5)
    create_error_histogram(ax5, all_dft)

    # 6. Limiting step analysis
    ax6 = plt.subplot(2, 3, 6)
    create_limiting_step_analysis(ax6, survivors)

    plt.suptitle('ORR Catalyst Screening - Final DFT-Validated Results', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = OUTPUT_DIR / 'ORR_survivors_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to: {output_file}")
    plt.close()


def create_volcano_survivors(ax, survivors, all_dft):
    """Create volcano plot highlighting survivors"""

    # Create background volcano
    G_OH_range = np.linspace(-0.5, 2.5, 200)
    G_O_range = np.linspace(0, 5, 200)
    G_OH_grid, G_O_grid = np.meshgrid(G_OH_range, G_O_range)

    # Calculate limiting potential
    G_OOH = G_OH_grid + 3.2
    step1 = 4.92 - G_OOH
    step2 = G_OOH - G_O_grid
    step3 = G_O_grid - G_OH_grid
    step4 = G_OH_grid
    U_limiting = np.minimum(np.minimum(step1, step2), np.minimum(step3, step4))

    # Create contour
    contour = ax.contourf(G_OH_grid, G_O_grid, U_limiting,
                          levels=np.arange(-0.5, 1.3, 0.1),
                          cmap='inferno', alpha=0.7)

    # Plot all DFT points
    ax.scatter(all_dft['dft_e_ads_oh'], all_dft['dft_e_ads_o'],
              c='gray', s=30, alpha=0.5, label='DFT evaluated')

    # Highlight survivors
    ax.scatter(survivors['dft_e_ads_oh'], survivors['dft_e_ads_o'],
              c='yellow', s=100, marker='*', edgecolors='black',
              linewidth=1.5, label=f'Survivors (n={len(survivors)})', zorder=5)

    # Add text labels for top 3
    for i, row in survivors.head(3).iterrows():
        ax.annotate(row['surface_id'].split('_')[0],
                   (row['dft_e_ads_oh'], row['dft_e_ads_o']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold')

    ax.set_xlabel('G$_{OH}$ (eV)', fontsize=12)
    ax.set_ylabel('G$_{O}$ (eV)', fontsize=12)
    ax.set_title('Volcano Plot - DFT Results', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


def create_overpotential_bars(ax, survivors):
    """Create bar plot of overpotentials for survivors"""

    # Sort by DFT overpotential
    survivors_sorted = survivors.sort_values('dft_eta_ORR')

    # Create bars
    x = range(len(survivors_sorted))
    surface_ids = [s.replace('mp-', '').replace('mvc-', 'mvc') for s in survivors_sorted['surface_id']]

    # DFT values
    ax.bar(x, survivors_sorted['dft_eta_ORR'].values,
          color='gold', alpha=0.8, label='DFT')

    # GemNet predictions as points
    ax.scatter(x, survivors_sorted['gemnet_eta_ORR'].values,
              color='cyan', s=50, marker='o', edgecolors='black',
              linewidth=1, label='GemNet', zorder=5)

    # Add reference lines
    ax.axhline(y=0.43, color='red', linestyle='--', alpha=0.5, label='η = 0.43V')
    ax.axhline(y=0.40, color='orange', linestyle='--', alpha=0.5, label='η = 0.40V')

    ax.set_xticks(x)
    ax.set_xticklabels(surface_ids, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Overpotential η (V)', fontsize=12)
    ax.set_title('Survivors - Overpotential Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')


def create_correlation_plot(ax, all_dft):
    """Create GemNet vs DFT correlation plot"""

    # Separate by performance
    good = all_dft[all_dft['dft_eta_ORR'] < 0.40]
    acceptable = all_dft[(all_dft['dft_eta_ORR'] >= 0.40) & (all_dft['dft_eta_ORR'] < 0.43)]
    poor = all_dft[all_dft['dft_eta_ORR'] >= 0.43]

    # Plot different groups
    ax.scatter(good['gemnet_eta_ORR'], good['dft_eta_ORR'],
              c=colors['good'], s=50, alpha=0.7, label=f'Good (n={len(good)})')
    ax.scatter(acceptable['gemnet_eta_ORR'], acceptable['dft_eta_ORR'],
              c=colors['acceptable'], s=50, alpha=0.7, label=f'Acceptable (n={len(acceptable)})')
    ax.scatter(poor['gemnet_eta_ORR'], poor['dft_eta_ORR'],
              c=colors['poor'], s=30, alpha=0.5, label=f'Poor (n={len(poor)})')

    # Add diagonal line
    min_val = min(all_dft['gemnet_eta_ORR'].min(), all_dft['dft_eta_ORR'].min())
    max_val = max(all_dft['gemnet_eta_ORR'].max(), all_dft['dft_eta_ORR'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

    # Calculate R² and MAE
    from scipy import stats
    # Filter out outliers for statistics
    mask = all_dft['dft_eta_ORR'] < 5  # Remove extreme outliers
    filtered = all_dft[mask]
    r2 = stats.pearsonr(filtered['gemnet_eta_ORR'], filtered['dft_eta_ORR'])[0]**2
    mae = np.mean(np.abs(filtered['gemnet_eta_ORR'] - filtered['dft_eta_ORR']))

    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f} V',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('GemNet η (V)', fontsize=12)
    ax.set_ylabel('DFT η (V)', fontsize=12)
    ax.set_title('GemNet vs DFT Correlation', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.35, 0.45)
    ax.set_ylim(0.35, max(filtered['dft_eta_ORR'].max() + 0.1, 1.5))


def create_performance_pie(ax, all_dft):
    """Create pie chart of performance distribution"""

    # Count by performance category
    excellent = len(all_dft[all_dft['dft_eta_ORR'] < 0.30])
    good = len(all_dft[(all_dft['dft_eta_ORR'] >= 0.30) & (all_dft['dft_eta_ORR'] < 0.40)])
    acceptable = len(all_dft[(all_dft['dft_eta_ORR'] >= 0.40) & (all_dft['dft_eta_ORR'] < 0.43)])
    poor = len(all_dft[all_dft['dft_eta_ORR'] >= 0.43])

    sizes = [excellent, good, acceptable, poor]
    labels = [f'Excellent\n(η<0.30V)\nn={excellent}',
             f'Good\n(0.30≤η<0.40V)\nn={good}',
             f'Acceptable\n(0.40≤η<0.43V)\nn={acceptable}',
             f'Poor\n(η≥0.43V)\nn={poor}']
    colors_list = [colors['excellent'], colors['good'], colors['acceptable'], colors['poor']]

    # Create pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 9})

    ax.set_title('Performance Distribution', fontsize=12, fontweight='bold')


def create_error_histogram(ax, all_dft):
    """Create histogram of prediction errors"""

    # Filter out extreme outliers
    mask = np.abs(all_dft['error_eta']) < 2
    errors = all_dft[mask]['error_eta']

    # Create histogram
    n, bins, patches = ax.hist(errors, bins=20, color='skyblue',
                               edgecolor='black', alpha=0.7)

    # Color bars by error magnitude
    for i, patch in enumerate(patches):
        if abs(bins[i]) < 0.1:
            patch.set_facecolor('green')
        elif abs(bins[i]) < 0.2:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prediction Error (GemNet - DFT) [V]', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('GemNet Prediction Error Distribution', fontsize=12, fontweight='bold')

    # Add statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    ax.text(0.05, 0.95, f'MAE = {mae:.3f} V\nRMSE = {rmse:.3f} V',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_limiting_step_analysis(ax, survivors):
    """Analyze limiting steps for survivors"""

    # Count limiting steps
    step_counts = survivors['dft_limiting_step'].value_counts().sort_index()

    # Create bar plot
    steps = ['Step 1\n*OOH', 'Step 2\n*O', 'Step 3\n*OH', 'Step 4\nOH⁻']
    counts = [step_counts.get(i, 0) for i in range(1, 5)]

    bars = ax.bar(range(4), counts, color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'],
                  edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(steps)
    ax.set_ylabel('Number of Catalysts', fontsize=12)
    ax.set_title('Limiting Steps for Survivors', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


def print_final_summary():
    """Print a text summary of the final results"""

    survivors = pd.read_csv(SURVIVORS_FILE)

    print("\n" + "="*80)
    print("FINAL DFT-VALIDATED ORR CATALYST SURVIVORS")
    print("="*80)
    print(f"\nTotal survivors with η < 0.43V: {len(survivors)}")

    print("\nCOMPLETE LIST OF SURVIVORS (sorted by DFT overpotential):")
    print("-"*80)

    for idx, row in survivors.iterrows():
        print(f"\n{idx+1}. {row['surface_id']}")
        print(f"   Bulk ID: {row['bulk_id']}")
        print(f"   DFT η: {row['dft_eta_ORR']:.3f} V (U_lim = {row['dft_U_limiting']:.3f} V)")
        print(f"   GemNet η: {row['gemnet_eta_ORR']:.3f} V (error = {row['error_eta']:+.3f} V)")
        print(f"   Limiting step: {int(row['dft_limiting_step'])}")
        print(f"   Performance: {row['performance']}")

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY:")
    print("-"*80)
    print(f"Excellent (η < 0.30V): 0 catalysts")
    print(f"Good (0.30V ≤ η < 0.40V): {len(survivors[survivors['performance'] == 'Good'])} catalysts")
    print(f"Acceptable (0.40V ≤ η < 0.43V): {len(survivors[survivors['performance'] == 'Acceptable'])} catalysts")

    print("\n" + "="*80)


if __name__ == "__main__":
    print_final_summary()
    create_survivors_summary()
    print("\nVisualization complete!")