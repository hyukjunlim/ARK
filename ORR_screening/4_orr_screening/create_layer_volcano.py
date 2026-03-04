#!/usr/bin/env python3
"""
Create separate ORR volcano plot layers following design_component.md principles
Each layer saved as individual PNG file with dpi=300
Two versions with different axis limits:
  v1: OH (0-2), O (0.5-4.0)
  v2: OH (-0.5-2.0), O (0.5-4.5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path
import sys

# Set mathtext to regular (per design_component.md)
matplotlib.rcParams['mathtext.default'] = 'regular'

# Configuration
_REPO_ROOT = Path(__file__).resolve().parent.parent
O_FILE = _REPO_ROOT / "data" / "adsorption_energies" / "o_min_energy_oc20.csv"
OH_FILE = _REPO_ROOT / "data" / "adsorption_energies" / "oh_min_energy_oc20.csv"
FORK_ACTIVE_FILE = _REPO_ROOT / "data" / "screening_results" / "FORK_ORR_active_eta.csv"
GEMNET_FILE = _REPO_ROOT / "data" / "screening_results" / "GemNet_ORR_validated_eta.csv"
DFT_FILE = _REPO_ROOT / "data" / "screening_results" / "DFT_ORR_validated_final.csv"
OUTPUT_DIR = _REPO_ROOT / "data" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Font properties setup (from design_component.md)
try:
    sys.path.append('../../utility')
    from util import *
except ImportError:
    # Fallback font setup
    fs = 12      # Standard font size
    fss = 12     # Standard font size (secondary)
    fsss = 8     # Small font size
    fsl = 24     # Large font size

    font_properties_label = fm.FontProperties(family='DejaVu Sans', size=fs)
    font_properties_tick = fm.FontProperties(family='DejaVu Sans', size=fss)
    font_properties_annotate = fm.FontProperties(family='DejaVu Sans', size=fsss)

# Color scheme from design_component.md
colors = ['#840032', '#e59500', '#002642', 'gray']

# Axis limits - two versions
AXIS_VERSIONS = {
    'v1': {'x_min': 0.25, 'x_max': 1.5, 'y_min': 0.75, 'y_max': 4.25},
    'v2': {'x_min': -0.5, 'x_max': 2.0, 'y_min': 0.5, 'y_max': 4.5}
}

# Overpotential thresholds (from visualize_overpotential.py)
FORK_ETA_THRESHOLD = 0.80  # U_limiting > 0.43 V
GEMNET_ETA_THRESHOLD = 0.65  # U_limiting > 0.58 V
DFT_ETA_THRESHOLD_ALL = 0.43  # U_limiting > 0.8 V
DFT_ETA_THRESHOLD = 0.43  # U_limiting > 0.8 V


def apply_font_styling(ax):
    """Apply consistent font styling to axis tick labels"""
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)


def format_axis_labels(ax, xlabel, ylabel):
    """Apply standard axis labels with proper font properties"""
    ax.set_xlabel(xlabel, fontproperties=font_properties_label)
    ax.set_ylabel(ylabel, fontproperties=font_properties_label)
    apply_font_styling(ax)


def calculate_limiting_potential(G_OH, G_O):
    """Calculate ORR limiting potential"""
    G_OOH = G_OH + 3.2
    step1 = 4.92 - G_OOH
    step2 = G_OOH - G_O
    step3 = G_O - G_OH
    step4 = G_OH
    U_limiting = np.minimum(np.minimum(step1, step2), np.minimum(step3, step4))
    eta_ORR = 1.23 - U_limiting
    return U_limiting, eta_ORR


def find_eta_contour_boundary(eta_threshold=0.73, version='v1'):
    """
    Find the boundary of the region where η < eta_threshold
    Returns points that form the boundary polygon
    From visualize_overpotential.py
    """
    limits = AXIS_VERSIONS[version]

    # Define grid
    G_OH_range = np.linspace(limits['x_min'], limits['x_max'], 300)
    G_O_range = np.linspace(limits['y_min'], limits['y_max'], 300)
    G_OH_grid, G_O_grid = np.meshgrid(G_OH_range, G_O_range)

    # Calculate overpotential
    U_lim, eta = calculate_limiting_potential(G_OH_grid, G_O_grid)

    # Find contour at eta_threshold
    fig_temp, ax_temp = plt.subplots()
    cs = ax_temp.contour(G_OH_grid, G_O_grid, eta, levels=[eta_threshold])
    plt.close(fig_temp)

    # Extract contour paths
    paths = []
    for level_collection in cs.allsegs:
        for segment in level_collection:
            if len(segment) > 10:  # Only keep significant contours
                paths.append(segment)

    # Return the longest path (main boundary)
    if paths:
        longest = max(paths, key=len)
        return longest
    return None


def load_all_data():
    """Load all datasets"""
    print("Loading data...")

    # Load raw O and OH data
    o_df = pd.read_csv(O_FILE)
    oh_df = pd.read_csv(OH_FILE)

    # Merge on surface_id
    all_data = pd.merge(
        o_df[['surface_id', 'bulk_id', 'crack_e_ads', 'gemnet_e_ads']],
        oh_df[['surface_id', 'crack_e_ads', 'gemnet_e_ads']],
        on='surface_id',
        suffixes=('_o', '_oh')
    )

    # Load screening results (eta-filtered)
    fork_active = pd.read_csv(FORK_ACTIVE_FILE)
    gemnet_validated = pd.read_csv(GEMNET_FILE)
    gemnet_validated_dft = gemnet_validated[gemnet_validated['gemnet_eta_ORR'] < DFT_ETA_THRESHOLD]

    # Load DFT results
    dft_data = None
    if DFT_FILE.exists():
        dft_data = pd.read_csv(DFT_FILE)
        # Merge DFT data with GemNet validated data
        gemnet_validated_dft = pd.merge(gemnet_validated_dft, dft_data, on='surface_id', how='inner')
        print(f"  DFT results loaded: {len(dft_data)} surfaces")
        print(f"  DFT-GemNet matched: {len(gemnet_validated_dft)} surfaces")
        gemnet_validated_dft_threshold = gemnet_validated_dft[gemnet_validated_dft['gemnet_eta_ORR'] < DFT_ETA_THRESHOLD_ALL]

    print(f"  All data: {len(all_data)} samples")
    print(f"  FORK active (η<{FORK_ETA_THRESHOLD}V): {len(fork_active)} samples")
    print(f"  GemNet validated (η<{GEMNET_ETA_THRESHOLD}V): {len(gemnet_validated)} samples")
    print(f"  GemNet validated for DFT (η<{DFT_ETA_THRESHOLD}V): {len(gemnet_validated_dft)} samples")
    return all_data, fork_active, gemnet_validated, gemnet_validated_dft, gemnet_validated_dft_threshold

def create_base_volcano(ax, version='v1'):
    """Create background volcano contour (used in all layers)"""
    limits = AXIS_VERSIONS[version]

    # Define grid for background volcano
    G_OH_range = np.linspace(limits['x_min'], limits['x_max'], 200)
    G_O_range = np.linspace(limits['y_min'], limits['y_max'], 200)
    G_OH_grid, G_O_grid = np.meshgrid(G_OH_range, G_O_range)

    # Calculate limiting potential for background
    U_limiting, eta_ORR = calculate_limiting_potential(G_OH_grid, G_O_grid)

    # Create contour plot with inferno colormap - NO ALPHA per user request
    levels = np.arange(-0.5, 1.3, 0.1)
    contour = ax.contourf(G_OH_grid, G_O_grid, U_limiting, levels,
                         cmap='inferno', extend='both')

    # Add contour lines
    contour_lines = ax.contour(G_OH_grid, G_O_grid, U_limiting,
                               levels=[0.5, 0.8],
                               colors='white', linewidths=1.0, alpha=0.8, linestyles='-',zorder=10)
    # Replace the problematic line with:
    labels = ax.clabel(contour_lines, inline=True, fmt='%.2f V')

    # Then set font properties on the returned text objects:
    for label in labels:
        label.set_fontproperties(font_properties_tick)
    # Add scaling relation line
    G_OH_line = np.linspace(limits['x_min'], limits['x_max'], 100)
    #ax.plot(G_OH_line, 2 * G_OH_line, ':', color='cyan', linewidth=1.0,
    #       alpha=0.6, label='G$_{O}$ = 2$\\times$G$_{OH}$')

    return contour


def setup_axis(ax, version='v1'):
    """Setup axis properties following design_component.md"""
    limits = AXIS_VERSIONS[version]

    format_axis_labels(ax, 'G$_{OH}$ (eV)', 'G$_{O}$ (eV)')
    ax.set_xlim(limits['x_min'], limits['x_max'])
    ax.set_ylim(limits['y_min'], limits['y_max'])

    # Add minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    # No grid (per design_component.md)

def create_layer_0(all_data, version='v1'):
    """Layer 1: All datapoints (background)"""
    print(f"  Layer 0 [{version}]: All datapoints (background)")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create base volcano
    contour = create_base_volcano(ax, version)

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=np.arange(-0.4, 1.3, 0.2),
                       orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('U$_{limiting}$ (V)', rotation=90, labelpad=15,
                       fontproperties=font_properties_label)
    cbar.ax.tick_params(labelsize=fss)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)

    # Sample all data for background

    setup_axis(ax, version)
    ax.set_title('Layer 0: All Datapoints', fontproperties=font_properties_label, pad=10)

    # Legend with proper font
    #legend = ax.legend(loc='upper right', prop=font_properties_tick, framealpha=0.9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'layer_0_{version}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)

def create_layer_1(all_data, version='v1'):
    """Layer 1: All datapoints (background)"""
    print(f"  Layer 1 [{version}]: All datapoints (background)")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create base volcano
    contour = create_base_volcano(ax, version)

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=np.arange(-0.4, 1.3, 0.2),
                       orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('U$_{limiting}$ (V)', rotation=90, labelpad=15,
                       fontproperties=font_properties_label)
    cbar.ax.tick_params(labelsize=fss)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)

    # Sample all data for background
    sample_size = min(300000, len(all_data))
    all_sample = all_data.sample(n=sample_size, random_state=42)

    # Plot all datapoints - white color as requested
    ax.scatter(all_sample['crack_e_ads_oh'], all_sample['crack_e_ads_o'],
              c='white', s=25, alpha=0.5,
              edgecolors='none', rasterized=True,
              label=f'All data (n={len(all_data)})', zorder=1)

    setup_axis(ax, version)

    # Legend with proper font
    #legend = ax.legend(loc='upper right', prop=font_properties_tick, framealpha=0.9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'layer_1_{version}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)

def create_layer_2(fork_active, all_data, version='v1'):
    """Layer 2: FORK predicted (FORK active)"""
    print(f"  Layer 2 [{version}]: FORK active")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create base volcano
    contour = create_base_volcano(ax, version)

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=np.arange(-0.4, 1.3, 0.2),
                       orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('U$_{limiting}$ (V)', rotation=90, labelpad=15,
                       fontproperties=font_properties_label)
    cbar.ax.tick_params(labelsize=fss)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)


    sample_size = min(300000, len(all_data))
    all_sample = all_data.sample(n=sample_size, random_state=42)

    # Plot all datapoints - white color as requested
    ax.scatter(all_sample['crack_e_ads_oh'], all_sample['crack_e_ads_o'],
              c='white', s=25, alpha=0.5,
              edgecolors='none', rasterized=True,
              label=f'All data (n={len(all_data)})', zorder=1)

    # FORK active - style from visualize_overpotential.py
    ax.scatter(fork_active['crack_e_ads_oh'], fork_active['crack_e_ads_o'],
              c='gray', s=50, alpha=0.7,
              edgecolors='none',
              label=f'FORK active (n={len(fork_active)})', zorder=3)

    setup_axis(ax, version)

    # Legend with proper font
    #legend = ax.legend(loc='upper right', prop=font_properties_tick, framealpha=0.9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'layer_2_{version}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def create_layer_3(gemnet_validated, all_data, version='v1'):
    """Layer 3: GemNet predicted (GemNet validated) with FORK connection lines"""
    print(f"  Layer 3 [{version}]: GemNet validated with FORK connections")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create base volcano
    contour = create_base_volcano(ax, version)

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=np.arange(-0.4, 1.3, 0.2),
                       orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('U$_{limiting}$ (V)', rotation=90, labelpad=15,
                       fontproperties=font_properties_label)
    cbar.ax.tick_params(labelsize=fss)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)

    # Sample all data for background
    sample_size = min(300000, len(all_data))
    all_sample = all_data.sample(n=sample_size, random_state=42)

    # Plot all datapoints - white color as requested
    ax.scatter(all_sample['crack_e_ads_oh'], all_sample['crack_e_ads_o'],
              c='white', s=25, alpha=0.5,
              edgecolors='none', rasterized=True,
              label=f'All data (n={len(all_data)})', zorder=1)

    # Draw connecting lines between FORK and GemNet predictions
    for idx, row in gemnet_validated.iterrows():
        fork_oh = row['crack_e_ads_oh']
        fork_o = row['crack_e_ads_o']
        gemnet_oh = row['gemnet_e_ads_oh']
        gemnet_o = row['gemnet_e_ads_o']

        # Connection line - style from visualize_overpotential.py
        ax.plot([fork_oh, gemnet_oh], [fork_o, gemnet_o],
               color='white', linewidth=1.0, linestyle='-', alpha=0.3, zorder=4)

    # FORK predictions (hollow circles)
    ax.scatter(gemnet_validated['crack_e_ads_oh'],
              gemnet_validated['crack_e_ads_o'],
              c='gray', s=50, alpha=0.7,
              label=f'FORK pred', zorder=3)

    # GemNet validated (filled circles)
    ax.scatter(gemnet_validated['gemnet_e_ads_oh'],
              gemnet_validated['gemnet_e_ads_o'],
              s=50, alpha=0.7, color='black', edgecolors='black',
              label=f'GemNet validated (n={len(gemnet_validated)})', zorder=6)

    # Add η < 0.65V region (contour-based polygon - lime border only)
    #boundary_gemnet = find_eta_contour_boundary(eta_threshold=GEMNET_ETA_THRESHOLD, version=version)
    #if boundary_gemnet is not None:
    #    polygon_gemnet = Polygon(boundary_gemnet, alpha=0.0, facecolor='none',
    #                            edgecolor='lime', linewidth=3, linestyle='-',
    #                            label=f'η < {GEMNET_ETA_THRESHOLD}V region', zorder=2)
    #     ax.add_patch(polygon_gemnet)

    # Highlight top 5 with gold star markers - style from visualize_overpotential.py
    #top5 = gemnet_validated.nsmallest(5, 'gemnet_eta_ORR')
    #ax.scatter(top5['gemnet_e_ads_oh'], top5['gemnet_e_ads_o'],
    #          marker='*', s=500, facecolors='gold', edgecolors='black',
    #          linewidth=2.5, label='Top 5', zorder=7)

    setup_axis(ax, version)

    # Legend with proper font
    #legend = ax.legend(loc='upper right', prop=font_properties_tick, framealpha=0.9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'layer_3_{version}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def create_layer_4(gemnet_validated, gemnet_validated_dft, all_data, version='v1'):
    """Layer 4: FORK-GemNet comparison with connecting lines and screening boxes"""
    print(f"  Layer 4 [{version}]: FORK-GemNet comparison with screening boxes")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create base volcano
    contour = create_base_volcano(ax, version)

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=np.arange(-0.4, 1.3, 0.2),
                       orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('U$_{limiting}$ (V)', rotation=90, labelpad=15,
                       fontproperties=font_properties_label)
    cbar.ax.tick_params(labelsize=fss)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)
    # Sample all data for background
    sample_size = min(300000, len(all_data))
    all_sample = all_data.sample(n=sample_size, random_state=42)

    # Plot all datapoints - white color as requested
    ax.scatter(all_sample['crack_e_ads_oh'], all_sample['crack_e_ads_o'],
              c='white', s=25, alpha=0.5,
              edgecolors='none', rasterized=True,
              label=f'All data (n={len(all_data)})', zorder=1)
    # Draw connecting lines between FORK and GemNet predictions
    for idx, row in gemnet_validated.iterrows():
        fork_oh = row['crack_e_ads_oh']
        fork_o = row['crack_e_ads_o']
        gemnet_oh = row['gemnet_e_ads_oh']
        gemnet_o = row['gemnet_e_ads_o']

        # Connection line - style from visualize_overpotential.py
        ax.plot([fork_oh, gemnet_oh], [fork_o, gemnet_o],
               color='white', linewidth=1.0, linestyle='-', alpha=0.3, zorder=4)
    # FORK predictions (hollow circles)
    ax.scatter(gemnet_validated['crack_e_ads_oh'],
              gemnet_validated['crack_e_ads_o'],
              c='gray', s=50, alpha=0.7,
              label=f'FORK pred', zorder=3)
    ax.scatter(gemnet_validated['gemnet_e_ads_oh'],
              gemnet_validated['gemnet_e_ads_o'],
              s=50, alpha=0.7, color='black', edgecolors='black',
              label=f'GemNet validated (n={len(gemnet_validated)})', zorder=5)
    # GemNet predictions (filled circles) - style from visualize_overpotential.py
    #ax.scatter(gemnet_validated['gemnet_e_ads_oh'],
    #          gemnet_validated['gemnet_e_ads_o'],
    #          c=gemnet_validated['gemnet_eta_ORR'], s=50,
    #          cmap='inferno_r', edgecolors='white',
    #          label=f'GemNet validated', zorder=5)
    # GemNet predictions (filled circles) - style from visualize_overpotential.py
    ax.scatter(gemnet_validated_dft['gemnet_e_ads_oh'],
              gemnet_validated_dft['gemnet_e_ads_o'],
              s=50, alpha=0.7, color='cyan', edgecolors='cyan',marker='o',
              label=f'GemNet validated for DFT (n={len(gemnet_validated_dft)})', zorder=5)

    setup_axis(ax, version)

    # Legend with proper font
    #legend = ax.legend(loc='upper right', prop=font_properties_tick, framealpha=0.9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'layer_4_{version}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def create_layer_5(gemnet_validated, gemnet_validated_dft, gemnet_validated_dft_threshold, all_data, version='v1'):
    """Layer 5: DFT evaluated samples with GemNet-DFT connecti`ons (filtered for eta < 0.73)"""
    print(f"  Layer 5 [{version}]: DFT evaluated samples (eta < 0.73)")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create base volcano
    contour = create_base_volcano(ax, version)

    # Add colorbar
    cbar = plt.colorbar(contour, ticks=np.arange(-0.4, 1.3, 0.2),
                       orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('U$_{limiting}$ (V)', rotation=90, labelpad=15,
                       fontproperties=font_properties_label)
    cbar.ax.tick_params(labelsize=fss)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)

    # Sample all data for background
    sample_size = min(300000, len(all_data))
    all_sample = all_data.sample(n=sample_size, random_state=42)

    # Plot all datapoints - white color as requested
    ax.scatter(all_sample['crack_e_ads_oh'], all_sample['crack_e_ads_o'],
              c='white', s=25, alpha=0.5,
              edgecolors='none', rasterized=True,
              label=f'All data (n={len(all_data)})', zorder=1)

    # Draw connecting lines between GemNet and DFT predictions for DFT-evaluated samples
    if 'dft_e_ads_oh' in gemnet_validated_dft.columns:
        # Calculate DFT eta and filter for eta < 0.73
        dft_filtered = gemnet_validated_dft.copy()
        dft_filtered['dft_U_limiting'], dft_filtered['dft_eta_ORR'] = calculate_limiting_potential(
            dft_filtered['dft_e_ads_oh'], dft_filtered['dft_e_ads_o']
        )
        dft_filtered_all = dft_filtered[dft_filtered['dft_eta_ORR'] < 0.73]
        dft_filtered_real = dft_filtered[dft_filtered['dft_eta_ORR'] < DFT_ETA_THRESHOLD_ALL]
        print(f"    DFT samples after eta < 0.73 filter: {len(dft_filtered)} / {len(gemnet_validated_dft)}")

        for idx, row in dft_filtered_all.iterrows():
            gemnet_oh = row['gemnet_e_ads_oh']
            gemnet_o = row['gemnet_e_ads_o']
            dft_oh = row['dft_e_ads_oh']
            dft_o = row['dft_e_ads_o']

            # Connection line between GemNet and DFT
            ax.plot([gemnet_oh, dft_oh], [gemnet_o, dft_o],
                   color='cyan', linewidth=1.5, linestyle='-', alpha=0.6, zorder=4)

        ax.scatter(gemnet_validated_dft['gemnet_e_ads_oh'],
                gemnet_validated_dft['gemnet_e_ads_o'],
                s=50, alpha=0.7, color='cyan', edgecolors='cyan',marker='o',
                label=f'GemNet validated for DFT (n={len(gemnet_validated_dft)})', zorder=5)

        # Actual DFT results (yellow stars) - only those with eta < 0.73
        ax.scatter(dft_filtered_all['dft_e_ads_oh'],
                  dft_filtered_all['dft_e_ads_o'],
                  s=50, alpha=0.9, color='yellow', marker='o',
                  edgecolors='black', linewidth=1.5,
                  label=f'DFT results (η < 0.73, n={len(dft_filtered)})', zorder=6)
        ax.scatter(dft_filtered_real['dft_e_ads_oh'],
                  dft_filtered_real['dft_e_ads_o'],
                  s=200, alpha=0.9, color='yellow', marker='*',
                  edgecolors='black', linewidth=1.5,
                  label=f'DFT results (η < 0.73, n={len(dft_filtered)})', zorder=6)

    setup_axis(ax, version)

    # Legend with proper font
    #legend = ax.legend(loc='upper right', prop=font_properties_tick, framealpha=0.9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'layer_5_{version}.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def create_screening_funnel_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft):
    """Create screening funnel plot showing the number of candidates at each stage"""
    print("\n  Creating screening funnel plot...")

    # Calculate DFT samples with eta < 0.73
    dft_active = 0
    if 'dft_e_ads_oh' in gemnet_validated_dft.columns:
        dft_filtered = gemnet_validated_dft.copy()
        dft_filtered['dft_U_limiting'], dft_filtered['dft_eta_ORR'] = calculate_limiting_potential(
            dft_filtered['dft_e_ads_oh'], dft_filtered['dft_e_ads_o']
        )
        dft_active = len(dft_filtered[dft_filtered['dft_eta_ORR'] < 0.73])

    stages = ['All Data', f'FORK\n(η<{FORK_ETA_THRESHOLD}V)', f'GemNet\n(η<{GEMNET_ETA_THRESHOLD}V)',
              f'DFT Calc.\n(η<{DFT_ETA_THRESHOLD}V)', f'DFT Active\n(η<0.73V)']
    counts = [len(all_data), len(fork_active), len(gemnet_validated),
              len(gemnet_validated_dft), dft_active]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    bars = ax.bar(stages, counts, color=['#002642', '#840032', '#e59500', 'cyan', 'yellow'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count:,}',
               ha='center', va='bottom', fontproperties=font_properties_tick, fontsize=10)

    # Add screening rate percentages
    for i in range(1, len(counts)):
        if counts[i-1] > 0:
            rate = (counts[i] / counts[i-1]) * 100
            ax.text(i, counts[i] * 0.5, f'{rate:.2f}%',
                   ha='center', va='center', fontproperties=font_properties_tick,
                   fontsize=9, color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    ax.set_ylabel('Number of Candidates', fontproperties=font_properties_label)
    ax.set_title('Screening Funnel: FORK → GemNet → DFT', fontproperties=font_properties_label, pad=15)
    ax.set_yscale('log')
    apply_font_styling(ax)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'screening_funnel.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def create_success_rate_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft):
    """Create success rate plot showing retention at each screening stage"""
    print("\n  Creating success rate plot...")

    # Calculate DFT samples with eta < 0.73
    dft_active = 0
    if 'dft_e_ads_oh' in gemnet_validated_dft.columns:
        dft_filtered = gemnet_validated_dft.copy()
        dft_filtered['dft_U_limiting'], dft_filtered['dft_eta_ORR'] = calculate_limiting_potential(
            dft_filtered['dft_e_ads_oh'], dft_filtered['dft_e_ads_o']
        )
        dft_active = len(dft_filtered[dft_filtered['dft_eta_ORR'] < 0.73])

    # Calculate success rates (percentage of previous stage)
    stages = [f'FORK\n(η<{FORK_ETA_THRESHOLD}V)', f'GemNet\n(η<{GEMNET_ETA_THRESHOLD}V)',
              f'DFT Calc.\n(η<{DFT_ETA_THRESHOLD}V)', f'DFT Active\n(η<0.73V)']

    fork_rate = (len(fork_active) / len(all_data)) * 100
    gemnet_rate = (len(gemnet_validated) / len(fork_active)) * 100 if len(fork_active) > 0 else 0
    dft_calc_rate = (len(gemnet_validated_dft) / len(gemnet_validated)) * 100 if len(gemnet_validated) > 0 else 0
    dft_active_rate = (dft_active / len(gemnet_validated_dft)) * 100 if len(gemnet_validated_dft) > 0 else 0

    success_rates = [fork_rate, gemnet_rate, dft_calc_rate, dft_active_rate]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    bars = ax.bar(stages, success_rates, color=['#840032', '#e59500', 'cyan', 'yellow'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.2f}%',
               ha='center', va='bottom', fontproperties=font_properties_tick, fontsize=10)

    # Add horizontal reference line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylabel('Success Rate (%)', fontproperties=font_properties_label)
    ax.set_ylim(0, max(success_rates) * 1.2)
    ax.set_title('Stage Success Rate (% Retained from Previous Stage)',
                 fontproperties=font_properties_label, pad=15)
    apply_font_styling(ax)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'success_rate.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def create_filtering_distribution_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft):
    """Create stacked bar chart showing filtered vs retained samples at each stage"""
    print("\n  Creating filtering distribution plot...")

    # Calculate DFT samples with eta < 0.73
    dft_active = 0
    if 'dft_e_ads_oh' in gemnet_validated_dft.columns:
        dft_filtered = gemnet_validated_dft.copy()
        dft_filtered['dft_U_limiting'], dft_filtered['dft_eta_ORR'] = calculate_limiting_potential(
            dft_filtered['dft_e_ads_oh'], dft_filtered['dft_e_ads_o']
        )
        dft_active = len(dft_filtered[dft_filtered['dft_eta_ORR'] < 0.73])

    # Calculate retained and filtered counts
    stages = ['All Data →\nFORK', 'FORK →\nGemNet', 'GemNet →\nDFT Calc.', 'DFT Calc. →\nDFT Active']

    # Layer 1 → 2: All Data → FORK
    retained_1 = len(fork_active)
    filtered_1 = len(all_data) - len(fork_active)

    # Layer 2 → 3: FORK → GemNet
    retained_2 = len(gemnet_validated)
    filtered_2 = len(fork_active) - len(gemnet_validated)

    # Layer 3 → 4: GemNet → DFT Calculated
    retained_3 = len(gemnet_validated_dft)
    filtered_3 = len(gemnet_validated) - len(gemnet_validated_dft)

    # Layer 4 → 5: DFT Calculated → DFT Active
    retained_4 = dft_active
    filtered_4 = len(gemnet_validated_dft) - dft_active

    retained = [retained_1, retained_2, retained_3, retained_4]
    filtered = [filtered_1, filtered_2, filtered_3, filtered_4]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(stages))
    width = 0.6

    # Create stacked bars
    bars1 = ax.bar(x, retained, width, label='Retained (Passed)',
                   color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x, filtered, width, bottom=retained, label='Filtered Out',
                   color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for i, (ret, filt) in enumerate(zip(retained, filtered)):
        total = ret + filt

        # Retained count and percentage
        if ret > 0:
            ret_pct = (ret / total) * 100
            ax.text(i, ret/2, f'{ret:,}\n({ret_pct:.1f}%)',
                   ha='center', va='center', fontproperties=font_properties_tick,
                   fontsize=9, color='white', weight='bold')

        # Filtered count and percentage
        if filt > 0:
            filt_pct = (filt / total) * 100
            ax.text(i, ret + filt/2, f'{filt:,}\n({filt_pct:.1f}%)',
                   ha='center', va='center', fontproperties=font_properties_tick,
                   fontsize=9, color='white', weight='bold')

    ax.set_ylabel('Number of Samples', fontproperties=font_properties_label)
    ax.set_xlabel('Screening Transition', fontproperties=font_properties_label)
    ax.set_title('Filtering Distribution: Retained vs Filtered at Each Stage',
                 fontproperties=font_properties_label, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_yscale('log')
    apply_font_styling(ax)

    # Legend
    legend = ax.legend(loc='upper right', prop=font_properties_tick, framealpha=0.9)

    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'filtering_distribution.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def create_cumulative_filtering_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft):
    """Create area chart showing cumulative filtering effect"""
    print("\n  Creating cumulative filtering plot...")

    # Calculate DFT samples with eta < 0.73
    dft_active = 0
    if 'dft_e_ads_oh' in gemnet_validated_dft.columns:
        dft_filtered = gemnet_validated_dft.copy()
        dft_filtered['dft_U_limiting'], dft_filtered['dft_eta_ORR'] = calculate_limiting_potential(
            dft_filtered['dft_e_ads_oh'], dft_filtered['dft_e_ads_o']
        )
        dft_active = len(dft_filtered[dft_filtered['dft_eta_ORR'] < 0.73])

    # Stages and counts
    stages = ['All Data', 'FORK\nActive', 'GemNet\nValidated', 'DFT\nCalculated', 'DFT\nActive']
    counts = [len(all_data), len(fork_active), len(gemnet_validated),
              len(gemnet_validated_dft), dft_active]

    # Calculate cumulative filtered
    cumulative_filtered = [0]
    for i in range(len(counts)-1):
        cumulative_filtered.append(counts[0] - counts[i+1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Remaining samples
    x = np.arange(len(stages))
    ax1.fill_between(x, 0, counts, alpha=0.6, color='#2ecc71', label='Remaining Samples')
    ax1.plot(x, counts, 'o-', color='#27ae60', linewidth=2, markersize=8, markeredgecolor='black')

    for i, (stage, count) in enumerate(zip(stages, counts)):
        ax1.text(i, count, f'{count:,}', ha='center', va='bottom',
                fontproperties=font_properties_tick, fontsize=9, weight='bold')

    ax1.set_ylabel('Number of Samples', fontproperties=font_properties_label)
    ax1.set_xlabel('Screening Stage', fontproperties=font_properties_label)
    ax1.set_title('Remaining Samples After Each Screening Stage',
                  fontproperties=font_properties_label, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    apply_font_styling(ax1)

    # Right plot: Cumulative filtered
    ax2.fill_between(x, 0, cumulative_filtered, alpha=0.6, color='#e74c3c', label='Cumulative Filtered')
    ax2.plot(x, cumulative_filtered, 'o-', color='#c0392b', linewidth=2, markersize=8, markeredgecolor='black')

    for i, (stage, filt) in enumerate(zip(stages, cumulative_filtered)):
        if filt > 0:
            pct = (filt / len(all_data)) * 100
            ax2.text(i, filt, f'{filt:,}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontproperties=font_properties_tick, fontsize=9, weight='bold')

    ax2.set_ylabel('Cumulative Filtered Count', fontproperties=font_properties_label)
    ax2.set_xlabel('Screening Stage', fontproperties=font_properties_label)
    ax2.set_title('Cumulative Samples Filtered Out',
                  fontproperties=font_properties_label, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    apply_font_styling(ax2)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'cumulative_filtering.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    → {output_file}")
    plt.close(fig)


def main():
    print("\n" + "="*60)
    print("ORR VOLCANO PLOT - SEPARATE LAYERS")
    print("Design Component Compliant | Two Axis Versions")
    print("="*60 + "\n")

    # Load data
    all_data, fork_active, gemnet_validated, gemnet_validated_dft, gemnet_validated_dft_threshold = load_all_data()

    # Create each layer for both versions
    for version in ['v1']:
        limits = AXIS_VERSIONS[version]
        print(f"\n{'='*60}")
        print(f"Creating {version.upper()}: OH [{limits['x_min']}, {limits['x_max']}] | O [{limits['y_min']}, {limits['y_max']}]")
        print(f"{'='*60}")
        create_layer_0(all_data, version)
        create_layer_1(all_data, version)
        create_layer_2(fork_active, all_data, version)
        create_layer_3(gemnet_validated, all_data, version)
        create_layer_4(gemnet_validated, gemnet_validated_dft, all_data, version)
        create_layer_5(gemnet_validated, gemnet_validated_dft, gemnet_validated_dft_threshold, all_data, version)

    # Create screening analysis plots
    print(f"\n{'='*60}")
    print("Creating Screening Analysis Plots")
    print(f"{'='*60}")
    create_screening_funnel_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft)
    create_success_rate_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft)
    create_filtering_distribution_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft)
    create_cumulative_filtering_plot(all_data, fork_active, gemnet_validated, gemnet_validated_dft)

    # Print some statistics if DFT data is available
    if 'dft_e_ads_oh' in gemnet_validated_dft.columns:
        # Calculate DFT active count
        dft_filtered = gemnet_validated_dft_threshold.copy()
        dft_filtered['dft_U_limiting'], dft_filtered['dft_eta_ORR'] = calculate_limiting_potential(
            dft_filtered['dft_e_ads_oh'], dft_filtered['dft_e_ads_o']
        )
        dft_active = dft_filtered[dft_filtered['dft_eta_ORR'] < DFT_ETA_THRESHOLD_ALL]

        print("\n" + "="*60)
        print("SCREENING STATISTICS")
        print("="*60)
        print(f"All Data:           {len(all_data):,}")
        print(f"FORK Active:        {len(fork_active):,} ({(len(fork_active)/len(all_data)*100):.2f}%)")
        print(f"GemNet Validated:   {len(gemnet_validated):,} ({(len(gemnet_validated)/len(fork_active)*100):.2f}% of FORK)")
        print(f"DFT Calculated:     {len(gemnet_validated_dft):,} ({(len(gemnet_validated_dft)/len(gemnet_validated)*100):.2f}% of GemNet)")
        print(f"DFT Active (η<0.73): {len(dft_active):,} ({(len(dft_active)/len(gemnet_validated_dft)*100):.2f}% of DFT)")

        print("\nDFT-GemNet Comparison Statistics:")
        mae_oh = np.mean(np.abs(gemnet_validated_dft['gemnet_e_ads_oh'] - gemnet_validated_dft['dft_e_ads_oh']))
        mae_o = np.mean(np.abs(gemnet_validated_dft['gemnet_e_ads_o'] - gemnet_validated_dft['dft_e_ads_o']))
        print(f"  OH MAE: {mae_oh:.3f} eV")
        print(f"  O MAE: {mae_o:.3f} eV")

    print("\n" + "="*60)
    print("✅ All layers created successfully!")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Files: layer_0_v1.png, ... layer_5_v1.png")
    print(f"       screening_funnel.png, success_rate.png")
    print(f"       filtering_distribution.png, cumulative_filtering.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
