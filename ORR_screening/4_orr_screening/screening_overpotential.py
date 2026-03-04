#!/usr/bin/env python3
"""
ORR Catalyst Screening Pipeline - Overpotential-based approach
Filters based on overpotential thresholds instead of absolute energy ranges
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
matplotlib.use('Agg')  # Non-interactive backend
matplotlib.rcParams['mathtext.default'] = 'regular'

# Font properties setup
fs = 12      # Standard font size
fss = 12     # Standard font size (secondary)
fsss = 8     # Small font size
fsl = 24     # Large font size
font_properties_label = fm.FontProperties(family='DejaVu Sans', size=fs)
font_properties_tick = fm.FontProperties(family='DejaVu Sans', size=fss)
font_properties_annotate = fm.FontProperties(family='DejaVu Sans', size=fsss)

# Standard color palette
colors = ['#840032', '#e59500', '#002642', 'gray']

# Configuration — paths relative to repo root (data/adsorption_energies/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
O_ADS_DIR = _REPO_ROOT / "data" / "adsorption_energies"
OH_ADS_DIR = _REPO_ROOT / "data" / "adsorption_energies"
O_FILE = O_ADS_DIR / "o_min_energy_oc20.csv"
OH_FILE = OH_ADS_DIR / "oh_min_energy_oc20.csv"
OUTPUT_DIR = _REPO_ROOT / "data" / "screening_results"
OUTPUT_DIR.mkdir(exist_ok=True)
CONFUSION_OUTPUT_DIR = _REPO_ROOT / "data" / "screening_results" / "confusion_output"
CONFUSION_OUTPUT_DIR.mkdir(exist_ok=True)

# ORR thermodynamic parameters
U_0_ORR = 1.23  # Standard potential for ORR (V vs. SHE)
G_OOH_SCALING = 3.2  # G_OOH = G_OH + 3.2 eV (scaling relation)

# Overpotential-based screening criteria
FORK_ETA_THRESHOLD = 0.73  # FORK: η < 0.73 V (U_limiting > 0.5 V)
GEMNET_ETA_THRESHOLD = 0.73  # GemNet: η < 0.73 V (U_limiting > 0.5 V)


def calculate_orr_metrics(G_OH, G_O):
    """
    Calculate ORR limiting potential and overpotential

    Args:
        G_OH: OH adsorption free energy (eV)
        G_O: O adsorption free energy (eV)

    Returns:
        tuple: (U_limiting, eta_ORR, limiting_step)
    """
    # Apply scaling relation: G_OOH = G_OH + 3.2 eV
    G_OOH = G_OH + G_OOH_SCALING

    # Calculate potentials for each elementary step
    step1 = 4.92 - G_OOH
    step2 = G_OOH - G_O
    step3 = G_O - G_OH
    step4 = G_OH

    # Limiting potential is the minimum (most endergonic step)
    steps = np.array([step1, step2, step3, step4])
    U_limiting = np.min(steps)
    limiting_step = np.argmin(steps) + 1

    # Overpotential: eta = U_0 - U_limiting
    eta_ORR = U_0_ORR - U_limiting

    return U_limiting, eta_ORR, limiting_step


def load_data():
    """Load O and OH adsorption data"""
    print("Loading data...")
    o_df = pd.read_csv(O_FILE)
    oh_df = pd.read_csv(OH_FILE)

    print(f"  O data: {len(o_df)} samples")
    print(f"  OH data: {len(oh_df)} samples")

    return o_df, oh_df


def merge_data(o_df, oh_df):
    """Merge O and OH data on surface_id"""
    print("\nMerging O and OH data...")
    merged = pd.merge(
        o_df[['surface_id', 'bulk_id', 'crack_e_ads', 'gemnet_e_ads']],
        oh_df[['surface_id', 'crack_e_ads', 'gemnet_e_ads']],
        on='surface_id',
        suffixes=('_o', '_oh')
    )
    print(f"  Merged: {len(merged)} samples")
    return merged


def screen_fork_overpotential(merged_df):
    """
    Screen using FORK model based on overpotential threshold

    Criterion: η_ORR < FORK_ETA_THRESHOLD
    """
    print("\n" + "="*60)
    print("FORK SCREENING (OVERPOTENTIAL-BASED)")
    print("="*60)

    df = merged_df.copy()

    # Calculate ORR metrics using FORK energies
    results = df.apply(
        lambda row: calculate_orr_metrics(row['crack_e_ads_oh'], row['crack_e_ads_o']),
        axis=1
    )

    df['fork_U_limiting'] = [r[0] for r in results]
    df['fork_eta_ORR'] = [r[1] for r in results]
    df['fork_limiting_step'] = [r[2] for r in results]

    # Apply overpotential threshold
    print(f"\nInitial samples: {len(df)}")
    print(f"FORK criterion: η_ORR < {FORK_ETA_THRESHOLD} V (U_limiting > {U_0_ORR - FORK_ETA_THRESHOLD:.2f} V)")

    df_active = df[df['fork_eta_ORR'] < FORK_ETA_THRESHOLD].copy()
    print(f"FORK active: {len(df_active)}/{len(df)} samples ({len(df_active)/len(df)*100:.1f}%)")

    # Statistics
    if len(df_active) > 0:
        print(f"\nFORK Active Catalysts Statistics:")
        print(f"  G_OH range: [{df_active['crack_e_ads_oh'].min():.2f}, {df_active['crack_e_ads_oh'].max():.2f}] eV")
        print(f"  G_O range: [{df_active['crack_e_ads_o'].min():.2f}, {df_active['crack_e_ads_o'].max():.2f}] eV")
        print(f"  U_limiting range: [{df_active['fork_U_limiting'].min():.2f}, {df_active['fork_U_limiting'].max():.2f}] V")
        print(f"  η_ORR range: [{df_active['fork_eta_ORR'].min():.2f}, {df_active['fork_eta_ORR'].max():.2f}] V")
        print(f"  Best catalyst: η = {df_active['fork_eta_ORR'].min():.3f} V (surface_id: {df_active.loc[df_active['fork_eta_ORR'].idxmin(), 'surface_id']})")

    return df, df_active  # Return both full dataset and active set


def screen_gemnet_overpotential(fork_active_df, merged_df):
    """
    Validate FORK active catalysts using GemNet model

    Criterion: η_ORR < GEMNET_ETA_THRESHOLD
    """
    print("\n" + "="*60)
    print("GEMNET VALIDATION (OVERPOTENTIAL-BASED)")
    print("="*60)

    fork_ids = fork_active_df['surface_id'].unique()
    df = merged_df[merged_df['surface_id'].isin(fork_ids)].copy()

    print(f"\nValidating {len(df)} FORK active samples with GemNet...")
    print(f"GemNet criterion: η_ORR < {GEMNET_ETA_THRESHOLD} V (U_limiting > {U_0_ORR - GEMNET_ETA_THRESHOLD:.2f} V)")

    # Calculate ORR metrics using GemNet energies
    results = df.apply(
        lambda row: calculate_orr_metrics(row['gemnet_e_ads_oh'], row['gemnet_e_ads_o']),
        axis=1
    )

    df['gemnet_U_limiting'] = [r[0] for r in results]
    df['gemnet_eta_ORR'] = [r[1] for r in results]
    df['gemnet_limiting_step'] = [r[2] for r in results]

    # Apply overpotential threshold
    df_validated = df[df['gemnet_eta_ORR'] < GEMNET_ETA_THRESHOLD].copy()
    print(f"GemNet validated: {len(df_validated)}/{len(df)} samples ({len(df_validated)/len(df)*100:.1f}%)")

    # Statistics
    if len(df_validated) > 0:
        print(f"\nGemNet Validated Catalysts Statistics:")
        print(f"  G_OH range: [{df_validated['gemnet_e_ads_oh'].min():.2f}, {df_validated['gemnet_e_ads_oh'].max():.2f}] eV")
        print(f"  G_O range: [{df_validated['gemnet_e_ads_o'].min():.2f}, {df_validated['gemnet_e_ads_o'].max():.2f}] eV")
        print(f"  U_limiting range: [{df_validated['gemnet_U_limiting'].min():.2f}, {df_validated['gemnet_U_limiting'].max():.2f}] V")
        print(f"  η_ORR range: [{df_validated['gemnet_eta_ORR'].min():.2f}, {df_validated['gemnet_eta_ORR'].max():.2f}] V")
        print(f"  Best catalyst: η = {df_validated['gemnet_eta_ORR'].min():.3f} V (surface_id: {df_validated.loc[df_validated['gemnet_eta_ORR'].idxmin(), 'surface_id']})")

    return df_validated


def calculate_confusion_matrix(df, fork_threshold, gemnet_threshold):
    """
    Calculate confusion matrix for given thresholds.

    Ground Truth: GemNet predictions (η < gemnet_threshold is "active")
    Predictions: FORK predictions (η < fork_threshold is "active")

    Args:
        df: DataFrame with fork_eta_ORR and gemnet_eta_ORR columns
        fork_threshold: FORK overpotential threshold (V)
        gemnet_threshold: GemNet overpotential threshold (V)

    Returns:
        dict: {TP, TN, FP, FN, precision, recall, f1, accuracy}
    """
    # Predictions: FORK
    fork_active = df['fork_eta_ORR'] < fork_threshold

    # Ground Truth: GemNet
    gemnet_active = df['gemnet_eta_ORR'] < gemnet_threshold

    # Confusion matrix
    TP = np.sum(fork_active & gemnet_active)
    TN = np.sum(~fork_active & ~gemnet_active)
    FP = np.sum(fork_active & ~gemnet_active)
    FN = np.sum(~fork_active & gemnet_active)

    total = len(df)

    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total if total > 0 else 0

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'TP_pct': 100 * TP / total,
        'TN_pct': 100 * TN / total,
        'FP_pct': 100 * FP / total,
        'FN_pct': 100 * FN / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'total': total
    }


def plot_confusion_matrix(cm_dict, fork_threshold, gemnet_threshold, output_path):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm_dict: Confusion matrix dictionary from calculate_confusion_matrix
        fork_threshold: FORK threshold used
        gemnet_threshold: GemNet threshold used
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create confusion matrix array
    cm = np.array([[cm_dict['TN'], cm_dict['FP']],
                   [cm_dict['FN'], cm_dict['TP']]])

    # Plot heatmap
    im = ax.imshow(cm, cmap='Blues', alpha=0.8)

    # Labels with font properties
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Inactive', 'Active'])
    ax.set_yticklabels(['Inactive', 'Active'])
    ax.set_xlabel('GemNet Prediction (Ground Truth)', fontproperties=font_properties_label)
    ax.set_ylabel('FORK Prediction', fontproperties=font_properties_label)

    # Apply font styling to tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)

    # Add values and percentages
    for i in range(2):
        for j in range(2):
            labels = [['TN', 'FP'], ['FN', 'TP']]
            pcts = [['TN_pct', 'FP_pct'], ['FN_pct', 'TP_pct']]
            text = ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}\n({cm_dict[pcts[i][j]]:.1f}%)",
                          ha="center", va="center", color="black",
                          fontproperties=font_properties_label)

    # Title and metrics using subscript notation
    title = f'Confusion Matrix\nFORK $\\eta$ < {fork_threshold:.2f} V | GemNet $\\eta$ < {gemnet_threshold:.2f} V\n'
    title += f'Accuracy: {cm_dict["accuracy"]:.3f} | Precision: {cm_dict["precision"]:.3f} | '
    title += f'Recall: {cm_dict["recall"]:.3f} | F1: {cm_dict["f1"]:.3f}'
    ax.set_title(title, fontproperties=font_properties_label, pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', fontproperties=font_properties_label)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font_properties_tick)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix plot to {output_path}")


def plot_confusion_scatter_2d(df, fork_threshold, gemnet_threshold, output_path):
    """
    Plot 2D scatter visualization with all data points.
    X-axis: GemNet overpotential, Y-axis: FORK overpotential
    Threshold lines divide the plot into 4 quadrants (TP, TN, FP, FN)

    Args:
        df: DataFrame with fork_eta_ORR and gemnet_eta_ORR columns
        fork_threshold: FORK overpotential threshold (V)
        gemnet_threshold: GemNet overpotential threshold (V)
        output_path: Path to save the plot
    """
    # Calculate confusion matrix for statistics
    cm_dict = calculate_confusion_matrix(df, fork_threshold, gemnet_threshold)

    # Classify each point
    fork_active = df['fork_eta_ORR'] < fork_threshold
    gemnet_active = df['gemnet_eta_ORR'] < gemnet_threshold

    # Assign category to each point
    categories = np.where(fork_active & gemnet_active, 'TP',
                 np.where(~fork_active & ~gemnet_active, 'TN',
                 np.where(fork_active & ~gemnet_active, 'FP', 'FN')))

    # Color map for each category
    color_map = {
        'TN': colors[2],  # Dark blue
        'TP': colors[1],  # Orange
        'FP': colors[0],  # Dark red
        'FN': colors[3]   # Gray
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each category separately for proper legend
    for category in ['TN', 'FP', 'FN', 'TP']:
        mask = categories == category
        count = np.sum(mask)
        pct = 100 * count / len(df)

        ax.scatter(df.loc[mask, 'gemnet_eta_ORR'],
                  df.loc[mask, 'fork_eta_ORR'],
                  c=color_map[category],
                  s=100,
                  alpha=0.6,
                  edgecolors='black',
                  linewidth=0.5,
                  label=f'{category}: {count} ({pct:.1f}%)',
                  zorder=2)

    # Add threshold lines
    ax.axvline(gemnet_threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'GemNet threshold = {gemnet_threshold:.2f} V', zorder=3)
    ax.axhline(fork_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'FORK threshold = {fork_threshold:.2f} V', zorder=3)

    # Set axis labels
    ax.set_xlabel('GemNet Overpotential $\\eta$ (V)', fontproperties=font_properties_label)
    ax.set_ylabel('FORK Overpotential $\\eta$ (V)', fontproperties=font_properties_label)

    # Apply font styling
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)

    # Title with metrics
    title = f'Confusion Matrix 2D Scatter Plot\n'
    title += f'Accuracy: {cm_dict["accuracy"]:.3f} | Precision: {cm_dict["precision"]:.3f} | '
    title += f'Recall: {cm_dict["recall"]:.3f} | F1: {cm_dict["f1"]:.3f}'
    ax.set_title(title, fontproperties=font_properties_label, pad=20)

    # Legend
    ax.legend(prop=font_properties_tick, loc='best', framealpha=0.9)

    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate positions for quadrant labels (center of each quadrant)
    x_tn = (xlim[0] + gemnet_threshold) / 2
    x_fp = (gemnet_threshold + xlim[1]) / 2
    y_tn = (ylim[0] + fork_threshold) / 2
    y_fn = (fork_threshold + ylim[1]) / 2

    # Add semi-transparent quadrant labels
    label_props = dict(fontsize=14, fontweight='bold', alpha=0.3, ha='center', va='center')
    ax.text(x_tn, y_tn, 'TN', **label_props, zorder=1)
    ax.text(x_fp, y_tn, 'FP', **label_props, zorder=1)
    ax.text(x_tn, y_fn, 'FN', **label_props, zorder=1)
    ax.text(x_fp, y_fn, 'TP', **label_props, zorder=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion scatter 2D plot to {output_path}")


def print_parameters_at_eta(df, eta_values=[1.23, 0.53]):
    """
    Print all parameters for samples at specific overpotential values

    Args:
        df: DataFrame with fork_eta_ORR and gemnet_eta_ORR columns
        eta_values: List of overpotential values to extract (default: [1.23, 0.53])
    """
    print("\n" + "="*60)
    print("PARAMETERS AT SPECIFIC η VALUES")
    print("="*60)

    for eta in eta_values:
        print(f"\n{'─'*60}")
        print(f"Overpotential η = {eta:.2f} V")
        print(f"{'─'*60}")

        # Find samples close to this eta value (within ±0.01 V)
        tolerance = 0.01

        # FORK model
        fork_mask = np.abs(df['fork_eta_ORR'] - eta) < tolerance
        fork_samples = df[fork_mask]

        print(f"\n🔧 FORK Model (within ±{tolerance} V):")
        print(f"   Found {len(fork_samples)} samples")

        if len(fork_samples) > 0:
            for idx, row in fork_samples.head(5).iterrows():
                print(f"\n   Sample: surface_id = {row['surface_id']}")
                print(f"      G_OH = {row['crack_e_ads_oh']:.3f} eV")
                print(f"      G_O  = {row['crack_e_ads_o']:.3f} eV")
                print(f"      U_limiting = {row['fork_U_limiting']:.3f} V")
                print(f"      η_ORR = {row['fork_eta_ORR']:.3f} V")
                print(f"      Limiting step = {int(row['fork_limiting_step'])}")

        # GemNet model
        gemnet_mask = np.abs(df['gemnet_eta_ORR'] - eta) < tolerance
        gemnet_samples = df[gemnet_mask]

        print(f"\n💎 GemNet Model (within ±{tolerance} V):")
        print(f"   Found {len(gemnet_samples)} samples")

        if len(gemnet_samples) > 0:
            for idx, row in gemnet_samples.head(5).iterrows():
                print(f"\n   Sample: surface_id = {row['surface_id']}")
                print(f"      G_OH = {row['gemnet_e_ads_oh']:.3f} eV")
                print(f"      G_O  = {row['gemnet_e_ads_o']:.3f} eV")
                print(f"      U_limiting = {row['gemnet_U_limiting']:.3f} V")
                print(f"      η_ORR = {row['gemnet_eta_ORR']:.3f} V")
                print(f"      Limiting step = {int(row['gemnet_limiting_step'])}")

        # If no exact matches, show closest samples
        if len(fork_samples) == 0 and len(gemnet_samples) == 0:
            print(f"\n   ℹ️  No samples within ±{tolerance} V. Showing closest samples:")

            # Find closest FORK sample
            fork_closest_idx = (df['fork_eta_ORR'] - eta).abs().idxmin()
            fork_closest = df.loc[fork_closest_idx]
            print(f"\n   🔧 Closest FORK: η = {fork_closest['fork_eta_ORR']:.3f} V")
            print(f"      surface_id = {fork_closest['surface_id']}")
            print(f"      G_OH = {fork_closest['crack_e_ads_oh']:.3f} eV, G_O = {fork_closest['crack_e_ads_o']:.3f} eV")

            # Find closest GemNet sample
            gemnet_closest_idx = (df['gemnet_eta_ORR'] - eta).abs().idxmin()
            gemnet_closest = df.loc[gemnet_closest_idx]
            print(f"\n   💎 Closest GemNet: η = {gemnet_closest['gemnet_eta_ORR']:.3f} V")
            print(f"      surface_id = {gemnet_closest['surface_id']}")
            print(f"      G_OH = {gemnet_closest['gemnet_e_ads_oh']:.3f} eV, G_O = {gemnet_closest['gemnet_e_ads_o']:.3f} eV")


def threshold_sweep_analysis(df, fork_thresholds, gemnet_threshold):
    """
    Perform threshold sweep analysis varying FORK threshold.

    Args:
        df: DataFrame with fork_eta_ORR and gemnet_eta_ORR columns
        fork_thresholds: Array of FORK thresholds to test
        gemnet_threshold: Fixed GemNet threshold

    Returns:
        DataFrame with sweep results
    """
    results = []

    for fork_th in fork_thresholds:
        cm = calculate_confusion_matrix(df, fork_th, gemnet_threshold)
        results.append({
            'fork_threshold': fork_th,
            'TP': cm['TP'], 'TN': cm['TN'], 'FP': cm['FP'], 'FN': cm['FN'],
            'TP_pct': cm['TP_pct'], 'TN_pct': cm['TN_pct'],
            'FP_pct': cm['FP_pct'], 'FN_pct': cm['FN_pct'],
            'precision': cm['precision'], 'recall': cm['recall'],
            'f1': cm['f1'], 'accuracy': cm['accuracy']
        })

    return pd.DataFrame(results)


def plot_threshold_sweep(sweep_df, gemnet_threshold, output_path):
    """
    Plot threshold sweep results showing how confusion matrix changes with FORK threshold.

    Args:
        sweep_df: DataFrame from threshold_sweep_analysis
        gemnet_threshold: GemNet threshold used
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Color assignments for consistency
    cm_colors = {
        'TP': colors[1],  # Orange
        'TN': colors[2],  # Dark blue
        'FP': colors[0],  # Dark red
        'FN': colors[3]   # Gray
    }

    # Plot 1: Confusion matrix components (counts)
    ax = axes[0, 0]
    ax.plot(sweep_df['fork_threshold'], sweep_df['TP'], 'o-', label='TP',
            color=cm_colors['TP'], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['TN'], 's-', label='TN',
            color=cm_colors['TN'], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['FP'], '^-', label='FP',
            color=cm_colors['FP'], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['FN'], 'v-', label='FN',
            color=cm_colors['FN'], linewidth=1.5, markersize=6, zorder=2)
    ax.set_xlabel('FORK Threshold $\\eta$ (V)', fontproperties=font_properties_label)
    ax.set_ylabel('Count', fontproperties=font_properties_label)
    ax.set_title(f'Confusion Matrix Components vs FORK Threshold\n(GemNet threshold = {gemnet_threshold:.2f} V)',
                 fontproperties=font_properties_label)
    ax.legend(prop=font_properties_tick, loc='best')
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)
    # Plot 2: Confusion matrix components (percentages)
    ax = axes[0, 1]
    ax.plot(sweep_df['fork_threshold'], sweep_df['TP_pct'], 'o-', label='TP %',
            color=cm_colors['TP'], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['TN_pct'], 's-', label='TN %',
            color=cm_colors['TN'], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['FP_pct'], '^-', label='FP %',
            color=cm_colors['FP'], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['FN_pct'], 'v-', label='FN %',
            color=cm_colors['FN'], linewidth=1.5, markersize=6, zorder=2)
    ax.set_xlabel('FORK Threshold $\\eta$ (V)', fontproperties=font_properties_label)
    ax.set_ylabel('Percentage (%)', fontproperties=font_properties_label)
    ax.set_title(f'Confusion Matrix Percentages vs FORK Threshold\n(GemNet threshold = {gemnet_threshold:.2f} V)',
                 fontproperties=font_properties_label)
    ax.legend(prop=font_properties_tick, loc='best')
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)
    ax.set_ylim([0, 100])

    # Plot 3: Performance metrics
    ax = axes[1, 0]
    metric_colors = [colors[0], colors[1], colors[2], colors[3]]
    ax.plot(sweep_df['fork_threshold'], sweep_df['precision'], 'o-', label='Precision',
            color=metric_colors[0], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['recall'], 's-', label='Recall',
            color=metric_colors[1], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['f1'], '^-', label='F1 Score',
            color=metric_colors[2], linewidth=1.5, markersize=6, zorder=2)
    ax.plot(sweep_df['fork_threshold'], sweep_df['accuracy'], 'v-', label='Accuracy',
            color=metric_colors[3], linewidth=1.5, markersize=6, zorder=2)
    ax.set_xlabel('FORK Threshold $\\eta$ (V)', fontproperties=font_properties_label)
    ax.set_ylabel('Score', fontproperties=font_properties_label)
    ax.set_title(f'Performance Metrics vs FORK Threshold\n(GemNet threshold = {gemnet_threshold:.2f} V)',
                 fontproperties=font_properties_label)
    ax.legend(ncol=2, prop=font_properties_tick, loc='best')
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)
    ax.set_ylim([0, 1.0])
    ax.set_xlim([0.5, 1.5])

    # Plot 4: Stacked area chart of percentages
    ax = axes[1, 1]
    ax.fill_between(sweep_df['fork_threshold'], 0, sweep_df['TN_pct'],
                     label='TN', color=cm_colors['TN'], alpha=0.7)
    ax.fill_between(sweep_df['fork_threshold'], sweep_df['TN_pct'],
                     sweep_df['TN_pct'] + sweep_df['FP_pct'],
                     label='FP', color=cm_colors['FP'], alpha=0.7)
    ax.fill_between(sweep_df['fork_threshold'], sweep_df['TN_pct'] + sweep_df['FP_pct'],
                     sweep_df['TN_pct'] + sweep_df['FP_pct'] + sweep_df['FN_pct'],
                     label='FN', color=cm_colors['FN'], alpha=0.7)
    ax.fill_between(sweep_df['fork_threshold'],
                     sweep_df['TN_pct'] + sweep_df['FP_pct'] + sweep_df['FN_pct'],
                     100, label='TP', color=cm_colors['TP'], alpha=0.7)
    ax.set_xlabel('FORK Threshold $\\eta$ (V)', fontproperties=font_properties_label)
    ax.set_ylabel('Percentage (%)', fontproperties=font_properties_label)
    ax.set_title(f'Stacked Confusion Matrix Distribution\n(GemNet threshold = {gemnet_threshold:.2f} V)',
                 fontproperties=font_properties_label)
    ax.legend(prop=font_properties_tick, loc='best')
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font_properties_tick)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font_properties_tick)
    ax.set_ylim([0, 100])
    ax.set_xlim([0.5, 1.5])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved threshold sweep plot to {output_path}")


def main():
    """Main screening pipeline"""
    print("\n" + "="*60)
    print("ORR CATALYST SCREENING - OVERPOTENTIAL-BASED")
    print("="*60)

    # Load and merge data
    o_df, oh_df = load_data()
    merged_df = merge_data(o_df, oh_df)

    # FORK screening (returns full dataset with metrics + active subset)
    fork_full, fork_active = screen_fork_overpotential(merged_df)

    # Calculate GemNet metrics for all samples (needed for confusion matrix)
    print("\n" + "="*60)
    print("CALCULATING GEMNET METRICS FOR ALL SAMPLES")
    print("="*60)
    results = fork_full.apply(
        lambda row: calculate_orr_metrics(row['gemnet_e_ads_oh'], row['gemnet_e_ads_o']),
        axis=1
    )
    fork_full['gemnet_U_limiting'] = [r[0] for r in results]
    fork_full['gemnet_eta_ORR'] = [r[1] for r in results]
    fork_full['gemnet_limiting_step'] = [r[2] for r in results]
    print(f"GemNet metrics calculated for {len(fork_full)} samples")

    # Save FORK results
    fork_full_output = OUTPUT_DIR / "FORK_all_with_eta.csv"
    fork_active_output = OUTPUT_DIR / "FORK_ORR_active_eta.csv"
    fork_full.to_csv(fork_full_output, index=False)
    fork_active.to_csv(fork_active_output, index=False)
    print(f"\n✅ FORK full dataset saved to {fork_full_output}")
    print(f"✅ FORK active saved to {fork_active_output}")

    # GemNet validation
    if len(fork_active) > 0:
        gemnet_validated = screen_gemnet_overpotential(fork_active, merged_df)

        # Save GemNet results
        gemnet_output = OUTPUT_DIR / "GemNet_ORR_validated_eta.csv"
        gemnet_validated.to_csv(gemnet_output, index=False)
        print(f"✅ GemNet results saved to {gemnet_output}")
    else:
        print("\n⚠️  No FORK active catalysts found, skipping GemNet validation")
        gemnet_validated = pd.DataFrame()

    # Confusion Matrix Analysis
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)

    # Analyze thresholds: 0.73 V and 1.23 V
    gemnet_thresholds = [0.73, 1.23]

    for gemnet_th in gemnet_thresholds:
        print(f"\n--- GemNet Threshold = {gemnet_th:.2f} V ---")

        # Calculate confusion matrix for FORK threshold = 0.73 V
        cm_073 = calculate_confusion_matrix(fork_full, 0.73, gemnet_th)
        print(f"\nFORK threshold = 0.73 V:")
        print(f"  TP: {cm_073['TP']} ({cm_073['TP_pct']:.1f}%)")
        print(f"  TN: {cm_073['TN']} ({cm_073['TN_pct']:.1f}%)")
        print(f"  FP: {cm_073['FP']} ({cm_073['FP_pct']:.1f}%)")
        print(f"  FN: {cm_073['FN']} ({cm_073['FN_pct']:.1f}%)")
        print(f"  Accuracy: {cm_073['accuracy']:.3f} | Precision: {cm_073['precision']:.3f} | Recall: {cm_073['recall']:.3f} | F1: {cm_073['f1']:.3f}")

        # Plot confusion matrix for 0.73 V
        cm_plot_path = CONFUSION_OUTPUT_DIR / f"confusion_matrix_fork073_gemnet{gemnet_th:.2f}.png"
        plot_confusion_matrix(cm_073, 0.73, gemnet_th, cm_plot_path)

        # Plot 2D scatter for 0.73 V
        scatter_plot_path = CONFUSION_OUTPUT_DIR / f"confusion_scatter2d_fork073_gemnet{gemnet_th:.2f}.png"
        plot_confusion_scatter_2d(fork_full, 0.73, gemnet_th, scatter_plot_path)

        # Calculate confusion matrix for FORK threshold = 1.23 V
        cm_123 = calculate_confusion_matrix(fork_full, 1.23, gemnet_th)
        print(f"\nFORK threshold = 1.23 V:")
        print(f"  TP: {cm_123['TP']} ({cm_123['TP_pct']:.1f}%)")
        print(f"  TN: {cm_123['TN']} ({cm_123['TN_pct']:.1f}%)")
        print(f"  FP: {cm_123['FP']} ({cm_123['FP_pct']:.1f}%)")
        print(f"  FN: {cm_123['FN']} ({cm_123['FN_pct']:.1f}%)")
        print(f"  Accuracy: {cm_123['accuracy']:.3f} | Precision: {cm_123['precision']:.3f} | Recall: {cm_123['recall']:.3f} | F1: {cm_123['f1']:.3f}")

        # Plot confusion matrix for 1.23 V
        cm_plot_path = CONFUSION_OUTPUT_DIR / f"confusion_matrix_fork123_gemnet{gemnet_th:.2f}.png"
        plot_confusion_matrix(cm_123, 1.23, gemnet_th, cm_plot_path)

        # Plot 2D scatter for 1.23 V
        scatter_plot_path = CONFUSION_OUTPUT_DIR / f"confusion_scatter2d_fork123_gemnet{gemnet_th:.2f}.png"
        plot_confusion_scatter_2d(fork_full, 1.23, gemnet_th, scatter_plot_path)

    # Print parameters at specific η values
    print_parameters_at_eta(fork_full, eta_values=[1.23, 0.53])

    # Threshold Sweep Analysis
    print("\n" + "="*60)
    print("THRESHOLD SWEEP ANALYSIS")
    print("="*60)

    # Define FORK threshold range for sweep
    fork_thresholds = np.linspace(0.3, 1.5, 50)
    # Add specific thresholds of interest
    fork_thresholds = np.sort(np.append(fork_thresholds, [1.23, 0.53]))

    for gemnet_th in gemnet_thresholds:
        print(f"\nPerforming sweep for GemNet threshold = {gemnet_th:.2f} V...")

        # Perform sweep
        sweep_df = threshold_sweep_analysis(fork_full, fork_thresholds, gemnet_th)

        # Save sweep results
        sweep_csv_path = CONFUSION_OUTPUT_DIR / f"threshold_sweep_gemnet{gemnet_th:.2f}.csv"
        sweep_df.to_csv(sweep_csv_path, index=False)
        print(f"  Saved sweep data to {sweep_csv_path}")

        # Plot sweep results
        sweep_plot_path = CONFUSION_OUTPUT_DIR / f"threshold_sweep_gemnet{gemnet_th:.2f}.png"
        plot_threshold_sweep(sweep_df, gemnet_th, sweep_plot_path)

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(merged_df)}")
    print(f"FORK active (η < {FORK_ETA_THRESHOLD} V): {len(fork_active)} ({len(fork_active)/len(merged_df)*100:.2f}%)")
    print(f"GemNet validated (η < {GEMNET_ETA_THRESHOLD} V): {len(gemnet_validated)} ({len(gemnet_validated)/len(merged_df)*100:.2f}%)")

    if len(gemnet_validated) > 0:
        print(f"\n🏆 Top 5 Catalysts (by GemNet η_ORR):")
        top5 = gemnet_validated.nsmallest(5, 'gemnet_eta_ORR')[['surface_id', 'gemnet_eta_ORR', 'gemnet_U_limiting', 'gemnet_e_ads_oh', 'gemnet_e_ads_o']]
        print(top5.to_string(index=False))

    print(f"\n📊 Confusion matrix analysis complete!")
    print(f"   Results saved to: {CONFUSION_OUTPUT_DIR}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
