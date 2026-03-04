"""
Adsorption energy calculator module using OC20 predefined gas energies.

This module uses predefined atomic energies from OC20 dataset instead of
calculated gas molecule energies:
- H: -3.477 eV
- O: -7.204 eV
- C: -7.282 eV
- N: -8.083 eV
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from data_loader import DataLoader


class AdsorptionCalculatorOC20:
    """Calculate adsorption energies using OC20 predefined gas energies."""

    # Optimal targets from literature
    DELTA_E_CO_OPTIMAL = -0.67  # eV
    DELTA_E_H_OPTIMAL = 0.27    # eV
    DISCOVERY_RANGE = 0.2       # ±0.2 eV

    # OC20 adsorbate atom energies in electron volts (eV)
    # All values given to 4 significant figures
    ADSORBATE_ENERGIES = {
        'H': -3.477,
        'O': -7.204,
        'C': -7.282,
        'N': -8.083
    }

    def __init__(self, loader: DataLoader):
        """
        Initialize adsorption calculator with OC20 energies.

        Args:
            loader: DataLoader instance with loaded data
        """
        self.loader = loader
        self.co_results = []
        self.h_results = []

    def calculate_adsorption_energy(
        self,
        surface_id: str,
        adslab_energy: float,
        adsorbate: str,
        calculator: str
    ) -> float:
        """
        Calculate adsorption energy using OC20 predefined atomic energies.

        ΔE_ads = E_adslab - E_slab - E_gas

        For OC20 reference:
        - H adsorption: E_gas = E_H = -3.477 eV
        - CO adsorption: E_gas = E_C + E_O = -7.282 + (-7.204) = -14.486 eV

        Args:
            surface_id: Surface identifier
            adslab_energy: Energy of adsorbate+slab system
            adsorbate: Adsorbate type ('CO' or 'H')
            calculator: Calculator name ('crack' or 'gemnet')

        Returns:
            Adsorption energy in eV
        """
        # Get slab energy
        slab_energy, _ = self.loader.get_slab_energy(surface_id, calculator)

        # Get gas energy from OC20 predefined atomic energies
        if adsorbate == 'H':
            # H adsorption: use atomic H energy
            gas_energy = self.ADSORBATE_ENERGIES['H']
        elif adsorbate == 'CO':
            # CO adsorption: use C + O atomic energies
            gas_energy = self.ADSORBATE_ENERGIES['C'] + self.ADSORBATE_ENERGIES['O']
        else:
            raise ValueError(f"Unknown adsorbate: {adsorbate}")

        # Calculate adsorption energy
        delta_e = adslab_energy - slab_energy - gas_energy

        return delta_e

    def process_all_adsorption_data(self):
        """
        Process all adsorption data and calculate energies.

        Creates two lists:
        - self.co_results: CO adsorption results
        - self.h_results: H adsorption results
        """
        print("=" * 60)
        print("Calculating adsorption energies (OC20 gas energies)...")
        print("Using predefined atomic energies:")
        print(f"  H: {self.ADSORBATE_ENERGIES['H']} eV")
        print(f"  C: {self.ADSORBATE_ENERGIES['C']} eV")
        print(f"  O: {self.ADSORBATE_ENERGIES['O']} eV")
        print(f"  CO: {self.ADSORBATE_ENERGIES['C'] + self.ADSORBATE_ENERGIES['O']:.3f} eV")
        print("=" * 60)

        # Process CO adsorption
        self.co_results = self._process_adsorbate('CO', self.loader.co_adslab_data)
        print(f"Processed {len(self.co_results)} CO configurations")

        # Process H adsorption
        self.h_results = self._process_adsorbate('H', self.loader.h_adslab_data)
        print(f"Processed {len(self.h_results)} H configurations")

        print("=" * 60)
        print("Adsorption energy calculation complete!")
        print("=" * 60)

    def _process_adsorbate(self, adsorbate: str, adslab_data: Dict) -> List[Dict]:
        """
        Process adsorbate data and calculate adsorption energies.

        Args:
            adsorbate: 'CO' or 'H'
            adslab_data: Adslab data dictionary

        Returns:
            List of dictionaries with results
        """
        results = []

        for surface_id, surface_data in adslab_data.items():
            bulk_id = surface_data['bulk_id']
            miller = tuple(surface_data['miller'])
            total_configs = surface_data['total_configs']

            # Process each configuration
            for config in surface_data['configurations']:
                config_idx = config['config_idx']
                num_atoms = config['num_atoms']

                # Process crack results
                crack_data = config.get('crack', {})
                if crack_data.get('converged', False):
                    crack_e_ads = self.calculate_adsorption_energy(
                        surface_id,
                        crack_data['energy'],
                        adsorbate,
                        'crack'
                    )
                    crack_time = crack_data['time']
                    crack_steps = crack_data['steps']
                else:
                    crack_e_ads = None
                    crack_time = None
                    crack_steps = None

                # Process gemnet results
                gemnet_data = config.get('gemnet', {})
                if gemnet_data.get('converged', False):
                    gemnet_e_ads = self.calculate_adsorption_energy(
                        surface_id,
                        gemnet_data['energy'],
                        adsorbate,
                        'gemnet'
                    )
                    gemnet_time = gemnet_data['time']
                    gemnet_steps = gemnet_data['steps']
                else:
                    gemnet_e_ads = None
                    gemnet_time = None
                    gemnet_steps = None

                # Store results
                results.append({
                    'surface_id': surface_id,
                    'bulk_id': bulk_id,
                    'miller': miller,
                    'config_idx': config_idx,
                    'num_atoms': num_atoms,
                    'adsorbate': adsorbate,
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

    def get_dataframe(self, adsorbate: str = None) -> pd.DataFrame:
        """
        Get results as pandas DataFrame.

        Args:
            adsorbate: Filter by adsorbate ('CO', 'H', or None for all)

        Returns:
            DataFrame with adsorption results
        """
        if adsorbate == 'CO':
            return pd.DataFrame(self.co_results)
        elif adsorbate == 'H':
            return pd.DataFrame(self.h_results)
        elif adsorbate is None:
            return pd.DataFrame(self.co_results + self.h_results)
        else:
            raise ValueError(f"Unknown adsorbate: {adsorbate}")

    def identify_active_catalysts(
        self,
        calculator: str,
        tolerance: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Identify active catalysts within optimal range.

        Args:
            calculator: 'crack' or 'gemnet'
            tolerance: Discovery range (default: ±0.2 eV)

        Returns:
            Tuple of (co_active, h_active) DataFrames
        """
        if tolerance is None:
            tolerance = self.DISCOVERY_RANGE

        co_df = self.get_dataframe('CO')
        h_df = self.get_dataframe('H')

        # Filter converged results
        co_col = f'{calculator}_e_ads'
        h_col = f'{calculator}_e_ads'

        # CO active: within ±tolerance of optimal
        co_active = co_df[
            (co_df[f'{calculator}_converged']) &
            (co_df[co_col].notna()) &
            (co_df[co_col] >= self.DELTA_E_CO_OPTIMAL - tolerance) &
            (co_df[co_col] <= self.DELTA_E_CO_OPTIMAL + tolerance)
        ]

        # H active: within ±tolerance of optimal
        h_active = h_df[
            (h_df[f'{calculator}_converged']) &
            (h_df[h_col].notna()) &
            (h_df[h_col] >= self.DELTA_E_H_OPTIMAL - tolerance) &
            (h_df[h_col] <= self.DELTA_E_H_OPTIMAL + tolerance)
        ]

        return co_active, h_active

    def get_discovery_timeline(self, calculator: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get timeline of discoveries sorted by cumulative time.

        Args:
            calculator: 'crack' or 'gemnet'

        Returns:
            Tuple of (co_timeline, h_timeline) DataFrames with cumulative time
        """
        co_active, h_active = self.identify_active_catalysts(calculator)

        # Sort by time and add cumulative time
        time_col = f'{calculator}_time'

        if len(co_active) > 0:
            co_timeline = co_active.sort_values(time_col).copy()
            co_timeline['cumulative_time'] = co_timeline[time_col].cumsum()
            co_timeline['discovery_number'] = range(1, len(co_timeline) + 1)
        else:
            co_timeline = pd.DataFrame()

        if len(h_active) > 0:
            h_timeline = h_active.sort_values(time_col).copy()
            h_timeline['cumulative_time'] = h_timeline[time_col].cumsum()
            h_timeline['discovery_number'] = range(1, len(h_timeline) + 1)
        else:
            h_timeline = pd.DataFrame()

        return co_timeline, h_timeline

    def calculate_statistics(self) -> Dict:
        """
        Calculate statistical metrics for crack vs gemnet comparison.

        Returns:
            Dictionary of statistics
        """
        co_df = self.get_dataframe('CO')
        h_df = self.get_dataframe('H')

        # Filter for pairs where both converged
        co_both = co_df[co_df['crack_converged'] & co_df['gemnet_converged']].copy()
        h_both = h_df[h_df['crack_converged'] & h_df['gemnet_converged']].copy()

        # Calculate errors
        co_both['error'] = co_both['crack_e_ads'] - co_both['gemnet_e_ads']
        h_both['error'] = h_both['crack_e_ads'] - h_both['gemnet_e_ads']

        stats = {
            'CO': {
                'total_configs': len(co_df),
                'crack_converged': co_df['crack_converged'].sum(),
                'gemnet_converged': co_df['gemnet_converged'].sum(),
                'both_converged': len(co_both),
                'mae': np.abs(co_both['error']).mean() if len(co_both) > 0 else np.nan,
                'rmse': np.sqrt((co_both['error'] ** 2).mean()) if len(co_both) > 0 else np.nan,
                'mean_error': co_both['error'].mean() if len(co_both) > 0 else np.nan,
                'std_error': co_both['error'].std() if len(co_both) > 0 else np.nan,
                'corr': co_both[['crack_e_ads', 'gemnet_e_ads']].corr().iloc[0, 1] if len(co_both) > 0 else np.nan,
                'crack_mean_time': co_df[co_df['crack_converged']]['crack_time'].mean(),
                'gemnet_mean_time': co_df[co_df['gemnet_converged']]['gemnet_time'].mean(),
                'crack_mean_steps': co_df[co_df['crack_converged']]['crack_steps'].mean(),
                'gemnet_mean_steps': co_df[co_df['gemnet_converged']]['gemnet_steps'].mean(),
            },
            'H': {
                'total_configs': len(h_df),
                'crack_converged': h_df['crack_converged'].sum(),
                'gemnet_converged': h_df['gemnet_converged'].sum(),
                'both_converged': len(h_both),
                'mae': np.abs(h_both['error']).mean() if len(h_both) > 0 else np.nan,
                'rmse': np.sqrt((h_both['error'] ** 2).mean()) if len(h_both) > 0 else np.nan,
                'mean_error': h_both['error'].mean() if len(h_both) > 0 else np.nan,
                'std_error': h_both['error'].std() if len(h_both) > 0 else np.nan,
                'corr': h_both[['crack_e_ads', 'gemnet_e_ads']].corr().iloc[0, 1] if len(h_both) > 0 else np.nan,
                'crack_mean_time': h_df[h_df['crack_converged']]['crack_time'].mean(),
                'gemnet_mean_time': h_df[h_df['gemnet_converged']]['gemnet_time'].mean(),
                'crack_mean_steps': h_df[h_df['crack_converged']]['crack_steps'].mean(),
                'gemnet_mean_steps': h_df[h_df['gemnet_converged']]['gemnet_steps'].mean(),
            }
        }

        return stats

    def print_statistics(self):
        """Print statistical summary."""
        stats = self.calculate_statistics()

        print("\n" + "=" * 60)
        print("STATISTICAL SUMMARY (OC20 Gas Energies)")
        print("=" * 60)

        for adsorbate in ['CO', 'H']:
            print(f"\n{adsorbate} Adsorption:")
            print("-" * 60)
            s = stats[adsorbate]
            print(f"  Total configurations: {s['total_configs']}")
            print(f"  Crack converged: {s['crack_converged']} ({s['crack_converged']/s['total_configs']*100:.1f}%)")
            print(f"  GemNet converged: {s['gemnet_converged']} ({s['gemnet_converged']/s['total_configs']*100:.1f}%)")
            print(f"  Both converged: {s['both_converged']}")
            print(f"\n  Energy Comparison (eV):")
            print(f"    MAE: {s['mae']:.4f}")
            print(f"    RMSE: {s['rmse']:.4f}")
            print(f"    Mean Error: {s['mean_error']:.4f}")
            print(f"    Std Error: {s['std_error']:.4f}")
            print(f"    Correlation: {s['corr']:.4f}")
            print(f"\n  Computational Time (s):")
            print(f"    Crack mean: {s['crack_mean_time']:.4f}")
            print(f"    GemNet mean: {s['gemnet_mean_time']:.4f}")
            print(f"    Speedup: {s['crack_mean_time']/s['gemnet_mean_time']:.2f}x")
            print(f"\n  Convergence Steps:")
            print(f"    Crack mean: {s['crack_mean_steps']:.1f}")
            print(f"    GemNet mean: {s['gemnet_mean_steps']:.1f}")

        # Discovery statistics
        print(f"\n" + "=" * 60)
        print("CATALYST DISCOVERY STATISTICS")
        print("=" * 60)

        for calc in ['crack', 'gemnet']:
            co_active, h_active = self.identify_active_catalysts(calc)
            print(f"\n{calc.upper()}:")
            print(f"  CO active catalysts: {len(co_active)}")
            print(f"  H active catalysts: {len(h_active)}")

        print("=" * 60)


if __name__ == "__main__":
    # Test the calculator
    from data_loader import DataLoader

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_path', required=True, help='Path to ML relaxation results')
    args = parser.parse_args()

    print("Loading data...")
    loader = DataLoader(args.sim_path)
    loader.load_all_data()

    print("\nCalculating adsorption energies (OC20 method)...")
    calculator = AdsorptionCalculatorOC20(loader)
    calculator.process_all_adsorption_data()

    print("\nPrinting statistics...")
    calculator.print_statistics()

    print("\nSaving DataFrames...")
    co_df = calculator.get_dataframe('CO')
    h_df = calculator.get_dataframe('H')

    print(f"\nCO DataFrame shape: {co_df.shape}")
    print(f"H DataFrame shape: {h_df.shape}")

    print("\nFirst few CO results:")
    print(co_df.head())
