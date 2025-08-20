"""
Mass-Weighted OU Hypothesis Test FX Pairs
-------------------------------------------

Hypothesis:
The pair with lower 'mass' (liquidity proxy) corrects more often toward the
movement of the pair with higher 'mass'.

- H0: no relation (p >= 0.5)
- H1: relation exists (p < 0.5)

Outputs:
Console summary per pair and overall binomial tests (overall & event-based).

Author: Tim (2025)
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import binom
import warnings
warnings.filterwarnings('ignore')

# Local data directories per timeframe
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
path_dir = {
    '5min': os.path.join(parent_dir, 'data_5min'),
    '15min': os.path.join(parent_dir, 'data_15min'),
    '30min': os.path.join(parent_dir, 'data_30min'),
    'uur': os.path.join(parent_dir, 'data_uur'),
    'dag': os.path.join(parent_dir, 'data_dag')
}

def data_inladen(tijdframe, ticker):

    """Load parquet for given ticker/timeframe (expects 'close' and 'volume')."""

    if tijdframe not in path_dir:
        print(f'Invalid timeframe: {tijdframe}')
        return False
    tf_pad = path_dir[tijdframe]
    ticker_pad = os.path.join(tf_pad, f'{ticker}.parquet')
    if os.path.exists(ticker_pad):
        df = pd.read_parquet(ticker_pad)
        if df.empty:
            print(f'Empty dataframe for: {ticker}')
        return df
    else:
        print(f'No data available for {ticker}')
        return pd.DataFrame()

def mass(ticker_a, ticker_b, timeframe):

    """Merge A/B data and compute liquidity proxies (volume * close price)."""

    df_a = data_inladen(timeframe, ticker_a)
    df_b = data_inladen(timeframe, ticker_b)
    merged = pd.merge(df_a[['close', 'volume']], df_b[['close', 'volume']], 
                     left_index=True, right_index=True, suffixes=('_a', '_b'))
    merged['liquiditeit_a'] = merged['volume_a'] * merged['close_a']
    merged['liquiditeit_b'] = merged['volume_b'] * merged['close_b']
    merged = merged.dropna()
    return merged

def ou(ticker_a, ticker_b, timeframe, window):

    """Compute rolling OU-style parameters A→B and B→A with mass scaling."""

    df = mass(ticker_a, ticker_b, timeframe)

    # only use this d_t if timeframe is 5 min.
    d_t = 1/288

    # van ticker a naar ticker b
    X_t_a = df['close_a']
    Y_t_b = df['close_b']
    X_t_b = df['close_b']
    Y_t_a = df['close_a']
    mass_a = df['liquiditeit_a']
    mass_b = df['liquiditeit_b']

    roll = pd.DataFrame({
        'mean_X_t_a': X_t_a.rolling(window).mean(),
        'mean_X_t_b': X_t_b.rolling(window).mean(),
        'mean_Y_t_a': Y_t_a.rolling(window).mean(),
        'mean_Y_t_b': Y_t_b.rolling(window).mean(),
        'cov_XY_a': X_t_a.rolling(window).cov(Y_t_b),
        'cov_YX_b': Y_t_b.rolling(window).cov(X_t_a),
        'var_X_t_a': X_t_a.rolling(window).var(),
        'var_X_t_b': X_t_b.rolling(window).var(),
        'var_Y_t_b': Y_t_b.rolling(window).var(),
        'var_Y_t_a': Y_t_a.rolling(window).var(),
        'liquiditeit_a': mass_a,
        'liquiditeit_b': mass_b
    })
    
    np.random.seed(42) 
    epsilon_a = np.random.normal(0, 1, len(roll))
    epsilon_b = np.random.normal(0, 1, len(roll))
    
    den_a = roll['var_X_t_a'].replace(0, np.nan)
    den_b = roll['var_X_t_b'].replace(0, np.nan)

    # OU parameters voor A -> B (A reverts naar B)
    roll['beta_a_to_b'] = roll['cov_XY_a'] / den_a
    roll['beta_a_to_b'] = roll['beta_a_to_b'].clip(lower=1e-6, upper=0.999999)
    roll['alpha_a_to_b'] = roll['mean_Y_t_b'] - roll['beta_a_to_b'] * roll['mean_X_t_a']
    roll['mu_a_to_b'] = roll['alpha_a_to_b'] / (1 - roll['beta_a_to_b'])
    roll['theta_a_to_b'] = -np.log(roll['beta_a_to_b'])/d_t
    roll['sigma_a_to_b'] = X_t_a.rolling(window).std()
    roll['ou_a_to_b'] = X_t_a + ((roll['theta_a_to_b'] * (Y_t_b - X_t_a) * d_t) / mass_a) + ((roll['sigma_a_to_b'] * np.sqrt(d_t) * epsilon_a) / mass_a)
    
    # OU parameters voor B -> A (B reverts naar A)
    roll['beta_b_to_a'] = roll['cov_YX_b'] / den_b
    roll['beta_b_to_a'] = roll['beta_b_to_a'].clip(lower=1e-6, upper=0.999999)
    roll['alpha_b_to_a'] = roll['mean_Y_t_a'] - roll['beta_b_to_a'] * roll['mean_X_t_b']
    roll['mu_b_to_a'] = roll['alpha_b_to_a'] / (1 - roll['beta_b_to_a'])
    roll['theta_b_to_a'] = -np.log(roll['beta_b_to_a'])/d_t
    roll['sigma_b_to_a'] = X_t_b.rolling(window).std()
    roll['ou_b_to_a'] = X_t_b + ((roll['theta_b_to_a'] * (Y_t_a - X_t_b) * d_t) / mass_b) + ((roll['sigma_b_to_a'] * np.sqrt(d_t) * epsilon_b) / mass_b)
    
    return roll

def calculate_spread_and_zscore(ticker_a, ticker_b, timeframe, window):
    df = mass(ticker_a, ticker_b, timeframe)

    # log-price spread (scale-free)
    df['spread'] = np.log(df['close_a']) - np.log(df['close_b'])

    df['spread_mean'] = df['spread'].rolling(window).mean()
    df['spread_std']  = df['spread'].rolling(window).std()

    std = df['spread_std'].replace(0, np.nan)
    df['z_score'] = (df['spread'] - df['spread_mean']) / std
    return df


def threshold_based_hypothesis_test(ticker_a, ticker_b, timeframe='5min', window=20, threshold=2.0, lag=1):
    """
        Thresholded hypothesis test with lag:
        - Select only times where |z_score| exceeds threshold.
        - Compare theta dynamics lagged by 'lag' periods, mass-weighted.
    """
    ou_data = ou(ticker_a, ticker_b, timeframe, window)
    spread_data = calculate_spread_and_zscore(ticker_a, ticker_b, timeframe, window)
    
    # Merge de datasets
    combined_data = pd.merge(ou_data, spread_data[['z_score', 'spread']], 
                           left_index=True, right_index=True, how='inner')
    
    combined_data = combined_data.dropna()
    
    if len(combined_data) < 30:
        return None
    
    threshold_mask = abs(combined_data['z_score']) > threshold 
    
    if threshold_mask.sum() < 10:
        return None
    
    lagged_data = combined_data.copy()
    lagged_data['theta_a_to_b_lagged'] = lagged_data['theta_a_to_b'].shift(-lag)
    lagged_data['theta_b_to_a_lagged'] = lagged_data['theta_b_to_a'].shift(-lag)
    lagged_data['liquiditeit_a_lagged'] = lagged_data['liquiditeit_a'].shift(-lag)
    lagged_data['liquiditeit_b_lagged'] = lagged_data['liquiditeit_b'].shift(-lag)
    
    threshold_data = lagged_data[threshold_mask].dropna()
    
    if len(threshold_data) < 5:
        return None
    
    avg_mass_a = threshold_data['liquiditeit_a_lagged'].mean()
    avg_mass_b = threshold_data['liquiditeit_b_lagged'].mean()
    avg_theta_a = threshold_data['theta_a_to_b_lagged'].mean()
    avg_theta_b = threshold_data['theta_b_to_a_lagged'].mean()
    
    supported_events = 0
    total_events = len(threshold_data)
    
    for idx, row in threshold_data.iterrows():
        mass_a_event = row['liquiditeit_a_lagged']
        mass_b_event = row['liquiditeit_b_lagged']
        theta_a_event = row['theta_a_to_b_lagged']
        theta_b_event = row['theta_b_to_a_lagged']
        
        if mass_a_event < mass_b_event:
            if theta_a_event > theta_b_event:
                supported_events += 1
        else:
            if theta_b_event > theta_a_event:
                supported_events += 1
    
    if avg_mass_a < avg_mass_b:
        hypothesis_supported = avg_theta_a > avg_theta_b
        expected_higher_theta = 'A'
    else:
        hypothesis_supported = avg_theta_b > avg_theta_a
        expected_higher_theta = 'B'
    
    actual_higher_theta = 'A' if avg_theta_a > avg_theta_b else 'B'
    
    event_success_rate = supported_events / total_events
    
    return {
        'pair': f"{ticker_a}-{ticker_b}",
        'avg_mass_a': avg_mass_a,
        'avg_mass_b': avg_mass_b,
        'lower_mass': 'A' if avg_mass_a < avg_mass_b else 'B',
        'avg_theta_a': avg_theta_a,
        'avg_theta_b': avg_theta_b,
        'higher_theta': actual_higher_theta,
        'expected_higher_theta': expected_higher_theta,
        'hypothesis_supported': hypothesis_supported,
        'threshold_events': total_events,
        'supported_events': supported_events,
        'event_success_rate': event_success_rate,
        'threshold_used': threshold,
        'lag_used': lag,
        'total_observations': len(combined_data)
    }

def run_threshold_hypothesis_test(threshold=0.005, lag=1):
    """Run the refined hypothesis test across all pairs for a given threshold/lag."""

    # I got these pairs from hypo1.py output.
    pairs = [
        ["NZDJPY", "AUDJPY"],
        ["AUDJPY", "NZDJPY"],
        ["CHFJPY", "CADJPY"],
        ["CADJPY", "CHFJPY"],
        ["GBPUSD", "AUDUSD"],
        ["NZDUSD", "EURUSD"],
        ["AUDNZD", "EURNZD"],
        ["NZDCAD", "USDCAD"]
    ]
    
    print("REFINED MASS-OU HYPOTHESIS TEST (THRESHOLD + LAG)")
    print("=" * 70)
    print(f"Threshold (z-score): {threshold}")
    print(f"Lag: {lag} period(s)")
    print("=" * 70)
    
    results = []
    
    for ticker_a, ticker_b in pairs:
        print(f"\nTesting {ticker_a} vs {ticker_b}...")
        result = threshold_based_hypothesis_test(ticker_a, ticker_b, threshold=threshold, lag=lag)
        
        if result:
            results.append(result)
            print(f"  Lagere massa: {result['lower_mass']}")
            print(f"  Hogere theta: {result['higher_theta']}")
            print(f"  Overall hypothese: {result['hypothesis_supported']}")
            print(f"  Theta A: {result['avg_theta_a']:.4f}")
            print(f"  Theta B: {result['avg_theta_b']:.4f}")
        else:
            print(f"  ERROR: Onvoldoende data of threshold events")
    
    if results:

    # Binomial test: treat successes as Bernoulli trials; under H0 (symmetry) X ~ Binom(n, 0.5) → p-value (upper-tail) = 1 - CDF(k-1, n, 0.5).
    # Compare p < 0.05 to reject H0 for "more successes than chance".

        # Overall binomial (pair-level)
        overall_successes = sum(1 for r in results if r['hypothesis_supported'])
        total_pairs = len(results)
        
        # Binomiale test voor event-based analyse
        total_events = sum(r['threshold_events'] for r in results)
        total_supported_events = sum(r['supported_events'] for r in results)
        
        # Event-based binomial
        p_value_overall = 1 - binom.cdf(overall_successes - 1, total_pairs, 0.5)
        
        # Event-based p-waarde
        p_value_events = 1 - binom.cdf(total_supported_events - 1, total_events, 0.5)
        
        print(f"\n{'='*70}")
        print(f"EINDRESULTAAT:")
        print(f"{'='*70}")
        print(f"OVERALL ANALYSE:")
        print(f"  Paren met ondersteuning: {overall_successes}/{total_pairs} ({overall_successes/total_pairs:.1%})")
        print(f"  Overall p-waarde: {p_value_overall:.4f}")
        
        print(f"\nEVENT-BASED ANALYSE:")
        print(f"  Ondersteunde events: {total_supported_events}/{total_events} ({total_supported_events/total_events:.1%})")
        print(f"  Event-based p-waarde: {p_value_events:.4f}")
        
        # Conclusie op basis van beide testen
        overall_significant = p_value_overall < 0.05
        events_significant = p_value_events < 0.05
        
        print(f"\n*** CONCLUSIES: ***")
        if overall_significant:
            print(f"OVERALL: VERWERP H0 - ACCEPTEER H1")
        else:
            print(f"OVERALL: ACCEPTEER H0 - VERWERP H1")
            
        if events_significant:
            print(f"EVENTS: VERWERP H0 - ACCEPTEER H1")
        else:
            print(f"EVENTS: ACCEPTEER H0 - VERWERP H1")
        
        print(f"{'='*70}")
        
        # Detailoverzicht
        print(f"\nDETAIL OVERZICHT:")
        for r in results:
            support = "H1" if r['hypothesis_supported'] else "H0"
            print(f"{support} {r['pair']}: Events={r['threshold_events']}, "
                  f"Success_rate={r['event_success_rate']:.1%}, "
                  f"Lower_mass={r['lower_mass']}, Higher_theta={r['higher_theta']}")
        
        return {
            'overall_successes': overall_successes,
            'total_pairs': total_pairs,
            'overall_success_rate': overall_successes/total_pairs,
            'p_value_overall': p_value_overall,
            'total_events': total_events,
            'total_supported_events': total_supported_events,
            'event_success_rate': total_supported_events/total_events,
            'p_value_events': p_value_events,
            'overall_significant': overall_significant,
            'events_significant': events_significant,
            'threshold': threshold,
            'lag': lag
        }
    
    return None

if __name__ == '__main__':
    # Test with different thresholds and lags
    print("Testing with threshold=1.5, lag=1")
    result1 = run_threshold_hypothesis_test(threshold=1.5, lag=1)
    
    print("\n" + "="*70)
    print("Testing with threshold=2.0, lag=4")
    result2 = run_threshold_hypothesis_test(threshold=2.0, lag=4)
    
    print("\n" + "="*70)
    print("Testing with threshold=2.5, lag=10")
    result3 = run_threshold_hypothesis_test(threshold=2.5, lag=10)
