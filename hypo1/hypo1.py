"""
Lag/Lead Hypothesis Test FX Pairs
-----------------------------------

Hypothesis:
A strong move in pair A is followed within X candles by a move in pair B
in the same direction.

- H1: Pair B follows pair A within X candles.
- H0: Pair B does NOT follow pair A within X candles.

Output:
Log file plus DataFrame with significance per pair, lag and threshold.

Author: Tim (2025)
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(
    filename='resultaten_gridsearch_hyp1.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    """Load parquet file for a given ticker and timeframe."""
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

def lag_functie(ticker_a, ticker_b, timeframe, lag, threshold):
    """
    Test if returns of B (lagged) systematically move with
    extreme returns of A above a threshold.
    """
    df_a = data_inladen(timeframe, ticker_a)
    df_b = data_inladen(timeframe, ticker_b)

    if df_a is False or df_b is False or df_a.empty or df_b.empty:
        return None

    merged = pd.merge(
        df_a[['close']], df_b[['close']],
        left_index=True, right_index=True,
        suffixes=('_a', '_b')
    )

    if merged.empty:
        return None

    # log returns
    merged['ret_a'] = np.log(merged['close_a'] / merged['close_a'].shift(1))
    merged['ret_b'] = np.log(merged['close_b'] / merged['close_b'].shift(1))

    # strong move on pair A
    merged['sterke_beweging'] = (np.abs(merged['ret_a']) > threshold)

    merged['ret_b_lagged'] = merged['ret_b'].shift(-lag)

    # ~ == not
    mask = merged['sterke_beweging'] & ~merged['ret_a'].isna() & ~merged['ret_b_lagged'].isna()

    if mask.sum() < 10:
        return None

    a_moves = merged.loc[mask, 'ret_a']
    b_moves = merged.loc[mask, 'ret_b_lagged']

    aantal_tests = len(a_moves)

    # Pearson r (guard constant arrays)
    try:
        corr, p_value = pearsonr(a_moves, b_moves)
    except Exception:
        corr, p_value = np.nan, np.nan

    # | == or
    # check if same direction after lag
    same_dir = ((a_moves > 0) & (b_moves > 0)) | ((a_moves < 0) & (b_moves < 0))
    pct_same = same_dir.mean() * 100

    return {
        'van': ticker_a,
        'naar': ticker_b,
        'corr': corr,
        'p_value': p_value,
        'pct same': pct_same,
        'lag': lag,
        'threshold': threshold,
        'aantal_tests': aantal_tests
    }

def loopen(paar, timeframe, lag, threshold):
    """Run the lag test across all pair combinations."""
    resultaten = []
    for i in paar:
        for j in paar:
            if i != j:
                print(f'testing pair {i} -> {j}')
                result = lag_functie(i, j, timeframe, lag, threshold)
                if result is not None:
                    resultaten.append(result)
    if not resultaten:
        return None
    return pd.DataFrame(resultaten)

def resultaat(paar, timeframe, lag, threshold):
    """Report significant results and log best combinations."""
    print('-' * 50)
    print('hypothesis test results')
    print('-' * 50)
    alpha = 0.05

    df = loopen(paar, timeframe, lag, threshold)

    if df is None or df.empty:
        print("No valid results to analyze.")
        return None

    sig_dif = df[df['p_value'] < alpha]
    if len(sig_dif) > 0:
        print('')
        print('best combinations:')
        top_5 = sig_dif.nlargest(5, 'corr')
        for _, row in top_5.iterrows():
            logging.info(
                f"{row['van']} -> {row['naar']}: aantal testen= {row['aantal_tests']}, "
                f"p={row['p_value']:.4f}, lag={row['lag']}, threshold={threshold}"
            )
        logging.info("\nCONCLUSION: H1 ACCEPTED - there are significant relationships")
    else:
        print("\nCONCLUSION: H0 ACCEPTED - No significant relationships")
    return df

def main():
    paren = [
        ["EURUSD", "GBPUSD"],     # Majors tegen USD
        ["EURUSD", "AUDUSD"],
        ["EURUSD", "NZDUSD"],
        ["GBPUSD", "AUDUSD"],
        ["GBPUSD", "NZDUSD"],
        ["AUDUSD", "NZDUSD"],
        ["USDJPY", "EURJPY"],     # JPY-crosses
        ["USDJPY", "GBPJPY"],
        ["USDJPY", "AUDJPY"],
        ["USDJPY", "CHFJPY"],
        ["USDJPY", "NZDJPY"],
        ["EURJPY", "GBPJPY"],
        ["USDCHF", "EURCHF"],     # CHF-paren
        ["USDCHF", "CHFJPY"],
        ["EURCHF", "CHFJPY"],
        ["GBPCHF", "USDCHF"],
        ["USDCAD", "AUDCAD"],     # CAD-paren
        ["USDCAD", "NZDCAD"],
        ["AUDCAD", "NZDCAD"],
        ["EURCAD", "GBPCAD"],
        ["AUDNZD", "EURNZD"],     # NZD-crosses
        ["AUDNZD", "GBPNZD"],
        ["EURGBP", "GBPUSD"],     # EUR/GBP koppels
        ["EURGBP", "EURUSD"],
        ["AUDCHF", "CADCHF"],     # CHF-crosses
        ["AUDCHF", "GBPCHF"],
        ["CADJPY", "CHFJPY"],     # JPY-paren met minder frequentie
        ["AUDJPY", "NZDJPY"],
    ]

    total_tests = 0
    sig_tests = 0
    alpha = 0.05

    # small gridsearch
    lag = [1, 2, 4, 6, 8, 10]
    threshold_list = [0.005, 0.0075, 0.01, 0.0125, 0.02]
    timeframe = '5min'

    for paar in paren:
        for i in lag:
            for thresh in threshold_list:
                x = resultaat(paar, timeframe, i, thresh)
                if x is None or x.empty:
                    continue
                n = len(x)
                s = (x['p_value'] < alpha).sum()
                sig_tests += s
                total_tests += n

    print(f'total number of tests: {total_tests}')
    print(f'total number of significant tests: {sig_tests}')
    return x

if __name__ == '__main__':
    main()
