# Hypothesis 1 – Lead–Lag Effects in FX Pairs

## Overview
This project tests the hypothesis that **a strong move in currency pair A is followed by a move in pair B within X candles**.  
In other words, we investigate whether lead–lag relationships exist between major FX pairs using log-returns and statistical testing.

- **H1 (alternative hypothesis):** Pair B follows pair A within X candles.  
- **H0 (null hypothesis):** No significant relationship exists.  

## Methodology
- Data: 5-minute OHLC (open, high, low, close) data for major and cross FX pairs.
- Processing: log returns computed per pair and merged by timestamp.
- Test: Pearson correlation and direction check applied when pair A exceeds a volatility threshold.
- Significance: pairs with *p < 0.05* considered statistically significant.

### Summary of findings
- Significant combos logged: 116 (each is a specific direction, lag, threshold with p < 0.05).
- Typical sample size: median tests per combo = 152 (range 10–674)
- Unique relationships: 31 directed; 18 undirected pairs; 13/18 show bidirectional significance → suggests common factor/synchronous moves rather than clean causality.
- Notable clusters: JPY crosses (USDJPY→EURJPY/CHFJPY/AUDJPY, EURJPY→GBPJPY), CAD/CHF cross-pair links (CADJPY↔CHFJPY; GBPCHF↔USDCHF), and CAD vs NZD (NZDCAD→USDCAD). Many are bidirectional, reinforcing the “shared factor” story (USD/JPY risk, regional blocs).

### Sample results (subset)
| Pair              | p-value | Lag | Conclusion  |
|-------------------|---------|-----|-------------|
| NZDUSD → EURUSD   | 0.0011  | 1   | H1 Accepted |
| GBPUSD → AUDUSD   | 0.0204  | 1   | H1 Accepted |
| AUDUSD → GBPUSD   | 0.0000  | 4   | H1 Accepted |

**Note:** Multiple additional FX pairs also showed significant lead–lag effects (see raw output).

**Conclusion:**  
H1 is supported in several FX clusters, but not universally. Many significant effects were observed in JPY and CAD/CHF crosses, often bidirectional.
This suggests that some moves are driven by common macro factors rather than pure causality.
