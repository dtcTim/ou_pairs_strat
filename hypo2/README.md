# Hypothesis 2: Mass–OU Dynamics in FX Pairs

## Overview
This project tests whether differences in *mass* and *mean-reversion speed* (θ) across FX pairs can explain predictive lead–lag dynamics.  

The idea:  
- If one pair has a **lower mass** (faster response)  
- And another has a **higher mass (θ)** (stronger pullback)  
 then the first should anticipate moves in the second after extreme events.

- **H1 (alternative hypothesis):** FX pairs with lower mass systematically lead pairs with higher θ after threshold events.  
- **H0 (null hypothesis):** No systematic predictive relationship exists.

- ## Methodology
- **Data:** 5-minute log returns on correlated (based on hypo1) FX pairs (JPY, USD, CAD, NZD crosses).  
- **Filtering:** events defined by exceeding volatility thresholds (z > 1.5, 2.0, 2.5).  
- **Test logic:** check whether leader/follower structure (lower mass moves to higher θ) predicts outcomes at lags of 1, 4, 10.  
- **Evaluation:**  
  - *Overall analysis* (pair-level)  
  - *Event-based analysis* (per extreme move).
 
## Results

### Overall pair-level tests
- Threshold **1.5**, lag **1** → **0/8** pairs supported H1.  
- Threshold **2.0**, lag **4** → **1/8** pairs supported H1 (*NZDUSD → EURUSD*).  
- Threshold **2.5**, lag **10** → **1/8** pairs supported H1 (*NZDUSD → EURUSD*).

### Event-level tests
- Success rate: ~30–40% across thresholds.  
- Not statistically significant (*p ≈ 1.0*).

### Theta patterns
- Some alignment in **NZDUSD → EURUSD** at higher thresholds/lags.  
- Evidence was **isolated and weak**.

## Sample Results

| Pair            | Threshold | Lag | Supported | Success Rate | Notes                  |
|-----------------|-----------|-----|-----------|--------------|------------------------|
| NZDJPY → AUDJPY | 1.5       | 1   | No        | 38.5%        | Symmetric, no edge     |
| NZDUSD → EURUSD | 2.0       | 4   | Yes       | 39.8%        | Weak evidence for H1   |
| NZDUSD → EURUSD | 2.5       | 10  | Yes       | 41.1%        | Only at high threshold |

## Conclusions
- **H0 accepted:** no robust mass–OU asymmetry across tested FX pairs.  
- Only isolated evidence for *NZDUSD → EURUSD* at higher thresholds/lags.  
- Results suggest that OU parameters (mass, θ) capture mean-reversion profiles but do **not produce systematic lead–lag predictability** in this dataset.
