# Mass–OU FX Hypotheses

## Overview
This repository explores **quantitative hypotheses in FX markets** by borrowing concepts from physics and applying them to financial time series.  

Just as physical systems can exhibit **mass, inertia, and oscillatory dynamics**, currency pairs may also display structural relationships that are not obvious from price charts alone.  
By treating FX pairs as interacting systems, we test whether ideas such as **lead–lag causality** and **Ornstein–Uhlenbeck dynamics with asymmetry (mass/θ effects)** can generate predictive signals.

The core goal is to bridge **stochastic modeling and market microstructure**:  
- Can moves in one pair consistently *pull* or *push* another (lead–lag)?  
- Do asymmetries in OU parameters explain why some currencies behave like "heavy masses" with slower reversion, while others move faster?  
- If such effects exist, are they strong and stable enough to be the foundation of a trading signal?  

This project is part of a broader effort to **translate concepts from physics into finance**, testing rigorously whether they provide statistical evidence for predictability.

The project is structured into two hypotheses:
- **Hypothesis 1 – Lead–Lag Effects in FX Pairs**  
  Do significant moves in currency pair A systematically lead pair B within X candles?  
  *Result: Evidence of strong synchronous clusters (JPY, CHF, CAD blocs), but mostly bidirectional. Suggests shared factors rather than clean causality.*

- **Hypothesis 2 – Mass–OU Dynamics in FX Pairs**  
  Does asymmetry in OU parameters (mass, θ) explain leader/follower structures after extreme events?  
  *Result: No robust evidence. One isolated case (NZDUSD → EURUSD), but not statistically significant overall (p ≈ 1).*
