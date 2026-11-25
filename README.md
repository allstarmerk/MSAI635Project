# MSAI635 Project IAMM in a Synthetic Market

## Overview

**Reinforcement Learning for Inventory Aware Market Making in a Synthetic Market**

This project trains an RL agent to set bid/ask quotes in a synthetic market, aiming for a better profitability-vs-inventory-risk tradeoff than a fixed-spread baseline market maker. The environment is a simulated limit order book with Brownian mid-price dynamics, Poisson order arrivals, and distance-dependent fill probabilities; the agent (tabular Q-learning / DQN, with PPO as a possible extension) observes inventory and time-remaining and chooses discrete bid/ask tick distances to maximize PnL net of an inventory-risk penalty.

Team: Johnathan Sheikh, Chenhongshu Yu, Yating Zhang (Group 14).

## Status

The synthetic market simulator (`market_making/`) is implemented and verified: cash/inventory accounting, Poisson arrival rates, distance-dependent fill probability, and Brownian price increments all check out against the report's Section 4 claims (`tests/test_simulator.py`, all passing). A Gymnasium environment wrapper and the fixed-spread baseline are in place.

Stage 2: tabular **Q-learning (off-policy)** and **SARSA (on-policy)** agents (`market_making/agents.py`) share one implementation and differ only in the bootstrap target. `run_experiment.py` trains both, evaluates them against the fixed-spread baseline on held-out seeds (PnL, RMS inventory, fill ratio, terminal inventory), and sweeps the inventory-penalty coefficient ζ to trace the profit/risk tradeoff. Inventory risk is controlled through ζ in the reward, not through the choice of algorithm.

## Setup

```
python -m venv .venv
.venv\Scripts\activate   # or: source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
python run_experiment.py                      # zeta=0.1, 30k training episodes per agent
python run_experiment.py --zeta 0 0.01 0.1 1  # penalty sweep
```

Results (CSV tables, Q-tables, policy heatmaps, learning curves, tradeoff plot) are written to `results/`.