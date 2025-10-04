
# Multi-Strategy Combinatorial Optimization Framework

A Python framework for experimenting with multiple solution strategies on three classical combinatorial optimization problems.
Easily run, compare, and benchmark algorithms with ready-to-use datasets

- **TSP** (Travelling Salesperson)
  - Greedy (Nearest‑Neighbor + 2‑Opt polish)
  - Divide & Conquer (median split + stitch + 2‑Opt)
  - Dynamic Programming (Held‑Karp exact)
  - Backtracking (DFS + optimistic bound)
  - Branch & Bound: Priority-Queue guided pruning

- **0/1 Knapsack**
  - Greedy by value/weight ratio
  - Divide & Conquer (meet‑in‑the‑middle)
  - Dynamic Programming (exact table)
  - Backtracking (fractional bound)
  - Branch & Bound (PQ + fractional bound)

- **Assignment (Min‑Cost Bipartite Matching)**
  - Greedy row minima (heuristic)
  - Divide & Conquer (block heuristic)
  - Dynamic Programming (bitmask)
  - Backtracking (optimistic bound)
  - Hungarian algorithm (optimal, O(n³))

## Quick start
Create a virtual environment, install dependencies, and run benchmarks:
```bash
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m benchmarks.benchmark
```

Artifacts land in `benchmarks/out/` (PNG plots + JSON).

## CLI
🖥️ Command Line Usage
```bash
# TSP
python cli.py --problem tsp --algo greedy --data benchmarks/datasets/tsp_cities.csv

# Knapsack (requires capacity)
python cli.py --problem knapsack --algo dp --data benchmarks/datasets/knapsack_items.csv --capacity 50

# Assignment
python cli.py --problem assignment --algo hungarian --data benchmarks/datasets/assignment_cost.csv
```

## Datasets

- `benchmarks/datasets/tsp_cities.csv` — small 2D coordinates (x,y).
- `benchmarks/datasets/knapsack_items.csv` — (value, weight) items.
- `benchmarks/datasets/assignment_cost.csv` — square cost matrix CSV.

Easily replace with real-world datasets:

(1)TSP → convert TSPLIB .tsp files to CSV
(2)Knapsack → merchandising packs, procurement datasets
(3)Assignment → workforce scheduling, shift planning matrices
(4)Dataset handling is flexible and documented in cli.py.
