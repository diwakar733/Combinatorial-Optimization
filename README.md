# Combinatorial Optimization — Multi‑Strategy Approach

**Build:** A unified framework to solve combinatorial optimization problems (TSP, Knapsack, Graph Matching, …) using multiple algorithmic strategies — Greedy heuristics, Divide & Conquer, Dynamic Programming (memoization), Backtracking search, and Branch & Bound pruning — and compare their performance on real‑world datasets.

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Supported Problems](#supported-problems)
* [Design & Architecture](#design--architecture)
* [Algorithms & Strategy Implementations](#algorithms--strategy-implementations)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Install](#install)
  * [Run examples](#run-examples)
* [Datasets](#datasets)
* [Evaluation & Benchmarking](#evaluation--benchmarking)
* [Configuration & Tuning](#configuration--tuning)
* [How to Add a New Problem or Strategy](#how-to-add-a-new-problem-or-strategy)
* [Project Roadmap](#project-roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Overview

This repository provides an extendable experimental playground for comparing algorithmic strategies on combinatorial optimization tasks. The goal is pedagogical and empirical: help students, researchers, and practitioners quickly prototype solutions, compare tradeoffs (solution quality, runtime, memory), and visualize results across realistic datasets.

The framework emphasizes:

* Clean modular structure so new problems/solvers plug in easily.
* Instrumentation for timing, profiling, and reproducible experiments.
* A small set of canonical problems and implementations of five algorithmic strategies.

---

## Key Features

* Pluggable problem interface for implementing new tasks.
* Implementations of Greedy, Divide & Conquer, Dynamic Programming (memo), Backtracking, and Branch & Bound strategies.
* CLI tools and example notebooks for running experiments and producing plots/tables.
* Support for importing common datasets (TSPLIB-like, knapsack instances, graph datasets).
* Benchmarking harness producing CSV/JSON results for analysis.

---

## Supported Problems

1. **Travelling Salesman Problem (TSP)** — symmetric and asymmetric variants, Euclidean instances.
2. **0/1 Knapsack** — weight/value integer instances.
3. **Maximum Matching / Graph problems** — small graph matching instances for demonstration.
4. **(Extendable)** — add other problems by implementing the `Problem` interface.

---

## Design & Architecture

* `problems/` — problem definitions + instance loaders.
* `strategies/` — algorithm implementations grouped by strategy.
* `bench/` — benchmarking harness & experiment runner.
* `utils/` — timing, logging, configuration parsing, reproducibility helpers.
* `notebooks/` — Jupyter notebooks demonstrating experiments and visualizations.
* `examples/` — small runnable example scripts.

The framework uses a lightweight interface pattern:

```py
class Problem:
    def load_instance(path) -> Instance: ...
    def validate(solution, instance) -> bool: ...
    def score(solution, instance) -> float: ...

class Strategy:
    def solve(instance, config) -> Solution: ...
```

This makes it easy to pair any strategy with any compatible problem instance.

---

## Algorithms & Strategy Implementations

Implementations provided (each in `strategies/`):

* **Greedy** — fast heuristic rules (nearest neighbor for TSP, value/weight ratio for knapsack).
* **Divide & Conquer** — problem splitting and merge (useful in specially structured instances).
* **Dynamic Programming (memoization)** — exact / pseudo-polynomial solutions when feasible (Knapsack DP, Held–Karp baseline for small TSP instances).
* **Backtracking** — exhaustive search with heuristics to reduce branching.
* **Branch & Bound** — best-first search with lower/upper bounds and pruning for TSP/Knapsack.

Each strategy records: runtime, peak memory estimate (if available), solution cost/value, and node/explored count (where applicable).

---

## Getting Started

### Prerequisites

* Python 3.10+ (recommended)
* pip
* Optional: Jupyter for notebooks

Install required packages:

```bash
pip install -r requirements.txt
```

(Requirements typically include `numpy`, `pandas`, `networkx`, `matplotlib`, and `tqdm`.)

### Install

Clone the repository:

```bash
git clone <repo-url>
cd combinatorial-multi-strategy
```

Install in editable mode (optional):

```bash
pip install -e .
```

### Run examples

Run a quick example comparing strategies on a small knapsack instance:

```bash
python examples/run_benchmark.py --problem knapsack --instances examples/data/knap-01.json --strategies greedy,dp,branch_and_bound --out results/knap-comparison.csv
```

Run the TSP demo notebook (visualizes routes):

```bash
jupyter notebook notebooks/tsp_demo.ipynb
```

---

## Datasets

* `data/` contains sample toy instances.
* The project includes importers for TSPLIB-style TSP files and common knapsack formats.
* To evaluate on real datasets, place your instances in `datasets/` and reference them in the benchmark CLI.

**Reproducibility:** Each experiment stores a JSON metadata file with seed, timestamp, machine info, and configuration.

---

## Evaluation & Benchmarking

The benchmarking harness produces CSV/JSON outputs with columns such as:
`instance,problem,strategy,seed,solution_value,optimality_gap,time_s,nodes_explored,memory_estimate`

Suggested workflow:

1. Run baseline experiments on small instances to validate correctness.
2. Scale instances until strategies div
