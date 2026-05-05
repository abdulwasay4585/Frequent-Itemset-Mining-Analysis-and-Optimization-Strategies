# Frequent Itemset Mining: Analysis and Optimization Strategies

## Overview
This repository contains a comprehensive study and implementation of Frequent Itemset Mining (FIM) algorithms, developed for the CS-378 Design and Analysis of Algorithms course. The project focuses on a comparative analysis between the classic Apriori algorithm and a modern state-of-the-art approach (NS-FIM 2022), alongside proposed optimization strategies including bitset-based support counting and multi-core parallelization.

## Core Algorithms

### 1. Apriori Algorithm
The fundamental breadth-first search algorithm for frequent itemset mining. It utilizes the anti-monotonicity property (Apriori property) to prune the search space by ensuring that any subset of a frequent itemset must also be frequent.

### 2. NS-FIM (SOTA 2022)
An implementation inspired by modern Non-Sparse Frequent Itemset Mining techniques. This algorithm is specifically designed for dense datasets, utilizing a vertical data representation and bitset intersections to rapidly calculate support counts without repeated database scans.

## Optimization Strategies

### Bitset-based Support Counting
Instead of traditional horizontal database scans, this optimization represents each item's presence across transactions as a bitset (using NumPy boolean arrays). Support counting is reduced to bitwise AND operations, which are highly optimized at the hardware level, resulting in significant speedups.

### Parallel Candidate Evaluation
To address the computational bottleneck of support counting in large candidate sets, we implemented a parallelized version of the Apriori algorithm. It distributes candidate evaluation across multiple CPU cores using Python's multiprocessing module, effectively reducing wall-clock time on multi-core systems.

## Project Structure

- **algorithms.py**: Contains the core implementations of Apriori, NS-FIM, and optimized variants.
- **fim_main.py**: The primary execution script for running benchmarks across different datasets and support levels.
- **experiment_results.csv**: Automated output log containing performance metrics (Time, RAM usage, Frequent Itemsets found).
- **FIM_Report_Draft.md**: Technical documentation and preliminary analysis of the findings.

## Datasets
The project utilizes several standard FIM benchmark datasets:
- **Chess**: A high-density dataset (3,196 transactions, 75 items).
- **Connect**: A dense dataset derived from game states (67,557 transactions, 129 items).
- **Accidents**: A large, dense dataset (340,183 transactions, 468 items).

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed. The following libraries are required:
- numpy
- pandas
- matplotlib
- seaborn
- psutil

### Setup
Install the dependencies using pip:
```bash
pip install numpy pandas matplotlib seaborn psutil
```

## Usage

To run the full benchmarking suite and generate performance plots, execute the main script:
```bash
python fim_main.py
```

The script will:
1. Load the benchmark datasets.
2. Run each algorithm variant across multiple minimum support levels.
3. Track peak memory usage and execution time.
4. Generate comparative plots (Time vs Min Support and RAM vs Min Support) for each dataset.
5. Save the raw data to `experiment_results.csv`.

## Results and Performance
Initial experiments demonstrate that:
- NS-FIM significantly outperforms the standard Apriori algorithm on dense datasets.
- Bitset optimization provides a substantial performance boost (often exceeding 10x) compared to naive subset checking.
- Parallelization scales effectively for large candidate sets but introduces overhead for smaller datasets or high support levels.

## Authors
Developed as part of the CS-378: Design and Analysis of Algorithms course at GIK Institute.
