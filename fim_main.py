import argparse
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from algorithms import Apriori, NS_FIM, OptimizedApriori
import psutil
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def load_data(file_path, limit=None):
    transactions = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                transaction = [int(item) for item in line.strip().split() if item]
                if transaction:
                    transactions.append(transaction)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return transactions

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def run_experiment(dataset_name, transactions, min_sup_levels, alg_classes, timeout_sec=60):
    results = []
    for min_sup in min_sup_levels:
        print(f"Running experiments for {dataset_name} with min_sup={min_sup}")
        for alg_name, alg_class, alg_args in alg_classes:
            print(f"  Testing {alg_name}...")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)
            
            try:
                start_mem = get_memory_usage()
                alg = alg_class(transactions, min_sup, **alg_args)
                frequent_itemsets, duration = alg.run()
                end_mem = get_memory_usage()
                peak_mem = end_mem - start_mem if end_mem > start_mem else 0
                num_frequent = sum(len(lvl) for lvl in frequent_itemsets.values())
                
                results.append({
                    'Dataset': dataset_name, 'Algorithm': alg_name, 'min_sup': min_sup,
                    'Time (s)': duration, 'RAM (MB)': peak_mem, 'Frequent Itemsets': num_frequent,
                    'Status': 'Success'
                })
            except TimeoutException:
                print(f"    {alg_name} timed out after {timeout_sec}s")
                results.append({
                    'Dataset': dataset_name, 'Algorithm': alg_name, 'min_sup': min_sup,
                    'Time (s)': timeout_sec, 'RAM (MB)': 0, 'Frequent Itemsets': 0,
                    'Status': 'Timeout'
                })
            except Exception as e:
                print(f"    Failed {alg_name}: {e}")
            finally:
                signal.alarm(0)
    return results

def plot_results(df, dataset_name):
    success_df = df[df['Status'] == 'Success']
    if success_df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=success_df, x='min_sup', y='Time (s)', hue='Algorithm', marker='o')
    plt.title(f'Execution Time vs Minimum Support ({dataset_name})')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'{dataset_name}_time.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=success_df, x='min_sup', y='RAM (MB)', hue='Algorithm', marker='o')
    plt.title(f'Peak RAM Usage vs Minimum Support ({dataset_name})')
    plt.grid(True)
    plt.savefig(f'{dataset_name}_ram.png')
    plt.close()

def main():
    datasets = {'Chess': 'chess.txt', 'Connect': 'connect.txt', 'Accidents': 'accidents.txt'}
    all_results = []
    algs = [
        ('Apriori', Apriori, {}),
        ('NS-FIM (SOTA)', NS_FIM, {}),
        ('Optimized (Bitset)', OptimizedApriori, {'parallel': False}),
        ('Optimized (Parallel)', OptimizedApriori, {'parallel': True})
    ]

    for name, path in datasets.items():
        if not os.path.exists(path): continue
        print(f"\nProcessing {name} dataset...")
        sample_size = 1000 if name != 'Chess' else 3000
        transactions = load_data(path, limit=sample_size)
        
        if name == 'Chess': min_sup_levels = [0.8, 0.85, 0.9]
        elif name == 'Connect': min_sup_levels = [0.95, 0.97, 0.99]
        else: min_sup_levels = [0.6, 0.7, 0.8] # Accidents
        
        results = run_experiment(name, transactions, min_sup_levels, algs, timeout_sec=30)
        all_results.extend(results)
        df = pd.DataFrame(results)
        plot_results(df, name)

    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv('experiment_results.csv', index=False)
        print("\nExperiments complete. Results saved to experiment_results.csv and plots generated.")

if __name__ == "__main__":
    main()
