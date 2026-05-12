import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_summary_plots(csv_path):
    df = pd.read_csv(csv_path)
    
    # 1. Speedup Plot (relative to Apriori on Chess at 0.8)
    chess_08 = df[(df['Dataset'] == 'Chess') & (df['min_sup'] == 0.8)]
    apriori_time = chess_08[chess_08['Algorithm'] == 'Apriori']['Time (s)'].values[0]
    
    chess_08['Speedup'] = apriori_time / chess_08['Time (s)']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Algorithm', y='Speedup', data=chess_08, palette='viridis')
    plt.title('Speedup Ratio Relative to Apriori (Chess, min_sup=0.80)')
    plt.ylabel('Speedup (x)')
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('fig5_speedup.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Itemset Count Plot (Chess dataset)
    chess_df = df[df['Dataset'] == 'Chess'].drop_duplicates(subset=['min_sup'])
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='min_sup', y='Frequent Itemsets', data=chess_df, marker='o', color='crimson')
    plt.title('Number of Frequent Itemsets vs Minimum Support (Chess)')
    plt.xlabel('Minimum Support')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('fig6_itemsets.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Summary plots (fig5_speedup.png and fig6_itemsets.png) regenerated.")

if __name__ == "__main__":
    generate_summary_plots('experiment_results.csv')
