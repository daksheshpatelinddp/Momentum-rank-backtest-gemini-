import os
import pandas as pd
from tabulate import tabulate
from src.indicators import compute_indicator_gate
from src.backtester import execute_rank_backtest

def main():
    if not os.path.exists('data/clean/adj_close.parquet'):
        raise FileNotFoundError("Clean dataset not found. Please run the Data Pipeline workflow first.")

    close_p = pd.read_parquet('data/clean/adj_close.parquet')
    vol_p = pd.read_parquet('data/clean/adj_vol.parquet')

    # Calculate Indicator Gate & Rank Matrix
    tech_gate, rank_matrix = compute_indicator_gate(close_p, vol_p, min_turnover_cr=1.0)

    # Grid Search Testing Configurations
    test_params = [
        (10, 15), # Hold Top 10, exit past 15
        (10, 25), # Hold Top 10, exit past 25 (Wide Hysteresis)
        (20, 30), # Hold Top 20, exit past 30
        (30, 50), # Hold Top 30, exit past 50
    ]

    results = []

    print("--> Running Parameter Sweep Grid...")
    for top_n, exit_m in test_params:
        res = execute_rank_backtest(
            close_p, tech_gate, rank_matrix, 
            top_n_entry=top_n, exit_rank_m=exit_m
        )
        res['Entry (N)'] = top_n
        res['Exit (M)'] = exit_m
        results.append(res)

    df_res = pd.DataFrame(results)
    df_res = df_res[['Entry (N)', 'Exit (M)', 'CAGR (%)', 'Sharpe', 'Max Drawdown (%)', 'Trades']]
    
    print("\n" + tabulate(df_res, headers='keys', tablefmt='github'))

    # Save summary report artifact
    os.makedirs('reports', exist_ok=True)
    df_res.to_csv('reports/backtest_summary.csv', index=False)

if __name__ == '__main__':
    main()