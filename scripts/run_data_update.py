import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_ingestion import build_price_matrices, apply_corporate_action_adjustments

def main():
    os.makedirs('data/clean', exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3) # 3-Year historical window
    
    print(f"--> Initiating Data Ingestion from {start_date.date()} to {end_date.date()}")
    
    close_p, high_p, low_p, vol_p = build_price_matrices(
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    # Apply corporate action split/bonus file if present
    ca_file = 'data/corporate_actions.csv'
    adj_close, adj_high, adj_low, adj_vol = apply_corporate_action_adjustments(
        close_p, high_p, low_p, vol_p, corp_actions_csv=ca_file
    )
    
    # Save optimized Parquet data files
    adj_close.to_parquet('data/clean/adj_close.parquet')
    adj_high.to_parquet('data/clean/adj_high.parquet')
    adj_low.to_parquet('data/clean/adj_low.parquet')
    adj_vol.to_parquet('data/clean/adj_vol.parquet')
    
    print("--> Data Ingestion Pipeline completed successfully. Parquet artifacts saved.")

if __name__ == '__main__':
    main()