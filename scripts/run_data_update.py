import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_ingestion import build_price_matrices, apply_corporate_action_adjustments

def main():
    os.makedirs('data/clean', exist_ok=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3) # 3-Year rolling window
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"--> Initiating Data Pipeline from {start_str} to {end_str}")
    
    close_p, high_p, low_p, vol_p = build_price_matrices(start_str, end_str)
    
    adj_close, adj_high, adj_low, adj_vol = apply_corporate_action_adjustments(
        close_p, high_p, low_p, vol_p, start_str, end_str
    )
    
    # Save optimized binary Parquet datasets
    adj_close.to_parquet('data/clean/adj_close.parquet')
    adj_high.to_parquet('data/clean/adj_high.parquet')
    adj_low.to_parquet('data/clean/adj_low.parquet')
    adj_vol.to_parquet('data/clean/adj_vol.parquet')
    
    print("--> Data Pipeline completed successfully. Parquet files saved.")

if __name__ == '__main__':
    main()