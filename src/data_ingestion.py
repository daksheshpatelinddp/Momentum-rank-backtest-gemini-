import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_single_bhavcopy(date_obj: datetime) -> pd.DataFrame:
    """Downloads and cleans NSE Bhavcopy for a given date."""
    date_str = date_obj.strftime('%d%b%Y').upper()
    year_str = date_obj.strftime('%Y')
    month_str = date_obj.strftime('%b').upper()
    
    url = f"https://archives.nseindia.com/content/historical/EQUITIES/{year_str}/{month_str}/cm{date_str}bhav.csv.zip"
    
    try:
        response = requests.get(url, headers=NSE_HEADERS, timeout=10)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f)
                    df = df[df['SERIES'] == 'EQ'].copy()
                    df['DATE'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y')
                    return df[['DATE', 'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TOTTRDVAL']]
    except Exception as e:
        pass
    return None

def build_price_matrices(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Downloads range of Bhavcopies and pivots them into Close, High, Low, Volume matrices."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    all_records = []
    
    print(f"--> Starting download for {len(dates)} business days...")
    for idx, d in enumerate(dates):
        df = fetch_single_bhavcopy(d)
        if df is not None:
            all_records.append(df)
        if (idx + 1) % 50 == 0:
            print(f"     Processed {idx + 1}/{len(dates)} days...")

    if not all_records:
        raise ValueError("No Bhavcopy data downloaded. Verify date ranges or network access.")
        
    full_df = pd.concat(all_records, ignore_index=True)
    
    close_p = full_df.pivot(index='DATE', columns='SYMBOL', values='CLOSE')
    high_p = full_df.pivot(index='DATE', columns='SYMBOL', values='HIGH')
    low_p = full_df.pivot(index='DATE', columns='SYMBOL', values='LOW')
    vol_p = full_df.pivot(index='DATE', columns='SYMBOL', values='TOTTRDQTY')
    
    return close_p, high_p, low_p, vol_p

def apply_corporate_action_adjustments(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame, vol_df: pd.DataFrame, corp_actions_csv: str = None) -> tuple:
    """
    Adjusts historical prices for Splits & Bonuses.
    Reads corporate actions DataFrame containing: ['SYMBOL', 'EX_DATE', 'SPLIT_RATIO']
    """
    adj_close = close_df.copy()
    adj_high = high_df.copy()
    adj_low = low_df.copy()
    adj_vol = vol_df.copy()

    if corp_actions_csv and os.path.exists(corp_actions_csv):
        ca_df = pd.read_csv(corp_actions_csv)
        ca_df['EX_DATE'] = pd.to_datetime(ca_df['EX_DATE'])

        for _, row in ca_df.iterrows():
            sym = row['SYMBOL']
            ex_date = row['EX_DATE']
            ratio = float(row['SPLIT_RATIO']) # e.g., 2.0 for 1:2 split or 2:1 bonus

            if sym in adj_close.columns:
                # Multiply pre-ex-date prices by adjustment factor (1/ratio)
                mask = adj_close.index < ex_date
                adj_close.loc[mask, sym] = adj_close.loc[mask, sym] / ratio
                adj_high.loc[mask, sym] = adj_high.loc[mask, sym] / ratio
                adj_low.loc[mask, sym] = adj_low.loc[mask, sym] / ratio
                # Volume is inverse-adjusted
                adj_vol.loc[mask, sym] = adj_vol.loc[mask, sym] * ratio

    # Forward-fill gaps caused by holidays or missing trading ticks
    adj_close = adj_close.ffill().bfill()
    adj_high = adj_high.ffill().bfill()
    adj_low = adj_low.ffill().bfill()
    adj_vol = adj_vol.fillna(0)

    return adj_close, adj_high, adj_low, adj_vol