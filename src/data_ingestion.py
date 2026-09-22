import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.nse_fetcher import fetch_single_bhavcopy, fetch_nse_corporate_actions

def parse_action_factor(subject_text: str) -> float:
    """
    Parses NSE corporate action text descriptions into multiplier ratios.
    Examples:
    - "Split - From Rs 10/- To Rs 2/-" -> Factor: 5.0
    - "Bonus 1:1" -> Factor: 2.0
    - "Bonus 1:2" -> Factor: 1.5
    """
    text = str(subject_text).lower()
    
    # 1. Check for Stock Splits (e.g., "Split From Rs 10 To Rs 2")
    if 'split' in text or 'sub-division' in text:
        nums = re.findall(r'\d+', text)
        if len(nums) >= 2:
            old_val, new_val = float(nums[0]), float(nums[1])
            if old_val > 0 and new_val > 0:
                return old_val / new_val

    # 2. Check for Bonus Issues (e.g., "Bonus 1:1", "Bonus 1:2")
    if 'bonus' in text:
        nums = re.findall(r'\d+', text)
        if len(nums) >= 2:
            bonus, held = float(nums[0]), float(nums[1])
            if held > 0:
                return (bonus + held) / held

    # 3. Check for Spinoffs / Demergers or Rights (fallback to gap ratio approximation)
    if 'demerger' in text or 'spinoff' in text or 'rights' in text:
        # Flag for adjustment handling
        return -1.0
        
    return 1.0

def build_price_matrices(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Downloads range of Bhavcopies and pivots into price/volume matrices."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    all_records = []
    
    print(f"--> Ingesting Bhavcopies for {len(dates)} business days...")
    for idx, d in enumerate(dates):
        df = fetch_single_bhavcopy(d)
        if df is not None:
            all_records.append(df)
        if (idx + 1) % 100 == 0:
            print(f"     Downloaded {idx + 1}/{len(dates)} days...")

    if not all_records:
        raise ValueError("No Bhavcopy data downloaded.")
        
    full_df = pd.concat(all_records, ignore_index=True)
    
    close_p = full_df.pivot(index='DATE', columns='SYMBOL', values='CLOSE')
    high_p = full_df.pivot(index='DATE', columns='SYMBOL', values='HIGH')
    low_p = full_df.pivot(index='DATE', columns='SYMBOL', values='LOW')
    vol_p = full_df.pivot(index='DATE', columns='SYMBOL', values='TOTTRDQTY')
    
    return close_p, high_p, low_p, vol_p

def apply_corporate_action_adjustments(
    close_df: pd.DataFrame, 
    high_df: pd.DataFrame, 
    low_df: pd.DataFrame, 
    vol_df: pd.DataFrame,
    start_date: str,
    end_date: str
) -> tuple:
    """Applies dynamic corporate action adjustments to price/volume matrices."""
    adj_close = close_df.copy()
    adj_high = high_df.copy()
    adj_low = low_df.copy()
    adj_vol = vol_df.copy()

    # Fetch Corporate Actions directly from NSE
    print("--> Fetching Corporate Actions from NSE API...")
    ca_df = fetch_nse_corporate_actions(start_date, end_date)
    
    if not ca_df.empty:
        for _, row in ca_df.iterrows():
            sym = row['SYMBOL']
            ex_date = row['EX_DATE']
            subject = row['SUBJECT']

            if sym in adj_close.columns:
                factor = parse_action_factor(subject)
                
                # If Spinoff/Demerger gap, compute adjustment ratio using ex-date open/close gap
                if factor == -1.0:
                    if ex_date in adj_close.index:
                        loc_idx = adj_close.index.get_loc(ex_date)
                        if loc_idx > 0:
                            pre_close = adj_close[sym].iloc[loc_idx - 1]
                            post_open = adj_close[sym].iloc[loc_idx]
                            if pre_close > 0 and post_open > 0 and (pre_close / post_open) > 1.15:
                                factor = pre_close / post_open

                # Apply factor to pre-ex-date historical prices
                if factor > 1.0:
                    mask = adj_close.index < ex_date
                    adj_close.loc[mask, sym] = adj_close.loc[mask, sym] / factor
                    adj_high.loc[mask, sym] = adj_high.loc[mask, sym] / factor
                    adj_low.loc[mask, sym] = adj_low.loc[mask, sym] / factor
                    adj_vol.loc[mask, sym] = adj_vol.loc[mask, sym] * factor

    # Forward fill missing trading days and backfill start
    adj_close = adj_close.ffill().bfill()
    adj_high = adj_high.ffill().bfill()
    adj_low = adj_low.ffill().bfill()
    adj_vol = adj_vol.fillna(0)

    return adj_close, adj_high, adj_low, adj_vol