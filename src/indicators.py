import pandas as pd
import numpy as np

def calculate_rsi(price_df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates RSI across all symbol columns simultaneously."""
    delta = price_df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_indicator_gate(
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    min_turnover_cr: float = 1.0,
    rsi_min: float = 45,
    rsi_max: float = 70,
    vol_surge_mult: float = 1.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates Technical Gate Mask and Risk-Adjusted Momentum Rank Matrix."""
    # 1. Trend Filter (Close > 200 SMA)
    sma_200 = close_prices.rolling(window=200).mean()
    trend_filter = close_prices > sma_200

    # 2. RSI Gate
    rsi_df = calculate_rsi(close_prices, window=14)
    rsi_filter = (rsi_df >= rsi_min) & (rsi_df <= rsi_max)

    # 3. Volume Expansion Gate
    vol_sma_20 = volumes.rolling(window=20).mean()
    vol_filter = volumes >= (vol_sma_20 * vol_surge_mult)

    # 4. Hard Liquidity Filter (Turnover >= min_turnover_cr)
    turnover_cr = (close_prices * volumes) / 1e7
    liquidity_filter = turnover_cr.rolling(window=20).mean() >= min_turnover_cr

    # Master Technical Gate
    technical_gate = trend_filter & rsi_filter & vol_filter & liquidity_filter

    # 5. Risk-Adjusted Momentum (6M return / annualized vol)
    ret_6m = close_prices.shift(10) / close_prices.shift(126) - 1
    ann_vol = close_prices.pct_change().rolling(window=126).std() * np.sqrt(252)
    mom_score = ret_6m / ann_vol

    # Rank across Universe (Rank 1 = Highest Momentum)
    valid_scores = mom_score.where(liquidity_filter, np.nan)
    rank_matrix = valid_scores.rank(axis=1, ascending=False, method='min')

    return technical_gate, rank_matrix