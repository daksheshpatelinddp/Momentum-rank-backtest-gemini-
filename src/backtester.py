import pandas as pd
import numpy as np
import vectorbt as vbt

def execute_rank_backtest(
    close_prices: pd.DataFrame,
    technical_gate: pd.DataFrame,
    rank_matrix: pd.DataFrame,
    top_n_entry: int = 10,
    exit_rank_m: int = 25,
    rebalance_freq: str = '1M',
    fee_rate: float = 0.0015
) -> dict:
    """Executes dynamic portfolio backtest with N-Entry and M-Exit buffer rules."""
    rebalance_dates = close_prices.resample(rebalance_freq).last().index
    
    entries = pd.DataFrame(False, index=close_prices.index, columns=close_prices.columns)
    exits = pd.DataFrame(False, index=close_prices.index, columns=close_prices.columns)

    sma_200 = close_prices.rolling(window=200).mean()

    for date in rebalance_dates:
        if date in rank_matrix.index:
            day_ranks = rank_matrix.loc[date]
            tech_pass = technical_gate.loc[date]

            # Entry Logic: Top N Rank + Passes Technical Filters
            entries.loc[date] = (day_ranks <= top_n_entry) & tech_pass

            # Exit Logic: Rank drops past M OR stock drops below 200 SMA
            below_sma = close_prices.loc[date] < sma_200.loc[date]
            exits.loc[date] = (day_ranks > exit_rank_m) | below_sma

    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=entries,
        exits=exits,
        init_cash=1000000, # ₹10 Lakhs
        fees=fee_rate,
        freq='1D'
    )

    stats = {
        'CAGR (%)': portfolio.total_return().mean() * 100, # Aggregate portfolio estimate
        'Sharpe': portfolio.sharpe_ratio().mean(),
        'Max Drawdown (%)': portfolio.max_drawdown().mean() * 100,
        'Trades': portfolio.trades.count().sum()
    }
    return stats