# Purpose: Compute MA, RSI, returns, volume changes, etc.

import pandas as pd
import ta  # install with: pip install ta

def compute_features(df):
    df = df.copy()

    # Basic returns
    df['return'] = df['close'].pct_change()

    # Technical indicators
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_21'] = df['close'].rolling(window=21).mean()

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # Volume change
    df['volume_change'] = df['volume'].pct_change()

    # Drop rows with NaNs (due to rolling)
    df = df.dropna()

    return df
