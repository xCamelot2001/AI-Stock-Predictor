# Purpose: Scaling (normalization), train/test split helpers

from sklearn.preprocessing import StandardScaler
import pandas as pd

def normalize_features(df, exclude_cols=[]):
    scaler = StandardScaler()
    cols = [col for col in df.columns if col not in exclude_cols]
    df_scaled = df.copy()
    df_scaled[cols] = scaler.fit_transform(df[cols])
    return df_scaled, scaler
