# train stacking
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

LSTM_OOF = './lstm_optimized/lstm_oof.parquet'
GBM_OOF = './gbm_optimized/oof_predictions.parquet'
OUTPUT_DIR = './stacking'

def train_stacking():
    print("Loading OOFs...")
    if not os.path.exists(LSTM_OOF) or not os.path.exists(GBM_OOF):
        print("OOF files not found!")
        return
        
    df_lstm = pd.read_parquet(LSTM_OOF)
    df_gbm = pd.read_parquet(GBM_OOF)
    
    # Merge
    print("Merging OOFs...")
    # Check if idx exists in both
    if 'idx' in df_lstm.columns and 'idx' in df_gbm.columns:
        print("Merging on 'idx'...")
        df = pd.merge(df_gbm, df_lstm[['idx', 'lstm_pred']], on='idx', how='inner')
    else:
        print("Warning: 'idx' not found in one of the dataframes. Fallback to flight_id/fuel_kg merge (risky).")
        df = pd.merge(df_gbm, df_lstm, on=['flight_id', 'fuel_kg'], how='inner')
    
    print(f"Merged samples: {len(df)}")
    
    # --- Outlier Filtering ---
    print("Applying outlier filtering...")
    print(f"Samples before filtering: {len(df)}")
    
    # Calculate Flow Rate for filtering
    # Need duration from the GBM OOF data
    if 'duration' in df.columns and 'aircraft_type' in df.columns:
        duration_safe = df['duration'].replace(0, 1.0)
        flow_rate_filter = df['fuel_kg'] / duration_safe
        
        # Define Limits
        narrow_bodies = ['A320', 'A319', 'A321', 'B737', 'B738', 'B739', 'A20N', 'A21N', 'B38M', 'A318', 'B39M', 'B752']
        
        # Limits (kg/s)
        LIMIT_NARROW = 4.0 
        LIMIT_WIDE = 12.0
        
        # Create masks
        mask_narrow = (df['aircraft_type'].isin(narrow_bodies)) & (flow_rate_filter > LIMIT_NARROW)
        mask_wide = (~df['aircraft_type'].isin(narrow_bodies)) & (flow_rate_filter > LIMIT_WIDE)
        
        # Filter
        outliers = df[mask_narrow | mask_wide]
        df = df[~(mask_narrow | mask_wide)]
        
        print(f"Removed {len(outliers)} outliers.")
        print(f"Samples after filtering: {len(df)}")
    else:
        print("WARNING: 'duration' or 'aircraft_type' not found in merged data. Skipping outlier filtering.")
    
    # Features for Meta-Model
    # Simple Stacking: Just predictions
    X = df[['lstm_pred', 'oof_pred']].values
    y = df['fuel_kg'].values
    
    # Train Ridge (Linear Stacking) without scaling
    # We want coefficients to sum to ~1 and be positive.
    # fit_intercept=False forces the model to rely on the weighted sum of inputs.
    meta_model = Ridge(alpha=0.001, positive=True, fit_intercept=False)
    meta_model.fit(X, y)
    
    # Evaluate
    preds = meta_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    print(f"Stacking RMSE: {rmse:.4f}")
    print(f"Coefficients: LSTM={meta_model.coef_[0]:.4f}, GBM={meta_model.coef_[1]:.4f}")
    print(f"Intercept: {meta_model.intercept_:.4f}")
    
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    joblib.dump(meta_model, os.path.join(OUTPUT_DIR, 'meta_model.pkl'))
    # No scaler to save
    print("Saved meta-model.")

if __name__ == "__main__":
    train_stacking()