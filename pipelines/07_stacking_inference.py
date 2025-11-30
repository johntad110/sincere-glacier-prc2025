# infere stacking
import pandas as pd
import numpy as np
import os
import joblib

# Configuration
OUTPUT_DIR = './submissions'
STACKING_DIR = './stacking'

def stack_inference(lstm_sub_path, gbm_sub_path, output_name):
    print(f"Stacking {lstm_sub_path} and {gbm_sub_path}...")
    
    if not os.path.exists(lstm_sub_path) or not os.path.exists(gbm_sub_path):
        print("Input files not found.")
        return
        
    df_lstm = pd.read_parquet(lstm_sub_path)
    df_gbm = pd.read_parquet(gbm_sub_path)
    
    # Merge
    df = df_lstm.merge(df_gbm[['idx', 'fuel_kg']], on='idx', suffixes=('_lstm', '_gbm'))
    
    # Load Meta-Model
    model_path = os.path.join(STACKING_DIR, 'meta_model.pkl')
    
    if not os.path.exists(model_path):
        print("Meta-model not found!")
        return
        
    meta_model = joblib.load(model_path)
    
    # Prepare Input
    X = df[['fuel_kg_lstm', 'fuel_kg_gbm']].values
    
    # Predict
    preds = meta_model.predict(X)
    
    # Create Submission
    final_sub = df[['idx', 'flight_id', 'start', 'end']].copy()
    final_sub['fuel_kg'] = preds
    final_sub['fuel_kg'] = final_sub['fuel_kg'].fillna(0).clip(lower=0)
    
    out_path = os.path.join(OUTPUT_DIR, output_name)
    final_sub.to_parquet(out_path)
    print(f"Saved stacked submission to {out_path}")

def main():
    # Rank Phase
    stack_inference(
        './submissions/fuel_rank_submission.parquet',
        './submissions/submission_gbm_rank.parquet',
        'submission_stacking_rank.parquet'
    )
    
    # Final Phase
    stack_inference(
        './submissions/fuel_final_submission.parquet',
        './submissions/submission_gbm_final.parquet',
        'submission_stacking_final.parquet'
    )

if __name__ == "__main__":
    main()