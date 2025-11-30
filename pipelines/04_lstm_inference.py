# infere lstm
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from tqdm import tqdm

DATA_DIR = './prc-2025-datasets'
MODEL_DIR = './lstm_optimized'
OUTPUT_DIR = './submissions'
SEQ_LEN = 32
BATCH_SIZE = 512
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
SEQ_FEATURES = ['altitude', 'groundspeed', 'vertical_rate',
                'sin_track', 'cos_track', 'dist_step', 'acceleration', 'energy_rate']


class FuelLSTM(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, hidden_dim, num_layers, num_classes):
        super(FuelLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT
        )

        self.ac_embedding = nn.Embedding(num_classes + 1, 16)

        combined_dim = hidden_dim + 16 + (static_input_dim - 1)

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_static):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out[:, -1, :]

        ac_idx = x_static[:, 0].long()
        ac_idx = torch.clamp(ac_idx, 0, self.ac_embedding.num_embeddings - 1)

        ac_emb = self.ac_embedding(ac_idx)
        other_static = x_static[:, 1:]

        combined = torch.cat((lstm_out, ac_emb, other_static), dim=1)
        return self.fc(combined).squeeze()


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * \
        np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def resample_sequence(seq_data, target_len):
    if len(seq_data) == 0:
        return np.zeros((target_len, seq_data.shape[1]))
    if len(seq_data) == target_len:
        return seq_data
    x_old = np.linspace(0, 1, len(seq_data))
    x_new = np.linspace(0, 1, target_len)
    new_seq = np.zeros((target_len, seq_data.shape[1]))
    for i in range(seq_data.shape[1]):
        new_seq[:, i] = np.interp(x_new, x_old, seq_data[:, i])
    return new_seq


def process_flight_inference(flight_id, fuel_df_flight, traj_dirs, ac_idx):
    # Find trajectory file
    traj_path = None
    for d in traj_dirs:
        p = os.path.join(d, f"{flight_id}.parquet")
        if os.path.exists(p):
            traj_path = p
            break

    if traj_path is None:
        # Should not happen for valid submission files
        return []

    try:
        df = pd.read_parquet(traj_path)
    except:
        return []

    df = df.sort_values('timestamp').reset_index(drop=True)
    cols_to_interp = ['groundspeed', 'track',
                      'vertical_rate', 'altitude', 'latitude', 'longitude']
    df[cols_to_interp] = df[cols_to_interp].interpolate(
        method='linear', limit_direction='both').fillna(0)

    lats = df['latitude'].values
    lons = df['longitude'].values
    dist_step = np.zeros(len(df))
    dist_step[1:] = haversine_distance(
        lats[:-1], lons[:-1], lats[1:], lons[1:])
    df['dist_step'] = dist_step

    tracks = np.radians(df['track'].values)
    df['sin_track'] = np.sin(tracks)
    df['cos_track'] = np.cos(tracks)

    # Physics Features
    g = 9.81
    kt_to_ms = 0.514444
    ft_to_m = 0.3048

    dt = df['timestamp'].diff().dt.total_seconds().fillna(0).values
    dt[dt == 0] = 1.0

    v_ms = df['groundspeed'].values * kt_to_ms

    acc = np.zeros(len(df))
    acc[1:] = (v_ms[1:] - v_ms[:-1]) / dt[1:]
    df['acceleration'] = acc

    h_dot_ms = df['vertical_rate'].values * (ft_to_m / 60.0)

    energy_rate = h_dot_ms + (v_ms / g) * acc
    df['energy_rate'] = energy_rate

    # Clip Physics Features
    df['acceleration'] = df['acceleration'].clip(-5.0, 5.0)
    df['energy_rate'] = df['energy_rate'].clip(-50.0, 50.0)

    timestamps = df['timestamp'].values
    seq_feats_arr = df[SEQ_FEATURES].values
    dist_step_arr = df['dist_step'].values

    results = []
    for _, row in fuel_df_flight.iterrows():
        start_time = row['start']
        end_time = row['end']

        mask = (timestamps >= start_time) & (timestamps <= end_time)

        if not np.any(mask):
            # Fallback for empty intervals: zeros
            seq_data_resampled = np.zeros((SEQ_LEN, len(SEQ_FEATURES)))
            duration = (end_time - start_time).total_seconds()
            total_dist = 0
        else:
            seq_data = seq_feats_arr[mask]
            seq_data_resampled = resample_sequence(seq_data, SEQ_LEN)
            duration = (end_time - start_time).total_seconds()
            total_dist = np.sum(dist_step_arr[mask])

        static_data = [ac_idx, duration, total_dist]

        results.append({
            'X_seq': seq_data_resampled,
            'X_static': static_data,
            'idx': row['idx']
        })

    return results


def generate_submission(submission_file, traj_dirs, models, device, stats, ac_map, output_name):
    print(f"Processing {submission_file}...")
    sub_df = pd.read_parquet(submission_file)

    # Make a copy of the original submission to preserve all rows and order
    final_sub = sub_df.copy()

    # Group by flight_id for efficient processing
    sub_grouped = sub_df.groupby('flight_id')
    flight_ids = list(sub_grouped.groups.keys())

    print(f"Processing {len(flight_ids)} flights...")

    # Parallel processing
    results = Parallel(n_jobs=4)(
        delayed(process_flight_inference)(
            fid,
            sub_grouped.get_group(fid),
            traj_dirs,
            ac_map.get(fid, -1)
        )
        for fid in tqdm(flight_ids)
    )

    # Flatten results and collect all data
    all_results = []
    for flight_results in results:
        all_results.extend(flight_results)

    # Sort by idx to maintain order
    all_results.sort(key=lambda x: x['idx'])

    # Create batched arrays
    X_seq = np.array([r['X_seq'] for r in all_results], dtype=np.float32)
    X_static = np.array([r['X_static'] for r in all_results], dtype=np.float32)
    idxs = [r['idx'] for r in all_results]

    # Normalize
    X_seq = (X_seq - stats['seq_mean']) / stats['seq_std']
    X_static[:, 1:] = (X_static[:, 1:] - stats['static_mean']
                       ) / stats['static_std']

    # Create dataset and dataloader for batch processing
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_seq),
        torch.from_numpy(X_static)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Ensemble Prediction
    all_predictions = []

    for model in models:
        model.eval()
        model_predictions = []
        with torch.no_grad():
            for x_seq_batch, x_static_batch in tqdm(loader, desc="Predicting"):
                x_seq_batch = x_seq_batch.to(device)
                x_static_batch = x_static_batch.to(device)

                # Remove DataParallel wrapper if present
                if hasattr(model, 'module'):
                    outputs = model.module(x_seq_batch, x_static_batch)
                else:
                    outputs = model(x_seq_batch, x_static_batch)

                model_predictions.extend(outputs.cpu().numpy())

        all_predictions.append(model_predictions)

    # Average predictions across models
    avg_predictions = np.mean(all_predictions, axis=0)

    # Create idx to prediction mapping
    idx_to_pred = dict(zip(idxs, avg_predictions))

    # Assign predictions back to original dataframe
    final_sub['fuel_kg'] = final_sub['idx'].map(idx_to_pred)

    # Fill any missing predictions with 0 and ensure non-negative
    final_sub['fuel_kg'] = final_sub['fuel_kg'].fillna(0).clip(lower=0)

    # Verify we have all rows
    print(f"Original rows: {len(sub_df)}")
    print(f"Final rows: {len(final_sub)}")
    print(f"Rows with predictions: {final_sub['fuel_kg'].notna().sum()}")

    if len(final_sub) != len(sub_df):
        raise ValueError(
            f"Row count mismatch! Original: {len(sub_df)}, Final: {len(final_sub)}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    out_path = os.path.join(OUTPUT_DIR, output_name)
    final_sub.to_parquet(out_path)
    print(f"Saved submission to {out_path}")
    return final_sub


def main():
    # 1. Load Stats and Metadata from Training
    print("Loading training stats...")
    X_seq_train = np.load(os.path.join(MODEL_DIR, 'X_seq.npy'))
    X_static_train = np.load(os.path.join(MODEL_DIR, 'X_static.npy'))
    classes = np.load(os.path.join(
        MODEL_DIR, 'classes.npy'), allow_pickle=True)

    # Load stats
    stats_path = os.path.join(MODEL_DIR, 'stats.npz')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Stats file not found at {stats_path}. Please re-run the training pipeline.")

    loaded_stats = np.load(stats_path)
    stats = {
        'seq_mean': loaded_stats['seq_mean'],
        'seq_std': loaded_stats['seq_std'],
        'static_mean': loaded_stats['static_mean'],
        'static_std': loaded_stats['static_std']
    }

    # Re-create AC Map
    # We need flightlist to map flight_id to aircraft_type
    # We'll load all flightlists we can find
    fl_train = pd.read_parquet(os.path.join(
        DATA_DIR, 'flightlist_train.parquet'))
    fl_rank = pd.read_parquet(os.path.join(
        DATA_DIR, 'flightlist_rank.parquet'))
    # fl_final might exist
    fl_final_path = os.path.join(DATA_DIR, 'flightlist_final.parquet')
    if os.path.exists(fl_final_path):
        fl_final = pd.read_parquet(fl_final_path)
        fl_all = pd.concat([fl_train, fl_rank, fl_final])
    else:
        fl_all = pd.concat([fl_train, fl_rank])

    le = LabelEncoder()
    le.classes_ = classes

    # Filter for known classes, assign -1 to unknown
    fl_all['ac_idx'] = -1
    known_mask = fl_all['aircraft_type'].isin(classes)
    fl_all.loc[known_mask, 'ac_idx'] = le.transform(
        fl_all.loc[known_mask, 'aircraft_type'])

    ac_map = dict(zip(fl_all['flight_id'], fl_all['ac_idx']))

    # 2. Load Models (Ensemble)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = []
    model_files = glob.glob(os.path.join(MODEL_DIR, 'model_fold_*.pth'))

    if not model_files:
        # Fallback to single model if no folds found
        if os.path.exists(os.path.join(MODEL_DIR, 'best_lstm_model.pth')):
            model_files = [os.path.join(MODEL_DIR, 'best_lstm_model.pth')]
        else:
            raise FileNotFoundError("No model files found!")

    print(f"Found {len(model_files)} models for ensemble.")

    for model_path in model_files:
        model = FuelLSTM(
            seq_input_dim=X_seq_train.shape[2],
            static_input_dim=X_static_train.shape[1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=len(classes)
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {
                k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        model.eval()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        models.append(model)

    # 3. Generate Submissions
    traj_dirs = [
        os.path.join(DATA_DIR, 'flights_train'),
        os.path.join(DATA_DIR, 'flights_rank'),
        os.path.join(DATA_DIR, 'flights_final')
    ]

    # Phase 1
    generate_submission(
        os.path.join(DATA_DIR, 'fuel_rank_submission.parquet'),
        traj_dirs,
        models,  # Pass list of models
        device,
        stats,
        ac_map,
        'fuel_rank_submission.parquet'
    )

    # Phase 2
    if os.path.exists(os.path.join(DATA_DIR, 'fuel_final_submission.parquet')):
        generate_submission(
            os.path.join(DATA_DIR, 'fuel_final_submission.parquet'),
            traj_dirs,
            models,
            device,
            stats,
            ac_map,
            'fuel_final_submission.parquet'
        )


if __name__ == "__main__":
    main()
