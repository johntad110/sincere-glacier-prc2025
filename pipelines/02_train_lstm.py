# train lstm
from sklearn.model_selection import GroupShuffleSplit
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import time

DATA_DIR = './prc-2025-datasets'
OUTPUT_DIR = './lstm'

SEQ_LEN = 32
N_JOBS = 4

BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 0.001
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

SEQ_FEATURES = ['altitude', 'groundspeed', 'vertical_rate',
                'sin_track', 'cos_track', 'dist_step', 'acceleration', 'energy_rate']


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


def process_flight(traj_path, fuel_df_flight, ac_idx):
    """
    Process a single flight: load trajectory and extract all intervals.
    """
    try:
        df = pd.read_parquet(traj_path)
    except Exception:
        return None

    # Fast preprocessing
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Interpolation
    cols_to_interp = ['groundspeed', 'track',
                      'vertical_rate', 'altitude', 'latitude', 'longitude']
    df[cols_to_interp] = df[cols_to_interp].interpolate(
        method='linear', limit_direction='both').fillna(0)

    # Derived features
    lats = df['latitude'].values
    lons = df['longitude'].values

    # Shifted arrays for distance
    dist_step = np.zeros(len(df))
    dist_step[1:] = haversine_distance(
        lats[:-1], lons[:-1], lats[1:], lons[1:])
    df['dist_step'] = dist_step

    tracks = np.radians(df['track'].values)
    df['sin_track'] = np.sin(tracks)
    df['cos_track'] = np.cos(tracks)

    # Physics Features
    # Constants
    g = 9.81
    kt_to_ms = 0.514444
    ft_to_m = 0.3048

    # Calculate dt (time step in seconds)
    dt = df['timestamp'].diff().dt.total_seconds().fillna(0).values
    # Avoid division by zero
    dt[dt == 0] = 1.0

    # Velocity in m/s
    v_ms = df['groundspeed'].values * kt_to_ms

    # Acceleration (dV/dt)
    acc = np.zeros(len(df))
    acc[1:] = (v_ms[1:] - v_ms[:-1]) / dt[1:]
    df['acceleration'] = acc

    # Vertical Rate in m/s
    h_dot_ms = df['vertical_rate'].values * (ft_to_m / 60.0)  # ft/min -> m/s

    # Specific Total Energy Rate: h_dot + (V/g) * acc
    # This represents the power required per unit weight to change energy state
    energy_rate = h_dot_ms + (v_ms / g) * acc
    df['energy_rate'] = energy_rate

    # Clip Physics Features to remove outliers from interpolation artifacts
    df['acceleration'] = df['acceleration'].clip(-5.0, 5.0)  # m/s^2
    df['energy_rate'] = df['energy_rate'].clip(-50.0, 50.0)

    # Extract intervals
    flight_data = []

    # Pre-convert columns to numpy for faster indexing
    timestamps = df['timestamp'].values
    seq_feats_arr = df[SEQ_FEATURES].values
    dist_step_arr = df['dist_step'].values

    flight_id_str = os.path.basename(traj_path).replace('.parquet', '')

    for _, row in fuel_df_flight.iterrows():
        start_time = row['start']
        end_time = row['end']

        # Boolean mask on numpy array is faster
        mask = (timestamps >= start_time) & (timestamps <= end_time)

        if not np.any(mask):
            continue

        # Get sequence
        seq_data = seq_feats_arr[mask]
        seq_data_resampled = resample_sequence(seq_data, SEQ_LEN)

        # Static
        duration = (end_time - start_time).total_seconds()
        total_dist = np.sum(dist_step_arr[mask])

        # [ac_idx, duration, total_dist]
        static_data = [ac_idx, duration, total_dist]

        flight_data.append({
            'X_seq': seq_data_resampled,
            'X_static': static_data,
            'y': row['fuel_kg'],
            'id': row['idx'],
            'flight_id': flight_id_str
        })

    return flight_data


def prepare_data(data_dir):
    print("Loading metadata...")
    fuel_train = pd.read_parquet(os.path.join(data_dir, 'fuel_train.parquet'))
    flightlist_train = pd.read_parquet(
        os.path.join(data_dir, 'flightlist_train.parquet'))

    # Aircraft Type Encoding
    le = LabelEncoder()
    flightlist_train['aircraft_type'] = flightlist_train['aircraft_type'].astype(
        str)
    le.fit(flightlist_train['aircraft_type'])
    ac_classes = le.classes_

    # Map flight_id -> ac_idx
    flight_ac_map = dict(zip(flightlist_train['flight_id'], le.transform(
        flightlist_train['aircraft_type'])))

    # Group fuel data by flight_id for faster access
    fuel_grouped = fuel_train.groupby('flight_id')

    traj_files = glob.glob(os.path.join(
        data_dir, 'flights_train', '*.parquet'))
    print(
        f"Found {len(traj_files)} trajectory files. Starting parallel processing with {N_JOBS} jobs...")

    start_time = time.time()

    # Parallel Processing
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_flight)(
            f,
            fuel_grouped.get_group(os.path.basename(f).replace('.parquet', '')) if os.path.basename(
                f).replace('.parquet', '') in fuel_grouped.groups else pd.DataFrame(),
            flight_ac_map.get(os.path.basename(f).replace('.parquet', ''), -1)
        )
        for f in traj_files
    )

    print(f"Data processing took {time.time() - start_time:.2f} seconds.")

    # Flatten results
    flat_results = [item for sublist in results if sublist for item in sublist]

    if not flat_results:
        raise ValueError("No data processed!")

    print("Stacking arrays...")
    X_seq = np.array([x['X_seq'] for x in flat_results], dtype=np.float32)
    X_static = np.array([x['X_static']
                        for x in flat_results], dtype=np.float32)
    y = np.array([x['y'] for x in flat_results], dtype=np.float32)
    groups = np.array([x['flight_id'] for x in flat_results])
    ids = np.array([x['id'] for x in flat_results])

    # Outlier Filtering
    print(f"Training data shape before filtering: {len(y)}")

    # Calculate Flow Rate for filtering
    # X_static: [ac_idx, duration, total_dist]
    # Duration is at index 1
    duration_safe = np.maximum(X_static[:, 1], 1.0)  # Avoid division by zero
    flow_rate_filter = y / duration_safe

    # Get aircraft types from indices
    ac_indices = X_static[:, 0].astype(int)
    aircraft_types = ac_classes[ac_indices]

    # Define Limits
    # Narrow bodies: A320 family, B737 family
    narrow_bodies = ['A320', 'A319', 'A321', 'B737', 'B738',
                     'B739', 'A20N', 'A21N', 'B38M', 'A318', 'B39M', 'B752']

    # Limits (kg/s)
    LIMIT_NARROW = 4.0
    LIMIT_WIDE = 12.0

    # Create masks
    is_narrow = np.isin(aircraft_types, narrow_bodies)
    mask_narrow = is_narrow & (flow_rate_filter > LIMIT_NARROW)
    mask_wide = (~is_narrow) & (flow_rate_filter > LIMIT_WIDE)

    # Filter
    outlier_mask = mask_narrow | mask_wide
    n_outliers = np.sum(outlier_mask)

    if n_outliers > 0:
        print(f"Removing {n_outliers} outliers based on flow rate...")
        keep_mask = ~outlier_mask

        X_seq = X_seq[keep_mask]
        X_static = X_static[keep_mask]
        y = y[keep_mask]
        groups = groups[keep_mask]
        ids = ids[keep_mask]

    print(f"Training data shape after filtering: {len(y)}")

    # Normalize
    print("Normalizing...")
    seq_mean = np.mean(X_seq, axis=(0, 1))
    seq_std = np.std(X_seq, axis=(0, 1)) + 1e-8
    X_seq = (X_seq - seq_mean) / seq_std

    static_mean = np.mean(X_static[:, 1:], axis=0)
    static_std = np.std(X_static[:, 1:], axis=0) + 1e-8
    X_static[:, 1:] = (X_static[:, 1:] - static_mean) / static_std

    stats = {
        'seq_mean': seq_mean,
        'seq_std': seq_std,
        'static_mean': static_mean,
        'static_std': static_std
    }

    # Save IDs
    np.save(os.path.join(OUTPUT_DIR, 'ids.npy'), ids)

    return X_seq, X_static, y, groups, ac_classes, stats


class FuelDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.from_numpy(X_seq)
        self.X_static = torch.from_numpy(X_static)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


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

        self.ac_embedding = nn.Embedding(
            num_classes + 1, 16)  

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
        # x_seq: [batch, seq, feat]
        self.lstm.flatten_parameters()  # For DataParallel efficiency
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = lstm_out[:, -1, :]

        ac_idx = x_static[:, 0].long()
        # Clamp index to handle unknown/padding if any
        ac_idx = torch.clamp(ac_idx, 0, self.ac_embedding.num_embeddings - 1)

        ac_emb = self.ac_embedding(ac_idx)
        other_static = x_static[:, 1:]

        combined = torch.cat((lstm_out, ac_emb, other_static), dim=1)
        return self.fc(combined).squeeze()


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Prepare Data
    # Force re-calculation if stats don't exist, or just rely on user to clear cache
    # To be safe, we'll check for stats file
    if os.path.exists(os.path.join(OUTPUT_DIR, 'X_seq.npy')) and os.path.exists(os.path.join(OUTPUT_DIR, 'stats.npz')) and os.path.exists(os.path.join(OUTPUT_DIR, 'groups.npy')):
        print("Found cached data and stats, loading...")
        X_seq = np.load(os.path.join(OUTPUT_DIR, 'X_seq.npy'))
        X_static = np.load(os.path.join(OUTPUT_DIR, 'X_static.npy'))
        y = np.load(os.path.join(OUTPUT_DIR, 'y.npy'))
        groups = np.load(os.path.join(OUTPUT_DIR, 'groups.npy'))
        ac_classes = np.load(os.path.join(
            OUTPUT_DIR, 'classes.npy'), allow_pickle=True)
    else:
        print("Processing data (Stats missing or data missing)...")
        X_seq, X_static, y, groups, ac_classes, stats = prepare_data(DATA_DIR)
        # Cache it
        np.save(os.path.join(OUTPUT_DIR, 'X_seq.npy'), X_seq)
        np.save(os.path.join(OUTPUT_DIR, 'X_static.npy'), X_static)
        np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
        np.save(os.path.join(OUTPUT_DIR, 'groups.npy'), groups)
        np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), ac_classes)
        np.savez(os.path.join(OUTPUT_DIR, 'stats.npz'), **stats)

    print(f"Total Samples: {len(y)}")

    # 2. Cross-Validation (GroupKFold)
    from sklearn.model_selection import GroupKFold

    N_FOLDS = 5
    gkf = GroupKFold(n_splits=N_FOLDS)

    print(f"Starting {N_FOLDS}-Fold Group Cross-Validation...")

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq, y, groups)):
        print(f"\n{'='*20} FOLD {fold+1}/{N_FOLDS} {'='*20}")

        X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
        X_static_train, X_static_val = X_static[train_idx], X_static[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train samples: {len(y_train)}, Val samples: {len(y_val)}")

        # Datasets & Loaders
        train_ds = FuelDataset(X_seq_train, X_static_train, y_train)
        val_ds = FuelDataset(X_seq_val, X_static_val, y_val)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        # Model Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FuelLSTM(
            seq_input_dim=X_seq.shape[2],
            static_input_dim=X_static.shape[1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=len(ac_classes)
        ).to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scaler = GradScaler()

        best_rmse = float('inf')

        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0

            for x_s, x_st, target in train_loader:
                x_s, x_st, target = x_s.to(device, non_blocking=True), x_st.to(
                    device, non_blocking=True), target.to(device, non_blocking=True)

                optimizer.zero_grad()
                with autocast():
                    output = model(x_s, x_st)
                    loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            train_rmse = np.sqrt(train_loss / len(train_loader))

            # Validation
            model.eval()
            val_mse = 0
            with torch.no_grad():
                for x_s, x_st, target in val_loader:
                    x_s, x_st, target = x_s.to(device, non_blocking=True), x_st.to(
                        device, non_blocking=True), target.to(device, non_blocking=True)
                    output = model(x_s, x_st)
                    val_mse += criterion(output, target).item()

            val_rmse = np.sqrt(val_mse / len(val_loader))
            scheduler.step(val_rmse)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f}")

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(
                    OUTPUT_DIR, f'model_fold_{fold}.pth'))

        print(f"Fold {fold+1} Best RMSE: {best_rmse:.4f}")
        fold_scores.append(best_rmse)

    print(
        f"\nOverall CV RMSE: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print("Training complete.")


if __name__ == "__main__":
    main()
