import pandas as pd
import numpy as np
import glob
import os
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import time

DATA_DIR = './prc-2025-datasets'
OUTPUT_DIR = './gbm'

N_JOBS = 4

FEATURES = [
    'duration', 'n_points',
    'avg_alt', 'std_alt', 'min_alt', 'max_alt',
    'avg_speed', 'std_speed',
    'total_dist', 'avg_vertical_rate',
    'avg_acc', 'std_acc', 'min_acc', 'max_acc',
    'avg_energy_rate', 'std_energy_rate', 'min_energy_rate', 'max_energy_rate',
    'aircraft_type',
    'time_since_takeoff', 'time_to_landing', 'relative_time', 'od_distance',
    'origin_lat', 'origin_lon', 'origin_elev',
    'dest_lat', 'dest_lon', 'dest_elev',
    'avg_mach', 'avg_dynamic_pressure', 'avg_air_density',
    'avg_parasitic_power', 'avg_induced_power', 'avg_climb_power',
    'mass_proxy'
]
RARE_AC_MAP = {
    'B748': 'B744',  # 747-8 -> 747-400
    'B77L': 'B77W',  # 777-200LR -> 777-300ER
    'A388': 'B744',  # A380 -> 747-400 (Best proxy for heavy 4-engine)
    'B39M': 'B38M',  # 737 Max 9 -> Max 8
    'A318': 'A319',  # A318 -> A319
    'MD11': 'B772',  # MD11 -> 777-200 (Approximate heavy)
}


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * \
        np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def get_atmosphere_properties(altitude_ft):
    """
    ISA Standard Atmosphere Approximation
    Returns: air_density (kg/m^3), speed_of_sound (m/s)
    """
    # Constants
    h_m = altitude_ft * 0.3048

    # Troposphere (up to 11km)
    T0 = 288.15  # Sea level temp (K)
    P0 = 101325  # Sea level pressure (Pa)
    L = 0.0065  # Lapse rate (K/m)
    R = 287.05  # Gas constant
    g = 9.80665

    # Temperature
    T = np.maximum(T0 - L * h_m, 216.65)  # Cap at stratosphere temp

    # Pressure (Simplified for Troposphere)
    P = P0 * (1 - L * h_m / T0) ** (g / (R * L))

    # Density: rho = P / (R * T)
    rho = P / (R * T)

    # Speed of Sound: a = sqrt(gamma * R * T), gamma = 1.4
    a = np.sqrt(1.4 * R * T)

    return rho, a


def process_flight_features(traj_path, fuel_df_flight, ac_type):
    """
    Process a single flight: load trajectory, clean, and extract features for all intervals.
    """
    try:
        df = pd.read_parquet(traj_path)
    except Exception:
        return []

    # Fast preprocessing
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Interpolation
    cols_to_interp = ['groundspeed', 'track',
                      'vertical_rate', 'altitude', 'latitude', 'longitude']
    df[cols_to_interp] = df[cols_to_interp].interpolate(
        method='linear', limit_direction='both').fillna(0)

    # Derived features
    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

    # Vectorized distance
    lats = df['latitude'].values
    lons = df['longitude'].values
    dist_step = np.zeros(len(df))
    dist_step[1:] = haversine_distance(
        lats[:-1], lons[:-1], lats[1:], lons[1:])
    df['dist_step'] = dist_step

    # Physics Features
    g = 9.81
    kt_to_ms = 0.514444
    ft_to_m = 0.3048

    dt = df['dt'].values
    dt[dt == 0] = 1.0

    v_ms = df['groundspeed'].values * kt_to_ms

    acc = np.zeros(len(df))
    acc[1:] = (v_ms[1:] - v_ms[:-1]) / dt[1:]
    df['acceleration'] = acc.clip(-5.0, 5.0)

    h_dot_ms = df['vertical_rate'].values * (ft_to_m / 60.0)
    energy_rate = h_dot_ms + (v_ms / g) * acc
    df['energy_rate'] = energy_rate.clip(-50.0, 50.0)

    # Aerodynamics
    rho, a_sound = get_atmosphere_properties(df['altitude'].values)

    # Dynamic Pressure: q = 0.5 * rho * v^2
    df['dynamic_pressure'] = 0.5 * rho * (v_ms ** 2)

    # Mach Number
    df['mach'] = v_ms / a_sound

    # Power Proxies
    # Parasitic Power ~ rho * v^3
    df['parasitic_power'] = rho * (v_ms ** 3)

    # Induced Power ~ 1/v (simplified)
    v_safe = np.maximum(v_ms, 1.0)
    df['induced_power'] = 1.0 / v_safe

    # Climb Power ~ vertical_speed
    df['climb_power'] = h_dot_ms

    # Extract intervals
    features = []

    timestamps = df['timestamp'].values
    altitudes = df['altitude'].values
    speeds = df['groundspeed'].values
    dists = df['dist_step'].values
    v_rates = df['vertical_rate'].values
    accs = df['acceleration'].values
    e_rates = df['energy_rate'].values

    # Aero arrays
    machs = df['mach'].values
    qs = df['dynamic_pressure'].values
    rhos = rho  # already array
    p_paras = df['parasitic_power'].values
    p_inds = df['induced_power'].values
    p_climbs = df['climb_power'].values

    for _, row in fuel_df_flight.iterrows():
        start_time = row['start']
        end_time = row['end']

        # Calculate mid-point timestamp for this interval
        mid_ts = start_time + (end_time - start_time) / 2

        mask = (timestamps >= start_time) & (timestamps <= end_time)

        if not np.any(mask):
            feat = {
                'idx': row['idx'],
                'flight_id': row['flight_id'],
                'timestamp_mid': mid_ts,
                'duration': (end_time - start_time).total_seconds(),
                'n_points': 0,
                'avg_alt': np.nan, 'std_alt': np.nan, 'min_alt': np.nan, 'max_alt': np.nan,
                'avg_speed': np.nan, 'std_speed': np.nan,
                'total_dist': np.nan, 'avg_vertical_rate': np.nan,
                'avg_acc': np.nan, 'std_acc': np.nan, 'min_acc': np.nan, 'max_acc': np.nan,
                'avg_energy_rate': np.nan, 'std_energy_rate': np.nan, 'min_energy_rate': np.nan, 'max_energy_rate': np.nan,
                # Aero
                'avg_mach': np.nan, 'avg_dynamic_pressure': np.nan, 'avg_air_density': np.nan,
                'avg_parasitic_power': np.nan, 'avg_induced_power': np.nan, 'avg_climb_power': np.nan,
                'fuel_kg': row['fuel_kg'],
                'aircraft_type': ac_type
            }
        else:
            # Slice arrays
            int_alt = altitudes[mask]
            int_speed = speeds[mask]
            int_dist = dists[mask]
            int_vrate = v_rates[mask]
            int_acc = accs[mask]
            int_erate = e_rates[mask]

            # Aero slices
            int_mach = machs[mask]
            int_q = qs[mask]
            int_rho = rhos[mask]
            int_pp = p_paras[mask]
            int_pi = p_inds[mask]
            int_pc = p_climbs[mask]

            feat = {
                'idx': row['idx'],
                'flight_id': row['flight_id'],
                'timestamp_mid': mid_ts,
                'duration': (end_time - start_time).total_seconds(),
                'n_points': np.sum(mask),
                'avg_alt': np.mean(int_alt),
                'std_alt': np.std(int_alt),
                'min_alt': np.min(int_alt),
                'max_alt': np.max(int_alt),
                'avg_speed': np.mean(int_speed),
                'std_speed': np.std(int_speed),
                'total_dist': np.sum(int_dist),
                'avg_vertical_rate': np.mean(int_vrate),
                'avg_acc': np.mean(int_acc),
                'std_acc': np.std(int_acc),
                'min_acc': np.min(int_acc),
                'max_acc': np.max(int_acc),
                'avg_energy_rate': np.mean(int_erate),
                'std_energy_rate': np.std(int_erate),
                'min_energy_rate': np.min(int_erate),
                'max_energy_rate': np.max(int_erate),
                # Aero
                'avg_mach': np.mean(int_mach),
                'avg_dynamic_pressure': np.mean(int_q),
                'avg_air_density': np.mean(int_rho),
                'avg_parasitic_power': np.mean(int_pp),
                'avg_induced_power': np.mean(int_pi),
                'avg_climb_power': np.mean(int_pc),

                'fuel_kg': row['fuel_kg'],
                'aircraft_type': ac_type
            }
        features.append(feat)

    return features


def prepare_data(data_dir):
    print("Loading metadata...")
    fuel_train = pd.read_parquet(os.path.join(data_dir, 'fuel_train.parquet'))
    flightlist_train = pd.read_parquet(
        os.path.join(data_dir, 'flightlist_train.parquet'))
    apt_df = pd.read_parquet(os.path.join(data_dir, 'apt.parquet'))

    # Map flight_id to aircraft_type
    # Apply Rare Aircraft Mapping
    flightlist_train['aircraft_type'] = flightlist_train['aircraft_type'].replace(
        RARE_AC_MAP)
    ac_type_map = flightlist_train.set_index(
        'flight_id')['aircraft_type'].to_dict()

    # Calculate O-D Distance
    print("Calculating O-D Distances...")
    # Merge origin coords
    flightlist_train = flightlist_train.merge(
        apt_df[['icao', 'latitude', 'longitude', 'elevation']].rename(columns={
            'latitude': 'origin_lat',
            'longitude': 'origin_lon',
            'elevation': 'origin_elev'
        }),
        left_on='origin_icao', right_on='icao', how='left'
    )
    # Merge destination coords
    flightlist_train = flightlist_train.merge(
        apt_df[['icao', 'latitude', 'longitude', 'elevation']].rename(columns={
            'latitude': 'dest_lat',
            'longitude': 'dest_lon',
            'elevation': 'dest_elev'
        }),
        left_on='destination_icao', right_on='icao', how='left'
    )

    # Calculate distance
    flightlist_train['od_distance'] = haversine_distance(
        flightlist_train['origin_lat'], flightlist_train['origin_lon'],
        flightlist_train['dest_lat'], flightlist_train['dest_lon']
    )
    # Fill missing distances with mean
    flightlist_train['od_distance'] = flightlist_train['od_distance'].fillna(
        flightlist_train['od_distance'].mean())

    # Group fuel data
    fuel_grouped = fuel_train.groupby('flight_id')

    traj_files = glob.glob(os.path.join(
        data_dir, 'flights_train', '*.parquet'))
    print(
        f"Found {len(traj_files)} trajectory files. Starting parallel processing with {N_JOBS} jobs...")

    start_time = time.time()

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_flight_features)(
            f,
            fuel_grouped.get_group(os.path.basename(f).replace('.parquet', '')) if os.path.basename(
                f).replace('.parquet', '') in fuel_grouped.groups else pd.DataFrame(),
            ac_type_map.get(os.path.basename(
                f).replace('.parquet', ''), 'Unknown')
        )
        for f in traj_files
    )

    print(f"Data processing took {time.time() - start_time:.2f} seconds.")

    # Flatten
    flat_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flat_results)

    # Merge Global Context Features
    print("Merging global context features...")
    df = df.merge(
        flightlist_train[[
            'flight_id', 'takeoff', 'landed', 'od_distance',
            'origin_lat', 'origin_lon', 'origin_elev',
            'dest_lat', 'dest_lon', 'dest_elev'
        ]],
        on='flight_id', how='left'
    )

    # Merge Mass Proxy
    mass_proxy_path = os.path.join(data_dir, 'mass_proxy.parquet')
    if os.path.exists(mass_proxy_path):
        print("Merging Mass Proxy...")
        mass_df = pd.read_parquet(mass_proxy_path)
        df = df.merge(mass_df, on='flight_id', how='left')
        df['mass_proxy'] = df['mass_proxy'].fillna(
            0)  # Fill missing with 0 (average weight)
    else:
        print("WARNING: Mass Proxy file not found! Filling with 0.")
        df['mass_proxy'] = 0

    # Calculate Time Features
    df['time_since_takeoff'] = (
        df['timestamp_mid'] - df['takeoff']).dt.total_seconds()
    df['time_to_landing'] = (
        df['landed'] - df['timestamp_mid']).dt.total_seconds()
    df['flight_duration_total'] = (
        df['landed'] - df['takeoff']).dt.total_seconds()
    df['relative_time'] = df['time_since_takeoff'] / df['flight_duration_total']

    # Handle potential NaNs/Infs in time features
    df['relative_time'] = df['relative_time'].clip(0, 1).fillna(0.5)
    df['time_since_takeoff'] = df['time_since_takeoff'].fillna(0)
    df['time_to_landing'] = df['time_to_landing'].fillna(0)

    # Imputation Logic
    # Calculate means per aircraft type for numeric features
    numeric_cols = [c for c in FEATURES if c != 'aircraft_type']
    ac_means = df.groupby('aircraft_type')[numeric_cols].transform('mean')

    # Fill NaNs with aircraft type means
    df[numeric_cols] = df[numeric_cols].fillna(ac_means)

    # If any NaNs remain (e.g., if an aircraft type has NO valid data at all), fill with global mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df = df.fillna(0)  # Final fallback

    # Save means for inference
    means_df = df.groupby('aircraft_type')[numeric_cols].mean().reset_index()
    means_path = os.path.join(OUTPUT_DIR, 'ac_means.parquet')
    means_df.to_parquet(means_path)
    print(f"Saved aircraft means to {means_path}")

    return df


def train_gbm(df, output_dir):
    print("Preparing for training...")
    target = 'fuel_kg'

    # Categorical
    df['aircraft_type'] = df['aircraft_type'].astype('category')

    # Drop NaNs in target
    df = df.dropna(subset=[target])

    # Outlier Filtering
    print(f"Training data shape before filtering: {df.shape}")

    # Calculate Flow Rate for filtering
    # Avoid division by zero
    df['duration_safe'] = df['duration'].replace(0, 1.0)
    df['flow_rate_filter'] = df['fuel_kg'] / df['duration_safe']

    # Define Limits
    # Narrow bodies: A320 family, B737 family
    narrow_bodies = ['A320', 'A319', 'A321', 'B737', 'B738',
                     'B739', 'A20N', 'A21N', 'B38M', 'A318', 'B39M', 'B752']

    # Limits (kg/s)
    LIMIT_NARROW = 4.0
    LIMIT_WIDE = 12.0

    # Create masks
    mask_narrow = (df['aircraft_type'].isin(narrow_bodies)) & (
        df['flow_rate_filter'] > LIMIT_NARROW)
    mask_wide = (~df['aircraft_type'].isin(narrow_bodies)) & (
        df['flow_rate_filter'] > LIMIT_WIDE)

    boundary_mask = (df['relative_time'] < 0.05) | (df['relative_time'] > 0.95)

    # Filter
    outliers = df[mask_narrow | mask_wide | boundary_mask]
    df = df[~(mask_narrow | mask_wide | boundary_mask)]

    print(f"Removed {len(outliers)} outliers.")
    print(f"Training data shape after filtering: {df.shape}")

    # CV
    gkf = GroupKFold(n_splits=5)
    groups = df['flight_id']

    # OOF preds will be in FUEL KG
    oof_preds = np.zeros(len(df))
    models = []
    scores = []

    print("Starting LightGBM Training (Target: Total Fuel kg)...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[target], groups)):
        print(f"\nFold {fold+1}")

        X_train, y_train = df.iloc[train_idx][FEATURES], df.iloc[train_idx][target]
        X_val, y_val = df.iloc[val_idx][FEATURES], df.iloc[val_idx][target]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1
        }

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )

        models.append(model)

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        scores.append(rmse)
        print(f"Fold {fold+1} RMSE (kg): {rmse:.4f}")

        # Save model
        model.save_model(os.path.join(
            output_dir, f'lgb_model_fold_{fold}.txt'))

    # Overall RMSE in kg
    overall_rmse = np.sqrt(mean_squared_error(df[target], oof_preds))

    print(f"\nOverall CV RMSE (kg): {overall_rmse:.4f}")
    print(f"Average Fold RMSE (kg): {np.mean(scores):.4f}")

    # Save OOF
    df['oof_pred'] = oof_preds
    df.to_parquet(os.path.join(output_dir, 'oof_predictions.parquet'))


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    feature_path = os.path.join(OUTPUT_DIR, 'train_features.parquet')
    if os.path.exists(feature_path):
        print("Removing old cached features to force re-computation...")
        os.remove(feature_path)
    df = prepare_data(DATA_DIR)
    print(f"Saving features to {feature_path}")
    df.to_parquet(feature_path)
    train_gbm(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
