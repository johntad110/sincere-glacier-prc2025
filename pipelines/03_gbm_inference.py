import pandas as pd
import numpy as np
import glob
import os
import lightgbm as lgb
from tqdm import tqdm
from joblib import Parallel, delayed

DATA_DIR = './prc-2025-datasets'
MODEL_DIR = './gbm_optimized'
OUTPUT_DIR = './submissions'
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


def process_trajectory(traj_path):
    try:
        df = pd.read_parquet(traj_path)
    except:
        return None

    df = df.sort_values('timestamp').reset_index(drop=True)
    cols_to_interp = ['groundspeed', 'track',
                      'vertical_rate', 'altitude', 'latitude', 'longitude']
    df[cols_to_interp] = df[cols_to_interp].interpolate(
        method='linear', limit_direction='both')

    df['dt'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    # Vectorized distance
    lats = df['latitude'].values
    lons = df['longitude'].values
    dist_step = np.zeros(len(df))
    if len(df) > 1:
        dist_step[1:] = haversine_distance(
            lats[:-1], lons[:-1], lats[1:], lons[1:])
    df['dist_step'] = dist_step

    # NOTE: Removed calc_speed imputation to match training logic
    df = df.fillna(0)

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

    # --- Aerodynamics ---
    rho, a_sound = get_atmosphere_properties(df['altitude'].values)

    # Dynamic Pressure: q = 0.5 * rho * v^2
    q = 0.5 * rho * (v_ms ** 2)
    df['dynamic_pressure'] = q

    # Mach Number
    df['mach'] = v_ms / a_sound

    # Power Proxies
    # Parasitic Power ~ rho * v^3
    df['parasitic_power'] = rho * (v_ms ** 3)

    # Induced Power ~ 1/v
    v_safe = np.maximum(v_ms, 1.0)
    df['induced_power'] = 1.0 / v_safe

    # Climb Power ~ vertical_speed
    df['climb_power'] = h_dot_ms

    return df


def extract_features_for_flight(flight_id, sub_df_flight, traj_dirs, ac_type):
    # Find trajectory
    traj_path = None
    for d in traj_dirs:
        p = os.path.join(d, f"{flight_id}.parquet")
        if os.path.exists(p):
            traj_path = p
            break

    if traj_path is None:
        return []

    traj_df = process_trajectory(traj_path)
    if traj_df is None:
        return []

    features = []

    # Pre-calculate timestamps for faster lookup
    traj_timestamps = traj_df['timestamp'].values

    for _, row in sub_df_flight.iterrows():
        start_time = row['start']
        end_time = row['end']

        # Calculate mid-point timestamp for this interval
        mid_ts = start_time + (end_time - start_time) / 2

        mask = (traj_timestamps >= start_time) & (traj_timestamps <= end_time)
        interval_traj = traj_df[mask]

        if interval_traj.empty:
            feat = {
                'idx': row['idx'],
                'flight_id': flight_id,  # Ensure flight_id is present
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
                'aircraft_type': ac_type
            }
        else:
            feat = {
                'idx': row['idx'],
                'flight_id': flight_id,  # Ensure flight_id is present
                'timestamp_mid': mid_ts,
                'duration': (end_time - start_time).total_seconds(),
                'n_points': len(interval_traj),
                'avg_alt': interval_traj['altitude'].mean(),
                'std_alt': interval_traj['altitude'].std(),
                'min_alt': interval_traj['altitude'].min(),
                'max_alt': interval_traj['altitude'].max(),
                'avg_speed': interval_traj['groundspeed'].mean(),
                'std_speed': interval_traj['groundspeed'].std(),
                'total_dist': interval_traj['dist_step'].sum(),
                'avg_vertical_rate': interval_traj['vertical_rate'].mean(),
                'avg_acc': interval_traj['acceleration'].mean(),
                'std_acc': interval_traj['acceleration'].std(),
                'min_acc': interval_traj['acceleration'].min(),
                'max_acc': interval_traj['acceleration'].max(),
                'avg_energy_rate': interval_traj['energy_rate'].mean(),
                'std_energy_rate': interval_traj['energy_rate'].std(),
                'min_energy_rate': interval_traj['energy_rate'].min(),
                'max_energy_rate': interval_traj['energy_rate'].max(),
                # Aero
                'avg_mach': interval_traj['mach'].mean(),
                'avg_dynamic_pressure': interval_traj['dynamic_pressure'].mean(),
                # Approx, better to use array
                'avg_air_density': interval_traj['altitude'].apply(lambda x: get_atmosphere_properties(x)[0]).mean(),
                'avg_parasitic_power': interval_traj['parasitic_power'].mean(),
                'avg_induced_power': interval_traj['induced_power'].mean(),
                'avg_climb_power': interval_traj['climb_power'].mean(),

                'aircraft_type': ac_type
            }
        features.append(feat)
    return features


# Mapping for rare/heavy aircraft to common equivalents
RARE_AC_MAP = {
    'B748': 'B744',
    'B77L': 'B77W',
    'A388': 'B744',
    'B39M': 'B38M',
    'A318': 'A319',
    'MD11': 'B772',
}


def generate_gbm_submission(submission_file, traj_dirs, models, ac_map, ac_means_df, fl_all, output_name, save_features_path=None):
    print(f"Processing {submission_file}...")
    sub_df = pd.read_parquet(submission_file)

    if sub_df['idx'].duplicated().any():
        print(f"Warning: Duplicates found in {submission_file}. Dropping...")
        sub_df = sub_df.drop_duplicates(subset=['idx'])

    sub_grouped = sub_df.groupby('flight_id')
    flight_ids = list(sub_grouped.groups.keys())

    print("Extracting features...")
    results = Parallel(n_jobs=4)(
        delayed(extract_features_for_flight)(
            fid,
            sub_grouped.get_group(fid),
            traj_dirs,
            ac_map.get(fid, 'Unknown')
        )
        for fid in tqdm(flight_ids)
    )

    flat_results = [item for sublist in results for item in sublist]
    feature_df = pd.DataFrame(flat_results)

    # Fix for duplicates: Ensure unique idx before reindexing
    feature_df = feature_df.drop_duplicates(subset=['idx'])

    # Ensure correct order
    feature_df = feature_df.set_index('idx').loc[sub_df['idx']].reset_index()

    # Merge Global Context Features
    print("Merging global context features...")
    feature_df = feature_df.merge(
        fl_all[[
            'flight_id', 'takeoff', 'landed', 'od_distance',
            'origin_lat', 'origin_lon', 'origin_elev',
            'dest_lat', 'dest_lon', 'dest_elev'
        ]],
        on='flight_id', how='left'
    )

    # Merge Mass Proxy
    mass_proxy_path = os.path.join(DATA_DIR, 'mass_proxy.parquet')
    if os.path.exists(mass_proxy_path):
        print("Merging Mass Proxy...")
        mass_df = pd.read_parquet(mass_proxy_path)
        feature_df = feature_df.merge(mass_df, on='flight_id', how='left')
        feature_df['mass_proxy'] = feature_df['mass_proxy'].fillna(0)
    else:
        print("WARNING: Mass Proxy file not found! Filling with 0.")
        feature_df['mass_proxy'] = 0

    # Calculate Time Features
    feature_df['time_since_takeoff'] = (
        feature_df['timestamp_mid'] - feature_df['takeoff']).dt.total_seconds()
    feature_df['time_to_landing'] = (
        feature_df['landed'] - feature_df['timestamp_mid']).dt.total_seconds()
    feature_df['flight_duration_total'] = (
        feature_df['landed'] - feature_df['takeoff']).dt.total_seconds()
    feature_df['relative_time'] = feature_df['time_since_takeoff'] / \
        feature_df['flight_duration_total']

    # Handle potential NaNs/Infs in time features
    feature_df['relative_time'] = feature_df['relative_time'].clip(
        0, 1).fillna(0.5)
    feature_df['time_since_takeoff'] = feature_df['time_since_takeoff'].fillna(
        0)
    feature_df['time_to_landing'] = feature_df['time_to_landing'].fillna(0)
    feature_df['od_distance'] = feature_df['od_distance'].fillna(
        feature_df['od_distance'].mean())

    # Imputation
    print("Imputing missing features...")

    # Intensive properties (can be imputed with mean)
    intensive_cols = [
        'avg_alt', 'std_alt', 'min_alt', 'max_alt',
        'avg_speed', 'std_speed',
        'avg_vertical_rate',
        'avg_acc', 'std_acc', 'min_acc', 'max_acc',
        'avg_energy_rate', 'std_energy_rate', 'min_energy_rate', 'max_energy_rate',
        # Aero
        'avg_mach', 'avg_dynamic_pressure', 'avg_air_density',
        'avg_parasitic_power', 'avg_induced_power', 'avg_climb_power'
    ]

    # Merge with means for imputation
    feature_df = feature_df.merge(
        ac_means_df, on='aircraft_type', how='left', suffixes=('', '_mean'))

    # 1. Impute Intensive Properties
    for col in intensive_cols:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].fillna(feature_df[f'{col}_mean'])
            feature_df[col] = feature_df[col].fillna(
                feature_df[f'{col}_mean'].mean())  # Fallback

    # 2. Impute Extensive Properties (depend on duration)
    # total_dist = duration * avg_speed (converted to km/s approx)
    # avg_speed is in knots. 1 knot = 0.000514444 km/s
    if 'total_dist' in feature_df.columns and feature_df['total_dist'].isnull().any():
        mask = feature_df['total_dist'].isnull()
        # speed (knots) * 1.852 (km/h) / 3600 (s/h) * duration (s)
        # = speed * 0.000514444 * duration
        feature_df.loc[mask, 'total_dist'] = feature_df.loc[mask,
                                                            'duration'] * feature_df.loc[mask, 'avg_speed'] * 0.000514444

    # n_points = duration (approx 1 point per sec)
    if 'n_points' in feature_df.columns:
        feature_df['n_points'] = feature_df['n_points'].replace(0, np.nan)
        feature_df['n_points'] = feature_df['n_points'].fillna(
            feature_df['duration'])

    # Fill any remaining NaNs (e.g. if duration was NaN, unlikely)
    feature_df = feature_df.fillna(0)

    # Save Features if requested
    if save_features_path:
        print(f"Saving features to {save_features_path}...")
        feature_df.to_parquet(save_features_path)

    # Prepare for prediction
    X = feature_df[FEATURES].copy()
    X['aircraft_type'] = X['aircraft_type'].astype('category')

    print("Predicting...")
    preds_kg = np.zeros(len(X))
    for model in models:
        preds_kg += model.predict(X)
    preds_kg /= len(models)

    # Create submission
    final_sub = sub_df[['idx', 'flight_id', 'start', 'end']].copy()
    final_sub['fuel_kg'] = preds_kg
    final_sub['fuel_kg'] = final_sub['fuel_kg'].fillna(0).clip(lower=0)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    out_path = os.path.join(OUTPUT_DIR, output_name)
    final_sub.to_parquet(out_path)
    print(f"Saved GBM submission to {out_path}")


def main():
    # 1. Load Models
    model_files = glob.glob(os.path.join(MODEL_DIR, 'lgb_model_fold_*.txt'))
    if not model_files:
        raise FileNotFoundError("No GBM models found!")

    print(f"Found {len(model_files)} GBM models.")
    models = [lgb.Booster(model_file=f) for f in model_files]

    # 2. Load Aircraft Means
    means_path = os.path.join(MODEL_DIR, 'ac_means.parquet')
    if not os.path.exists(means_path):
        raise FileNotFoundError(
            "No aircraft means found! Run optimized_gbm_pipeline.py first.")
    ac_means_df = pd.read_parquet(means_path)

    # 3. Load Aircraft Map and Airport Data
    fl_train = pd.read_parquet(os.path.join(
        DATA_DIR, 'flightlist_train.parquet'))
    fl_rank = pd.read_parquet(os.path.join(
        DATA_DIR, 'flightlist_rank.parquet'))
    fl_final_path = os.path.join(DATA_DIR, 'flightlist_final.parquet')
    apt_df = pd.read_parquet(os.path.join(DATA_DIR, 'apt.parquet'))

    dfs = [fl_train, fl_rank]
    if os.path.exists(fl_final_path):
        dfs.append(pd.read_parquet(fl_final_path))

    fl_all = pd.concat(dfs)

    # Apply Rare Mapping
    fl_all['aircraft_type'] = fl_all['aircraft_type'].replace(RARE_AC_MAP)

    ac_map = fl_all.set_index('flight_id')['aircraft_type'].to_dict()

    # Calculate O-D Distance for all flights
    print("Calculating O-D Distances...")
    # Merge origin coords
    fl_all = fl_all.merge(
        apt_df[['icao', 'latitude', 'longitude', 'elevation']].rename(columns={
            'latitude': 'origin_lat',
            'longitude': 'origin_lon',
            'elevation': 'origin_elev'
        }),
        left_on='origin_icao', right_on='icao', how='left'
    )
    # Merge destination coords
    fl_all = fl_all.merge(
        apt_df[['icao', 'latitude', 'longitude', 'elevation']].rename(columns={
            'latitude': 'dest_lat',
            'longitude': 'dest_lon',
            'elevation': 'dest_elev'
        }),
        left_on='destination_icao', right_on='icao', how='left'
    )

    # Calculate distance
    fl_all['od_distance'] = haversine_distance(
        fl_all['origin_lat'], fl_all['origin_lon'],
        fl_all['dest_lat'], fl_all['dest_lon']
    )

    # 4. Generate
    traj_dirs = [
        os.path.join(DATA_DIR, 'flights_train'),
        os.path.join(DATA_DIR, 'flights_rank'),
        os.path.join(DATA_DIR, 'flights_final')
    ]

    generate_gbm_submission(
        os.path.join(DATA_DIR, 'fuel_rank_submission.parquet'),
        traj_dirs, models, ac_map, ac_means_df, fl_all, 'submission_gbm_rank.parquet',
        save_features_path=os.path.join(
            MODEL_DIR, 'test_features_rank.parquet')
    )

    if os.path.exists(os.path.join(DATA_DIR, 'fuel_final_submission.parquet')):
        generate_gbm_submission(
            os.path.join(DATA_DIR, 'fuel_final_submission.parquet'),
            traj_dirs, models, ac_map, ac_means_df, fl_all, 'submission_gbm_final.parquet',
            save_features_path=os.path.join(
                MODEL_DIR, 'test_features_final.parquet')
        )


if __name__ == "__main__":
    main()
