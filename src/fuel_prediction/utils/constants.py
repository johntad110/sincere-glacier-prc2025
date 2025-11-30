"""
Constants and configuration values used throughout the project.
"""

# Physical Constants
GRAVITY = 9.81  # m/s^2
GAS_CONSTANT = 287.05  # J/(kg·K)
SEA_LEVEL_TEMP = 288.15  # K
SEA_LEVEL_PRESSURE = 101325  # Pa
LAPSE_RATE = 0.0065  # K/m
GAMMA = 1.4  # Heat capacity ratio for air
EARTH_RADIUS = 6371  # km

# Conversion Factors
KT_TO_MS = 0.514444  # Knots to m/s
FT_TO_M = 0.3048  # Feet to meters

# Aircraft Type Mappings (Rare to Common)
RARE_AC_MAP = {
    'B748': 'B744',  # 747-8 -> 747-400
    'B77L': 'B77W',  # 777-200LR -> 777-300ER
    'A388': 'B744',  # A380 -> 747-400 (Best proxy for heavy 4-engine)
    'B39M': 'B38M',  # 737 Max 9 -> Max 8
    'A318': 'A319',  # A318 -> A319
    'MD11': 'B772',  # MD11 -> 777-200 (Approximate heavy)
}

# Aircraft Categories
NARROW_BODY_AIRCRAFT = [
    'A320', 'A319', 'A321', 'B737', 'B738',
    'B739', 'A20N', 'A21N', 'B38M', 'A318', 'B39M', 'B752'
]

WIDE_BODY_AIRCRAFT = [
    'B772', 'B77W', 'B763', 'B764', 'B788', 'B789',
    'A332', 'A333', 'A359', 'A35K', 'B744'
]

# Fuel Flow Rate Limits (kg/s) for outlier detection
FUEL_FLOW_LIMIT_NARROW = 4.0
FUEL_FLOW_LIMIT_WIDE = 12.0

# Feature Lists
GBM_FEATURES = [
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

TRAJECTORY_COLUMNS_TO_INTERPOLATE = [
    'groundspeed', 'track', 'vertical_rate', 
    'altitude', 'latitude', 'longitude'
]

# Outlier Detection
BOUNDARY_TIME_THRESHOLD = 0.05  # Remove first and last 5% of flight time
ACCELERATION_CLIP_RANGE = (-5.0, 5.0)  # m/s²
ENERGY_RATE_CLIP_RANGE = (-50.0, 50.0)  # m/s

# Model Parameters
DEFAULT_N_FOLDS = 5
DEFAULT_RANDOM_STATE = 42

# LSTM Parameters
DEFAULT_SEQ_LEN = 32
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 3
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LSTM_EPOCHS = 50

# GBM Parameters
DEFAULT_GBM_PARAMS = {
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

DEFAULT_GBM_NUM_BOOST_ROUND = 2000
DEFAULT_GBM_EARLY_STOPPING = 100

# Stacking Parameters
DEFAULT_STACKING_ALPHA = 1.0
DEFAULT_STACKING_POSITIVE = True

# File Paths (relative)
DEFAULT_DATA_DIR = './prc-2025-datasets'
DEFAULT_MODELS_DIR = './models'
DEFAULT_SUBMISSIONS_DIR = './submissions'
