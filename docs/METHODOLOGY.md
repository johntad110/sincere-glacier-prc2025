# Methodology Documentation

## Team Sincere Glacier - PRC 2025 Data Challenge

### Executive Summary

This document describes the technical approach used by Team Sincere Glacier for predicting aircraft fuel burn in the PRC 2025 Data Challenge. Our solution achieves competitive performance (~214 kg RMSE) through a hybrid ensemble combining physics-aware gradient boosting with sequence-aware deep learning.

---

## 1. Problem Understanding

### Objective
Predict total fuel consumption (in kg) for discretized flight intervals given:
- Trajectory data (timestamp, position, speed, altitude, vertical rate)
- Flight metadata (aircraft type, origin/destination)
- Airport information

### Challenges
1. **High Variability**: Different aircraft types have vastly different fuel consumption profiles
2. **Complex Physics**: Fuel burn depends on atmospheric conditions, aircraft weight, flight phase
3. **Sparse Features**: Limited direct information about mass, weather, or engine performance
4. **Temporal Dependencies**: Fuel consumption patterns evolve throughout flights

---

## 2. Solution Architecture

### 2.1 Hybrid Stacking Framework

Our solution uses a two-level stacking ensemble:

```
Level 0 (Base Models):
├── Physics-Aware LightGBM → Captures aggregate patterns
└── Sequence-Aware LSTM → Captures temporal dynamics

Level 1 (Meta-Model):
└── Non-Negative Ridge Regression → Weighted combination
```

### Key Design Decisions

**Why Both GBM and LSTM?**
- **GBM Strength**: Excellent at learning from aggregated statistics, handles missing data well, interpretable feature importance
- **LSTM Strength**: Captures sequential patterns, models flight phase transitions, learns temporal correlations
- **Complementary**: GBM excels on cruise/descent, LSTM better on climb (where sequence matters)

**Why Ridge with Positive Constraints?**
- Only positive weights ensure both models contribute (no cancellation)
- Ridge regularization prevents overfitting to OOF predictions
- Simple meta-model reduces ensemble complexity

---

## 3. Feature Engineering

### 3.1 Physics-Based Features

#### ISA Standard Atmosphere Model
```python
T(h) = T₀ - L × h  # Temperature at altitude h
ρ(h) = P(h) / (R × T(h))  # Air density
a(h) = √(γ × R × T(h))  # Speed of sound
```

Derived Features:
- **Mach Number**: `M = v / a(h)` - Critical for drag calculations
- **Dynamic Pressure**: `q = 0.5 × ρ × v²` - Proportional to aerodynamic forces
- **Air Density**: `ρ(h)` - Affects thrust and drag

#### Energy-Based Features

**Specific Energy Rate**:
```
E_rate = ḣ + (v/g) × a
```
where:
- `ḣ` = vertical speed (rate of climb)
- `a` = acceleration
- `g` = gravity

This captures the rate of change of total mechanical energy (potential + kinetic).

**Power Proxies**:
1. **Parasitic Power**: `P_para ∝ ρ × v³` (form drag, increases with speed)
2. **Induced Power**: `P_ind ∝ 1/v` (lift-induced drag, worse at low speeds)
3. **Climb Power**: `P_climb = ḣ` (energy required for altitude gain)

### 3.2 Contextual Features

#### Temporal Features
- `time_since_takeoff`: Proxy for fuel consumed (heavier at start)
- `time_to_landing`: Remaining flight duration
- `relative_time`: Normalized position in flight (0-1)

#### Spatial Features
- `od_distance`: Great circle distance between airports
- `origin_lat`, `origin_lon`, `origin_elev`: Departure conditions
- `dest_lat`, `dest_lon`, `dest_elev`: Arrival conditions

#### Aircraft Features
- `aircraft_type`: Categorical (embedded in LSTM, one-hot in GBM)
- Rare aircraft mapping (e.g., A388→B744) to improve generalization

### 3.3 Mass Proxy

Aircraft weight significantly affects fuel burn but isn't directly observed. We estimate takeoff weight from climb performance:

```
F_thrust - D = m × a  # Newton's 2nd law
Weight estimation from climb rate in early flight phase
```

This mass proxy is merged as an additional feature.

---

## 4. Model Details

### 4.1 LightGBM Model

**Architecture**:
- Objective: Regression (MSE)
- Boosting: GBDT with 2000 rounds
- Learning Rate: 0.05
- Max Leaves: 63
- Bagging: 80% feature + 80% data bagging

**Cross-Validation**:
- 5-fold GroupKFold (groups = `flight_id`)
- Ensures no data leakage between train/val

**Feature Importance** (Top 10):
1. `mass_proxy` (weight estimation)
2. `duration` (interval length)
3. `avg_climb_power` (energy expenditure)
4. `total_dist` (distance traveled)
5. `avg_energy_rate` (rate of energy change)
6. `avg_parasitic_power` (speed-related drag)
7. `od_distance` (total flight distance)
8. `aircraft_type` (aircraft model)
9. `relative_time` (flight phase)
10. `avg_alt` (cruise altitude)

**Outlier Removal**:
- Remove intervals with excessive fuel flow rates:
  - Narrow-body: `> 4.0 kg/s`
  - Wide-body: `> 12.0 kg/s`
- Remove boundary intervals (`relative_time < 0.05` or `> 0.95`) to exclude taxiing

### 4.2 LSTM Model

**Architecture**:
```
Input:
├── Sequence (32 timesteps): altitude, speed, vertical_rate, acceleration
└── Static: duration, total_dist, aircraft_type_embedding

Layers:
├── Aircraft Type Embedding (dim=16)
├── LSTM (3 layers, hidden_dim=256, dropout=0.2)
├── Fully Connected (hidden_dim → 128 → 64 → 1)
└── Output: Predicted fuel (kg)
```

**Training**:
- Loss: MSE
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Batch Size: 256
- Epochs: 50 (with early stopping)
- Device: GPU (CUDA) if available

**Data Preprocessing**:
1. Resample trajectories to fixed length (32 points)
2. Normalize sequences using z-score standardization
3. Handle missing trajectories with default patterns

### 4.3 Stacking Meta-Model

**Ridge Regression**:
```python
from sklearn.linear_model import Ridge

meta_model = Ridge(alpha=1.0, positive=True, fit_intercept=True)
```

**Input Features**:
- `pred_gbm`: GBM Out-of-Fold prediction
- `pred_lstm`: LSTM Out-of-Fold prediction

**Output**:
- `fuel_kg_final = w₁ × pred_gbm + w₂ × pred_lstm + b`

Typical learned weights: `w₁ ≈ 0.6`, `w₂ ≈ 0.4`

---

## 5. Training Pipeline

### 5.1 Phase 1: Base Model Training

**GBM Training**:
1. Load and process all trajectories in parallel
2. Extract physics features + aggregate statistics
3. Merge contextual features (airports, mass proxy)
4. 5-fold CV with GroupKFold
5. Save fold models + OOF predictions

**LSTM Training**:
1. Load trajectories and resample to fixed length
2. Create torch datasets with sequences
3. Train separate model per fold
4. Generate OOF predictions on validation folds
5. Save fold models + preprocessing stats

### 5.2 Phase 2: Meta-Model Training

1. Load GBM OOF predictions
2. Load LSTM OOF predictions
3. Merge on `idx`
4. Train Ridge regression
5. Save meta-model

---

## 6. Inference Pipeline

### 6.1 Prediction Generation

For each test dataset (ranking/final):

**GBM Inference**:
1. Load test trajectories
2. Extract same features as training
3. Predict with all 5 fold models
4. Average predictions

**LSTM Inference**:
1. Load test trajectories
2. Preprocess sequences
3. Predict with all 5 fold models
4. Average predictions

**Stacking Inference**:
1. Get GBM predictions
2. Get LSTM predictions
3. Apply meta-model: `final = meta.predict([[gbm_pred, lstm_pred]])`
4. Save submission file

---

## 7. Key Ideas

1. **Physics-Informed Features**: ISA atmosphere model + energy-based features ground the model in aerodynamics
2. **Mass Estimation**: Indirect weight proxy from climb performance significantly improves predictions
3. **Hybrid Architecture**: Combines strengths of tree-based and sequence models
4. **Robust Outlier Filtering**: Aircraft-specific flow rate thresholds remove bad data
5. **Temporal Awareness**: Relative time feature helps model distinguish takeoff/cruise/landing

---

## 8. Performance Analysis

### Cross-Validation Results

| Model | Mean RMSE (kg) | Std RMSE | Min RMSE | Max RMSE |
|-------|----------------|----------|----------|----------|
| GBM | ~205 | ~8 | ~195 | ~215 |
| LSTM | ~220 | ~12 | ~205 | ~235 |
| **Stacking** | **~200** | **~7** | **~192** | **~208** |

### Error Analysis

**High Error Cases**:
- Long-haul wide-body flights (B777, A350)
- Extreme weather deviations (not in features)
- Unusual flight profiles (diversions, holdings)

**Low Error Cases**:
- Short-haul narrow-body (A320, B737)
- Standard cruise profiles
- Common aircraft types with abundant training data

---

## 9. Lessons Learned

### What Worked Well
- Physics-based features provided strong signal  
- Mass proxy was most important feature for GBM  
- Ensemble diversity (GBM + LSTM) improved robustness  
- GroupKFold prevented leakage and improved generalization  
- Outlier removal significantly reduced noise  

### What Could Be Improved
❌ LSTM could benefit from attention mechanism  
❌ Weather data (wind, temperature) not available  
❌ Direct mass measurements would help significantly  
❌ More sophisticated phase detection (climb/cruise/descent)  
❌ Aircraft-specific models might reduce error on rare types  

---

## 10. Conclusion

Our hybrid stacking approach successfully combines physics-aware feature engineering (GBM) with sequence learning (LSTM) to achieve competitive fuel prediction performance. The modular, well-documented codebase ensures reproducibility and facilitates future improvements.

**Key Takeaway**: Domain knowledge (aviation physics) + modern ML = strong performance on complex regression tasks.

---

**Authors**: Team Sincere Glacier  
**Date**: November 2025  
**Competition**: PRC 2025 Data Challenge
