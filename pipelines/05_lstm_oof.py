import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

OUTPUT_DIR = './lstm_optimized'
BATCH_SIZE = 512
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

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

def generate_oof():
    print("Loading saved data...")
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'X_seq.npy')):
        raise FileNotFoundError("Saved data not found in " + OUTPUT_DIR)
        
    X_seq = np.load(os.path.join(OUTPUT_DIR, 'X_seq.npy'))
    X_static = np.load(os.path.join(OUTPUT_DIR, 'X_static.npy'))
    y = np.load(os.path.join(OUTPUT_DIR, 'y.npy'))
    groups = np.load(os.path.join(OUTPUT_DIR, 'groups.npy'))
    ac_classes = np.load(os.path.join(OUTPUT_DIR, 'classes.npy'), allow_pickle=True)
    ids = np.load(os.path.join(OUTPUT_DIR, 'ids.npy'))  # Load idx
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cross-Validation
    gkf = GroupKFold(n_splits=5)
    
    oof_preds = np.zeros(len(y))
    
    print("Generating OOF predictions...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq, y, groups)):
        print(f"Fold {fold+1}")
        
        # Load Model
        model_path = os.path.join(OUTPUT_DIR, f'model_fold_{fold}.pth')
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} not found, skipping...")
            continue
            
        model = FuelLSTM(
            seq_input_dim=X_seq.shape[2],
            static_input_dim=X_static.shape[1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=len(ac_classes)
        ).to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        
        # Validation Data
        X_seq_val = torch.from_numpy(X_seq[val_idx])
        X_static_val = torch.from_numpy(X_static[val_idx])
        
        ds = torch.utils.data.TensorDataset(X_seq_val, X_static_val)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for x_s, x_st in tqdm(loader, leave=False):
                x_s, x_st = x_s.to(device), x_st.to(device)
                out = model(x_s, x_st)
                preds.extend(out.cpu().numpy())
                
        oof_preds[val_idx] = np.array(preds)
        
    # Save OOF
    # Include idx for proper merging with GBM OOF
    df_oof = pd.DataFrame({
        'idx': ids,
        'flight_id': groups,
        'fuel_kg': y,
        'lstm_pred': oof_preds
    })
    
    
    out_path = os.path.join(OUTPUT_DIR, 'lstm_oof.parquet')
    df_oof.to_parquet(out_path)
    print(f"Saved LSTM OOF to {out_path}")

if __name__ == "__main__":
    generate_oof()