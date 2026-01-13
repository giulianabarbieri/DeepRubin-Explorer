import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LightCurveDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)  # Shape: (N, n_times, n_channels)
        self.y = pd.read_csv(y_path).values.squeeze()
        
        # Convertir etiquetas de texto a índices numéricos
        if self.y.dtype.kind in ['U', 'S', 'O']:  # Unicode, bytes, or object (string)
            unique_classes = sorted(np.unique(self.y))
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            self.y = np.array([self.class_to_idx[c] for c in self.y], dtype=np.int64)
        else:
            self.class_to_idx = None
            self.idx_to_class = None
            self.y = self.y.astype(np.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Obtener el ejemplo: shape (n_times, n_channels)
        x = self.X[idx]
        # Transponer a (n_channels, n_times) para Conv1d
        x = x.T  # Shape: (n_channels, n_times)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
