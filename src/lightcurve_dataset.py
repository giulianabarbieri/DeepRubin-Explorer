import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LightCurveDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = pd.read_csv(y_path).values.squeeze()
        # Si las etiquetas son texto, conviértelas a índices
        if self.y.dtype.type is np.str_:
            classes = sorted(set(self.y))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            self.y = np.array([self.class_to_idx[c] for c in self.y])
        else:
            self.class_to_idx = None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
