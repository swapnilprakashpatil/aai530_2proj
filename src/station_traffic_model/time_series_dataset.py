from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences_X, sequences_y):
        self.X = sequences_X
        self.y = sequences_y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
