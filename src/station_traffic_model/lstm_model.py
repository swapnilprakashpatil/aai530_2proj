import torch
import torch.nn as nn


class LSTMTrafficModel(nn.Module):
    def __init__(self, hidden_size, num_layers, prediction_horizon):
        super(LSTMTrafficModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_horizon = prediction_horizon

        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 2 * prediction_horizon)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(-1, self.prediction_horizon, 2)

        return out
