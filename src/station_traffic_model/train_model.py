import torch
import torch.nn as nn
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, Loss
from torch.utils.data import DataLoader

from lstm_model import LSTMTrafficModel
from preprocessor import Preprocessor
from time_series_dataset import TimeSeriesDataset

INPUT_WINDOW = 30
PREDICTION_HORIZON = 7
BATCH_SIZE = 32

data = Preprocessor.get_preprocessed_data('../../data/combined_tripdata_since_2023.csv')
sequences = Preprocessor.create_sequences(data, INPUT_WINDOW)

train_size = int(len(sequences) * 0.8)
train_sequences = sequences[:train_size]
val_sequences = sequences[train_size:]

train_X, train_y = Preprocessor.convert_sequences_to_tensors(train_sequences)
val_X, val_y = Preprocessor.convert_sequences_to_tensors(val_sequences)

train_dataset = TimeSeriesDataset(train_X, train_y)
val_dataset = TimeSeriesDataset(val_X, val_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = LSTMTrafficModel(hidden_size=64, num_layers=3, prediction_horizon=PREDICTION_HORIZON).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def val_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)
        return output, y


trainer = Engine(train_step)
evaluator = Engine(val_step)

metrics = {
    'mse': MeanSquaredError(),
    'loss': Loss(criterion)
}

for name, metric in metrics.items():
    metric.attach(evaluator, name)

# Add progress bar
ProgressBar().attach(trainer, output_transform=lambda x: {'loss': x})


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    mse = metrics['mse']
    val_loss = metrics['loss']
    print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg loss: {val_loss:.4f} Avg MSE: {mse:.4f}")


num_epochs = 20

trainer.run(train_loader, max_epochs=num_epochs)
