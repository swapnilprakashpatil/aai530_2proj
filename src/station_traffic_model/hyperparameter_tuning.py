import torch.nn as nn
import torch.optim as optim

from model_evaluator import ModelEvaluator
from station_traffic_model import StationTrafficModel


def objective(trial, input_size: int, output_size, output_sequence_length, patience, num_epochs, preprocessor,
              device):
    model_type = trial.suggest_categorical('model_type', ['lstm', 'gru'])
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)

    train_loader, val_loader, _ = preprocessor.get_loaders(batch_size)
    model = StationTrafficModel(model_type, input_size, hidden_size, output_size,
                                output_sequence_length, num_layers,
                                dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train_model(criterion, optimizer, train_loader, val_loader, num_epochs, patience, device)
    evaluator = ModelEvaluator(model)
    targets, predictions = evaluator.evaluate(val_loader)
    metrics = evaluator.calculate_metrics(targets, predictions)

    return metrics['mse']
