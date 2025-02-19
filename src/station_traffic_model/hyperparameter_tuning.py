import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from model_evaluator import ModelEvaluator
from station_traffic_model import StationTrafficModel


def objective(trial, x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, output_sequence_length, patience,
              num_epochs, device):
    model_type = trial.suggest_categorical('model_type', ['lstm', 'gru'])
    hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.01, step=0.001)

    train_dataset = data_utils.TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = data_utils.TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = StationTrafficModel(model_type, x_train_tensor.shape[2], hidden_size, y_train_tensor.shape[2], output_sequence_length, num_layers,
                                dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train_model(criterion, optimizer, train_loader, val_loader, num_epochs, patience, device)
    evaluator = ModelEvaluator(model)
    check_in_targets, check_in_predictions, check_out_targets, check_out_predictions = evaluator.evaluate(val_loader)
    metrics = evaluator.calculate_metrics(check_in_targets, check_in_predictions, check_out_targets,
                                          check_out_predictions)
    return metrics['check_out']['rmse'] + metrics['check_in']['rmse']
