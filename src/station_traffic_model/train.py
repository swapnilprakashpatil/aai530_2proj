import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from hyperparameter_tuning import objective
from model_evaluator import ModelEvaluator
from preprocessor import Preprocessor
from station_traffic_model import StationTrafficModel

DATA_PATH = "../../data/combined_tripdata.csv"
INPUT_SEQUENCE_LENGTH = 30
OUTPUT_SEQUENCE_LENGTH = 1

PATIENCE = 3
NUM_EPOCHS = 20

PERFORM_TUNING = True
NUM_TRIALS = 30
params = {
    'model_type': 'lstm',
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'weight_decay': 0.0001
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing
preprocessor = Preprocessor()
preprocessor.preprocess_data(DATA_PATH, input_sequence_length=INPUT_SEQUENCE_LENGTH,
                             output_sequence_length=OUTPUT_SEQUENCE_LENGTH)
input_size = preprocessor.train_dataset.tensors[0].shape[2]
output_size = preprocessor.train_dataset.tensors[1].shape[2]

# Hyperparameter Tuning
if PERFORM_TUNING:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, input_size, output_size,
                                           OUTPUT_SEQUENCE_LENGTH, PATIENCE, NUM_EPOCHS, preprocessor, device),
                   n_trials=NUM_TRIALS)
    print("Best hyperparameters:", study.best_params)
    params = params | study.best_params

# Training
train_loader, val_loader, test_loader = preprocessor.get_loaders(params['batch_size'])
model = StationTrafficModel(params['model_type'], input_size, params['hidden_size'],
                            output_size,
                            OUTPUT_SEQUENCE_LENGTH, params['num_layers'],
                            params['dropout'])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
model.train_model(criterion, optimizer, train_loader, val_loader, NUM_EPOCHS, PATIENCE, device)

# Evaluation
evaluator = ModelEvaluator(model)
targets, predictions = evaluator.evaluate(test_loader)
metrics = evaluator.calculate_metrics(targets, predictions)
evaluator.plot_predictions(targets, predictions)