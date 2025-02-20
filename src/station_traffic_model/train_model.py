import optuna
import torch
import torch.nn as nn
import torch.optim as optim

from hyperparameter_tuning import objective
from model_evaluator import ModelEvaluator
from preprocessor import Preprocessor
from station_traffic_model import StationTrafficModel

DATA_PATH = "../../data/combined_tripdata_2020.csv"
INPUT_SEQUENCE_LENGTH = 30
OUTPUT_SEQUENCE_LENGTH = 2

PATIENCE = 3
NUM_EPOCHS = 20

PERFORM_TUNING = True
NUM_TRIALS = 20
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
preprocessor = Preprocessor(input_sequence_length=INPUT_SEQUENCE_LENGTH,
                            output_sequence_length=OUTPUT_SEQUENCE_LENGTH)
preprocessor.preprocess_data(DATA_PATH)

# Hyperparameter Tuning
if PERFORM_TUNING:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, preprocessor.x_train_tensor, preprocessor.y_train_tensor,
                                           preprocessor.x_val_tensor, preprocessor.y_val_tensor, OUTPUT_SEQUENCE_LENGTH,
                                           PATIENCE, NUM_EPOCHS, device),
                   n_trials=NUM_TRIALS)
    print("Best hyperparameters:", study.best_params)
    params = params | study.best_params

# Training
train_loader, val_loader, test_loader = preprocessor.get_loaders(params['batch_size'])
model = StationTrafficModel(params['model_type'], preprocessor.x_train_tensor.shape[2], params['hidden_size'],
                            preprocessor.y_train_tensor.shape[2],
                            OUTPUT_SEQUENCE_LENGTH, params['num_layers'],
                            params['dropout'])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
model.train_model(criterion, optimizer, train_loader, val_loader, NUM_EPOCHS, PATIENCE, device)

# Evaluation
evaluator = ModelEvaluator(model)
check_in_targets, check_in_predictions, check_out_targets, check_out_predictions = evaluator.evaluate(test_loader)
evaluator.calculate_metrics(check_in_targets, check_in_predictions, check_out_targets, check_out_predictions)
evaluator.plot_predictions(check_in_targets, check_in_predictions, check_out_targets, check_out_predictions)
