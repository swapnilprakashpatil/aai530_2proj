import torch
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import LSTMTrafficModel
from model_evaluator import ModelEvaluator
from preprocessor import Preprocessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQUENCE_LENGTH = 14
BATCH_SIZE = 32

data = Preprocessor.get_preprocessed_data('../../data/combined_tripdata_since_2023.csv')
x, y = Preprocessor.create_sequences(data, SEQUENCE_LENGTH)
_, _, _, _, x_test, y_test = Preprocessor.split_data(x, y, train_proportion=0.6, val_proportion=0.2)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

input_size = x_test.shape[2]
output_size = y_test.shape[1]
model = LSTMTrafficModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=output_size)

best_model_path = 'saved_models/best_model.pt'
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
model.eval()

evaluator = ModelEvaluator(model)
check_in_targets, check_in_predictions, check_out_targets, check_out_predictions = evaluator.evaluate(test_loader)

metrics = evaluator.calculate_metrics(
    check_in_targets,
    check_in_predictions,
    check_out_targets,
    check_out_predictions
)
print("\nModel Evaluation Metrics:")
print(metrics)

print("\nGenerating visualization plots...")
evaluator.plot_predictions(
    check_in_targets,
    check_in_predictions,
    check_out_targets,
    check_out_predictions
)
evaluator.plot_error_distribution(
    check_in_targets,
    check_in_predictions,
    check_out_targets,
    check_out_predictions
)
