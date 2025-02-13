import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import LSTMTrafficModel
from model_evaluator import ModelEvaluator
from preprocessor import Preprocessor

SEQUENCE_LENGTH = 14
BATCH_SIZE = 32
PATIENCE = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Preprocessor.get_preprocessed_data('../../data/combined_tripdata_since_2023.csv')
x, y = Preprocessor.create_sequences(data, SEQUENCE_LENGTH)
x_train, y_train, x_val, y_val, x_test, y_test = Preprocessor.split_data(x, y, train_proportion=0.6, val_proportion=0.2)

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

input_size = x_train.shape[2]
output_size = y_train.shape[1]
model = LSTMTrafficModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=output_size)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model, metrics={'loss': Loss(criterion)}, device=device)

checkpoint_handler = ModelCheckpoint(
    'saved_models',
    'best_model',
    score_function=lambda engine: -engine.state.metrics['loss'],
    score_name='val_loss',
    n_saved=1,
    require_empty=False
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'model': model})


def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss


early_stopping_handler = EarlyStopping(
    patience=PATIENCE,
    score_function=score_function,
    trainer=trainer
)
evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)


# Progress tracking
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] - Loss: {metrics['loss']:.6f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(f"Validation Results - Epoch[{trainer.state.epoch}] - Loss: {metrics['loss']:.6f}")


trainer.run(train_loader, max_epochs=10)

best_model_path = checkpoint_handler.last_checkpoint
model.load_state_dict(torch.load(best_model_path))

evaluator = ModelEvaluator(model)
check_in_targets, check_in_predictions, check_out_targets, check_out_predictions = evaluator.evaluate(test_loader)
print(evaluator.calculate_metrics(check_in_targets, check_in_predictions, check_out_targets, check_out_predictions))
evaluator.plot_predictions(check_in_targets, check_in_predictions, check_out_targets, check_out_predictions)
evaluator.plot_error_distribution(check_in_targets, check_in_predictions, check_out_targets, check_out_predictions)
