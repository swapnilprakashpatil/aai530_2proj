import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import LSTMTrafficModel
from preprocessor import Preprocessor

SEQUENCE_LENGTH = 14
BATCH_SIZE = 32

data = Preprocessor.get_preprocessed_data('../../data/combined_tripdata_since_2023.csv')

x, y = Preprocessor.create_sequences(data, SEQUENCE_LENGTH)
x_train, y_train, x_val, y_val, x_test, y_test = Preprocessor.split_data(x, y, train_proportion=0.6, val_proportion=0.2)

# Create Datasets and Loaders
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

input_size = x_train.shape[2]
output_size = y_train.shape[1]
model = LSTMTrafficModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
best_val_loss = float('inf')
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, 'best_model.pth')

if best_model_state is not None:
    model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    test_predictions = []
    test_targets = []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        test_predictions.append(outputs.numpy())
        test_targets.append(targets.numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_targets = np.concatenate(test_targets, axis=0)

scaler_check_out = joblib.load('saved_models/scalers/scaler_check_out.save')
scaler_check_in = joblib.load('saved_models/scalers/scaler_check_in.save')
check_out_predictions = scaler_check_out.inverse_transform(test_predictions[:, 0].reshape(-1, 1))
check_in_predictions = scaler_check_in.inverse_transform(test_predictions[:, 1].reshape(-1, 1))

check_out_targets = scaler_check_out.inverse_transform(test_targets[:, 0].reshape(-1, 1))
check_in_targets = scaler_check_in.inverse_transform(test_targets[:, 1].reshape(-1, 1))

rmse_check_out = np.sqrt(mean_squared_error(check_out_targets, check_out_predictions))
mae_check_out = mean_absolute_error(check_out_targets, check_out_predictions)

rmse_check_in = np.sqrt(mean_squared_error(check_in_targets, check_in_predictions))
mae_check_in = mean_absolute_error(check_in_targets, check_in_predictions)

print(f'Check-Out Counts - RMSE: {rmse_check_out}, MAE: {mae_check_out}')
print(f'Check-In Counts - RMSE: {rmse_check_in}, MAE: {mae_check_in}')
