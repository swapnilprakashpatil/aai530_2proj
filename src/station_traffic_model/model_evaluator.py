import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            test_predictions = []
            test_targets = []
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(targets.numpy())

        test_predictions = np.concatenate(test_predictions, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)

        scaler_check_out = joblib.load('saved_models/scalers/scaler_check_out.save')
        scaler_check_in = joblib.load('saved_models/scalers/scaler_check_in.save')
        check_out_predictions = scaler_check_out.inverse_transform(test_predictions[:, 0].reshape(-1, 1))
        check_in_predictions = scaler_check_in.inverse_transform(test_predictions[:, 1].reshape(-1, 1))

        check_out_targets = scaler_check_out.inverse_transform(test_targets[:, 0].reshape(-1, 1))
        check_in_targets = scaler_check_in.inverse_transform(test_targets[:, 1].reshape(-1, 1))

        return check_in_targets, check_in_predictions, check_out_targets, check_out_predictions

    def calculate_metrics(self, check_in_targets, check_in_predictions, check_out_targets, check_out_predictions):
        metrics = {
            'check_out': {
                'rmse': np.sqrt(mean_squared_error(check_out_targets, check_out_predictions)),
                'mae': mean_absolute_error(check_out_targets, check_out_predictions)
            },
            'check_in': {
                'rmse': np.sqrt(mean_squared_error(check_in_targets, check_in_predictions)),
                'mae': mean_absolute_error(check_in_targets, check_in_predictions)
            }
        }
        return metrics

    def plot_predictions(self, check_in_targets, check_in_predictions, check_out_targets, check_out_predictions):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.scatter(check_out_targets, check_out_predictions, alpha=0.5)
        max_val = max(check_out_targets.max(), check_out_predictions.max())
        min_val = min(check_out_targets.min(), check_out_predictions.min())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax1.set_title('Check-Out Predictions vs Actual')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.legend()

        ax2.scatter(check_in_targets, check_in_predictions, alpha=0.5)
        max_val = max(check_in_targets.max(), check_in_predictions.max())
        min_val = min(check_in_targets.min(), check_in_predictions.min())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax2.set_title('Check-In Predictions vs Actual')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_error_distribution(self, check_in_targets, check_in_predictions, check_out_targets, check_out_predictions):
        check_out_errors = check_out_predictions - check_out_targets
        check_in_errors = check_in_predictions - check_in_targets

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(check_out_errors, ax=ax1, kde=True)
        ax1.set_title('Check-Out Error Distribution')
        ax1.set_xlabel('Error')

        sns.histplot(check_in_errors, ax=ax2, kde=True)
        ax2.set_title('Check-In Error Distribution')
        ax2.set_xlabel('Error')

        plt.tight_layout()
        plt.show()
