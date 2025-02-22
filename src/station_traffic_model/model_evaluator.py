import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def create_evaluator(self):

        def evaluation_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                return y_pred, y

        evaluator = Engine(evaluation_step)

        metrics = {
            'mse': MeanSquaredError(),
            'rmse': RootMeanSquaredError(),
            'mae': MeanAbsoluteError()
        }

        for name, metric in metrics.items():
            metric.attach(evaluator, name)

        return evaluator

    def evaluate(self, test_loader):
        self.model.eval()
        predictions = []
        targets = []

        evaluator = self.create_evaluator()
        evaluator.run(test_loader)

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y_pred = self.model(x)
                predictions.append(y_pred.cpu().numpy())
                targets.append(y.numpy())

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        return targets, predictions

    def calculate_metrics(self, targets, predictions):
        targets_2d = targets.reshape(-1, targets.shape[-1])
        predictions_2d = predictions.reshape(-1, predictions.shape[-1])

        mse = mean_squared_error(targets_2d, predictions_2d)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets_2d, predictions_2d)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        print("\nModel Evaluation Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return metrics

    def plot_predictions(self, targets, predictions):
        plt.figure(figsize=(10, 6))

        targets_flat = targets.reshape(-1)
        predictions_flat = predictions.reshape(-1)

        plt.scatter(targets_flat, predictions_flat, alpha=0.5, s=1)

        max_val = max(targets_flat.max(), predictions_flat.max())
        min_val = min(targets_flat.min(), predictions_flat.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
