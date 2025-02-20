import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss


class StationTrafficModel(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, output_size, output_sequence_length, num_layers, dropout):
        super(StationTrafficModel, self).__init__()
        self.model_type = model_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_sequence_length = output_sequence_length
        self.output_size = output_size

        if self.model_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif self.model_type == 'gru':
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("model_type must be either 'lstm' or 'gru'")

        self.decoder = nn.GRU(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        if self.model_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            _, (hidden, _) = self.encoder(x, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            _, hidden = self.encoder(x, h0)

        decoder_hidden = hidden[-1].unsqueeze(0)
        decoder_input = torch.zeros(batch_size, 1, self.output_size).to(x.device)
        outputs = []

        for _ in range(self.output_sequence_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            prediction = self.output_layer(decoder_output)
            outputs.append(prediction)
            decoder_input = prediction

        return torch.cat(outputs, dim=1)

    def train_model(self, criterion, optimizer, train_loader, val_loader, num_epochs, patience, device):
        self.to(device)

        trainer = create_supervised_trainer(self, optimizer, criterion, device=device)
        evaluator = create_supervised_evaluator(
            self,
            metrics={'loss': Loss(criterion)},
            device=device
        )
        history = {'train_loss': [], 'val_loss': []}

        pbar = ProgressBar()
        pbar.attach(trainer)

        checkpoint_handler = ModelCheckpoint(
            'checkpoints',
            'best_model',
            n_saved=1,
            require_empty=False,
            score_function=lambda engine: -engine.state.metrics['loss'],
            score_name='val_loss'
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {'model': self})

        early_stopping = EarlyStopping(
            patience=patience,
            score_function=lambda engine: -engine.state.metrics['loss'],
            trainer=trainer
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            train_loss = metrics['loss']
            history['train_loss'].append(train_loss)
            pbar.log_message(f"Epoch [{engine.state.epoch}/{num_epochs}] Train Loss: {train_loss:.4f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_loss = metrics['loss']
            history['val_loss'].append(val_loss)
            pbar.log_message(f"Epoch [{engine.state.epoch}/{num_epochs}] Validation Loss: {val_loss:.4f}")

        trainer.run(train_loader, max_epochs=num_epochs)

        best_model_path = checkpoint_handler.last_checkpoint
        if best_model_path:
            self.load_state_dict(torch.load(best_model_path))

        return self, history
