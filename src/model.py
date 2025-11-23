import torch
import torch.nn as nn
import math
import mlflow
import pickle

def log_model_to_mlflow(model, params, metrics, model_name="HydroTransNet", run_name=None):
    """
    Log the trained PyTorch model, parameters and metrics to MLflow.
    Args:
        model: PyTorch model instance
        params: dict of hyperparameters (config)
        metrics: dict of performance metrics
        model_name: name of the model
        run_name: name for the MLflow run, optional
    """
    clean_metrics = {k: v for k, v in metrics.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    parent_run = mlflow.active_run()
    parent_run_id = parent_run.info.run_id if parent_run else None

    try:
        # Try to log as a model if possible (local MLflow, supported server)
        if parent_run_id:
            with mlflow.start_run(run_id=parent_run_id):
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_params(params)
                    if clean_metrics: mlflow.log_metrics(clean_metrics)
                    mlflow.pytorch.log_model(model, model_name)
        else:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                if clean_metrics: mlflow.log_metrics(clean_metrics)
                mlflow.pytorch.log_model(model, model_name)
    except Exception as e:
        print("MLflow model registry/log_model failed, logging as artifact instead.")
        temp_model_path = f"{model_name}.pt"
        torch.save(model.state_dict(), temp_model_path)
        if parent_run_id:
            with mlflow.start_run(run_id=parent_run_id):
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_params(params)
                    if clean_metrics: mlflow.log_metrics(clean_metrics)
                    mlflow.log_artifact(temp_model_path)
        else:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                if clean_metrics: mlflow.log_metrics(clean_metrics)
                mlflow.log_artifact(temp_model_path)

    print(f"Model and metrics logged to MLflow with run: {mlflow.active_run().info.run_id}")

class PositionalEncoding(nn.Module):
    """Adds positional encoding to input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class HydroTransNet(nn.Module):
    """
    Transformer-based water quality prediction model, 
    enhanced with LSTM + GRU layers for temporal reasoning.
    Output layer uses ReLU activation for physically valid (≥0) water parameters.
    """
    def __init__(
        self,
        input_dim=7,
        d_model=128,
        nhead=4,
        num_encoder_layers=8,
        dim_feedforward=512,
        dropout=0.01,
        output_dim=3,
        lstm_hidden_dim=64,
        gru_hidden_dim=64,
        lstm_layers=1,
        gru_layers=1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.lstm = nn.LSTM(d_model, lstm_hidden_dim, num_layers=lstm_layers, batch_first=False)
        self.gru = nn.GRU(lstm_hidden_dim, gru_hidden_dim, num_layers=gru_layers, batch_first=False)
        self.fc1 = nn.Linear(gru_hidden_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, output_dim)
        self.out_relu = nn.ReLU()  # Ensures physical parameter ≥ 0
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        x = x[-1, :, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        output = self.fc_out(x)
        output = self.out_relu(output)  # Non-negative guarantee
        return output

def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":
    input_dim = 7
    d_model = 128
    nhead = 4
    num_encoder_layers = 8
    dim_feedforward = 512
    dropout = 0.01
    output_dim = 3
    model = HydroTransNet(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        output_dim=output_dim,
    )
    params = {
        "input_dim": input_dim,
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": num_encoder_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "output_dim": output_dim,
        "activation": "gelu",
        "batch_first": False,
        "architecture": "transformer_encoder + LSTM + GRU + FC-ReLU",
        "out_activation": "ReLU"
    }
    metrics = {
        "train_loss": 0.0123,
        "val_loss": 0.0156,
        "val_r2_score": 0.85,
    }
    log_model_to_mlflow(model, params, metrics, model_name="HydroTransNet", run_name="water_quality_prediction_run")
    print(f"Total trainable parameters: {count_parameters(model)}")
