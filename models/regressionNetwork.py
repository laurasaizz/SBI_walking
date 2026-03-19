import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.amp.grad_scaler import GradScaler

class RegressionNetwork(nn.Module):
    def __init__(self, input_size=200, hidden_size=400, layers=6, output_size=3):
        """ """
        super().__init__()
        self.input_size = input_size
        self.layers = layers
        self.encoder_layers = []
        self.encoder_layers.append((nn.Linear(input_size, hidden_size)))
        self.encoder_layers.append(nn.ReLU())
        for i in range(layers):
            self.encoder_layers.append(nn.Linear(hidden_size, hidden_size))
            self.encoder_layers.append(nn.ReLU())
        self.encoder_layers.append(nn.Linear(hidden_size, output_size))

        self.encoder = nn.Sequential(*self.encoder_layers)

    def forward(self, x):
        return self.encoder(x)


def train_regression_network(
    model: nn.Module,
    train_set_fn,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    epochs,
    history,
    device,
    batch_size=128,
):
    model.train()
    criterion = nn.MSELoss().to(device=device)
    history["train_loss"] = history.get("train_loss", [])
    for epoch in range(epochs):
        epoch_loss = torch.zeros((), requires_grad=False, device=device)
        epoch_length = 100
        # for now, define a epoch as 100 iterations, because
        # we have no train set size yet.
        for i in range(epoch_length):
            X, Y = train_set_fn(batch_size)

            X = torch.tensor(
                X.reshape(batch_size, -1), device=device, dtype=torch.float32
            )
            Y = torch.tensor(Y, device=device, dtype=torch.float32)
            optim.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                output = model.forward(X)
                loss = criterion(output,Y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
            epoch_loss += loss.mean()
        avg_epoch_loss = (epoch_loss / epoch_length).item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}", end="\r")
        history["train_loss"].append(avg_epoch_loss)
    return
