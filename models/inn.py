# adapted from our solution lecture exercise 4
import torch
import torch.nn as nn
from typing import Optional
from torch.nn.functional import one_hot

import torch.nn.utils as utils

from torch.amp.grad_scaler import GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_orthonormal_matrix(d: int) -> torch.Tensor:
    A = torch.randn((d, d))
    Q, _R = torch.linalg.qr(A)
    return Q.to(device)


class CouplingBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, condition_size: int = 0):
        super(CouplingBlock, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_size - input_size // 2 + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size // 2),
            nn.Tanh(),
            # remember to exp this
        )
        self.t_net = nn.Sequential(
            nn.Linear(input_size - input_size // 2 + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size // 2),
        )
        self.block_det = 0

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = torch.ones((0,), device=device)
    ):
        # assume x is of shape (batch_size, input_size)
        x1, x2 = torch.split(x, [x.size(1) - x.size(1) // 2, x.size(1) // 2], dim=1)
        # creates views!
        x1_cond = torch.cat([x1, y], dim=1)
        scaled_x1 = self.scale_net(x1_cond)
        z2 = x2 * torch.exp(scaled_x1) + self.t_net(x1_cond)
        # could be exp(a * ...), but also for backwards.
        # where a is hyperparam
        z = torch.cat([x1, z2], dim=1)
        # if self.training:
        # # just always compute, because we need it for loss calculation
        # its not necesarry for inference, but we only do training and testing
        self.block_det = torch.sum(scaled_x1)
        return z

    def reverse(self, z, y):
        z1, z2 = torch.split(z, [z.size(1) - z.size(1) // 2, z.size(1) // 2], dim=1)
        x1 = torch.cat([z1, y], dim=1)
        x2 = (z2 - self.t_net(x1)) / torch.exp(self.scale_net(x1))
        x = torch.cat([z1, x2], dim=1)
        return x


class RealNVP(nn.Module):
    condition_size: int

    def __init__(
        self, input_size: int, hidden_size: int, blocks: int, condition_size: int = 0
    ):
        super(RealNVP, self).__init__()
        self.input_size = input_size
        self.condition_size = condition_size
        self.blocks: nn.ModuleList[CouplingBlock] = nn.ModuleList()
        self.rotation_matrices = []
        for i in range(blocks - 1):
            self.blocks.append(CouplingBlock(input_size, hidden_size, condition_size))
            R = get_orthonormal_matrix(input_size)  # R means Rotational matrix,
            # could also be named Q.
            R.requires_grad = False  # prof said its not worth to learn (probably)
            # but it seams to decrease train loss. # but reconstruction is not possible if R becomes not Orthonormal
            self.rotation_matrices.append(R)

        #!TODO: readd
        self.blocks.append(CouplingBlock(input_size, hidden_size, condition_size))

    def forward(self, x, y=torch.ones((0,), device=device)):
        """
        y should be a onehot
        """
        for block, R in zip(
            self.blocks, self.rotation_matrices
        ):  # pyright: ignore[reportAssignmentType]
            block: CouplingBlock
            # last from blocks will not be used in this loop
            x = block.forward(x, y)
            x = x @ R  # rotation
        #!TODO: readd
        x = self.blocks[-1].forward(x, y)
        return x

    def reverse(self, z: torch.Tensor, y: torch.Tensor = torch.ones((0,))):
        #!TODO: readd
        # x = z
        x = self.blocks[-1].reverse(z, y)
        for block, R in zip(
            reversed(self.blocks[:-1]), reversed(self.rotation_matrices)
        ):
            block: CouplingBlock
            x = x @ R.T
            x = block.reverse(x, y)
        return x

    def sample(self, num_samples: int, conditions: Optional[torch.Tensor] = None):
        effective_num_samples = (
            num_samples if conditions is None else num_samples * conditions.size(0)
        )
        gaussians = torch.randn(effective_num_samples, self.input_size, device=device)
        if conditions is not None:  # conditions is provided
            conditions = conditions.repeat(num_samples, 1)
        else:  # conditions is not provided
            if self.condition_size == 0:  # stack empty tensor
                conditions = torch.ones((num_samples, 0), device=device)
            else:  # is a conditional model, but no classes were supplied, so draw random
                conditions = torch.randint(
                    0, self.condition_size, (num_samples,), device=device
                )
        return self.reverse(gaussians, conditions)


def calculate_loss(model, output):
    loss = torch.zeros((), requires_grad=True, device=device)
    for block in model.realNVP.blocks:
        block: CouplingBlock
        loss -= block.block_det
    loss += output.pow(2).sum() / 2  # sum over all axis
    return loss / output.size(0)  # mean over batch



std_tensor = torch.tensor([...], device=device).reshape((1, 3))
mean_tensor = torch.tensor([...], device=device).reshape((1, 3))


def y_scaling(y: torch.Tensor):
    # y shape: bs x 3
    return (y - mean_tensor) / std_tensor


def y_unscale(y_scaled: torch.Tensor):
    return y_scaled * std_tensor + mean_tensor


def train_inn_cond(
    model: nn.Module,
    train_set_fn,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    lr_scheduler :torch.optim.lr_scheduler.LRScheduler,
    epochs,
    history,
    batch_size=128,
):
    """
    model: A inn
    train_set_fn: a function that returns X,Y
    with X: the condition (like observations that will be available at inference time)
    Y: the hidden parameter that we want a posterior for later.

    # scaler = GradScaler("cuda", enabled=(device.type == 'cuda'))
    """
    history["train_loss"] = history.get("train_loss", [])
    model.train()

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
                output = model.forward(X, Y)
                # x is used for condition, y is input (see docstring)
                loss = calculate_loss(model, output)
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
