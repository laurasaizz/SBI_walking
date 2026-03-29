import torch
import torch.nn as nn
from typing import Optional
from nflows.transforms.splines.rational_quadratic import rational_quadratic_spline
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rational_quadratic_spline_extrapolated(
    inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
    inverse=False, left=-3.0, right=3.0, bottom=-3.0, top=3.0,
    min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3,
):
    # identify which inputs are inside the valid spline region
    if inverse:
        inside_mask = (inputs >= bottom) & (inputs <= top)
    else:
        inside_mask = (inputs >= left) & (inputs <= right)

    # clamp inputs so the spline never receives out-of-bounds values
    # (safe because we will overwrite OOB outputs with identity anyway)
    inputs_clamped = inputs.clone()
    inputs_clamped[~inside_mask] = (left + right) / 2.0 

    outputs, logabsdet = rational_quadratic_spline(
        inputs=inputs_clamped,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        left=left, right=right, bottom=bottom, top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    # overwrite OOB positions with identity and zero log-det
    outputs[~inside_mask] = inputs[~inside_mask].to(outputs.dtype)
    logabsdet[~inside_mask] = 0.0

    return outputs, logabsdet

class SummaryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=400, layers=4, output_size=16):
        super().__init__()

        modules = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(layers):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_size, output_size))

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class InvertibleLinear(nn.Module):
    """Learned invertible 1x1 'convolution' (linear map) with LU decomposition."""
    def __init__(self, dim):
        super().__init__()
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        # LU decomposition for stable log-det
        P, L, U = torch.linalg.lu(Q)
        self.register_buffer('P', P)  # fixed permutation
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.log_diag_U = nn.Parameter(torch.log(U.diag().abs()))

    def _get_W(self):
        L = torch.tril(self.L, -1) + torch.eye(self.L.shape[0], device=self.L.device)
        U = torch.triu(self.U, 1) + torch.diag(self.log_diag_U.exp())
        return self.P @ L @ U

    def forward(self, x):
        W = self._get_W()
        return x @ W.T, self.log_diag_U.sum()   # (z, logdet)

    def reverse(self, z):
        W = self._get_W()
        return z @ torch.linalg.inv(W).T


class ActNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                self.bias.data = -x.mean(dim=0)
                self.log_scale.data = -x.std(dim=0).log()
            self.initialized = True
        
        z = (x + self.bias) * self.log_scale.exp()
        logabsdet = self.log_scale.sum().expand(x.shape[0])
        return z, logabsdet

    def inverse(self, z):
        return z * (-self.log_scale).exp() - self.bias

class SplineCouplingBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        condition_size: int = 0,
        num_bins: int = 8,
        tail_bound: float = 3.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        super().__init__()

        self.input_size = input_size
        self.condition_size = condition_size
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.num_identity = input_size - input_size // 2
        self.num_transform = input_size // 2

        # For each transformed dimension we need:
        # num_bins widths + num_bins heights + (num_bins + 1) derivatives
        # This matches nflows' rational_quadratic_spline convention.
        self.params_per_dim = 3 * num_bins + 1

        self.param_net = nn.Sequential(
            nn.Linear(self.num_identity + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_transform * self.params_per_dim),
        )

        self.block_det = None  # will store per-sample logdet

    def _get_spline_params(self, x1_cond: torch.Tensor):
        batch_size = x1_cond.shape[0]
        params = self.param_net(x1_cond)
        params = params.view(batch_size, self.num_transform, self.params_per_dim)

        unnormalized_widths = params[..., :self.num_bins]
        unnormalized_heights = params[..., self.num_bins:2 * self.num_bins]
        unnormalized_derivatives = params[..., 2 * self.num_bins:]

        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = torch.ones((0,), device=device)
    ):
        x1, x2 = torch.split(
            x,
            [self.num_identity, self.num_transform],
            dim=1
        )
        
        if x2.min() < -self.tail_bound or x2.max() > self.tail_bound:
            print(f"OUT OF BOUNDS: x2 min={x2.min():.4f}, max={x2.max():.4f}, tail_bound=±{self.tail_bound}")

        x1_cond = torch.cat([x1, y], dim=1)
        uw, uh, ud = self._get_spline_params(x1_cond)

        z2, logabsdet = rational_quadratic_spline_extrapolated(
            inputs=x2,
            unnormalized_widths=uw,
            unnormalized_heights=uh,
            unnormalized_derivatives=ud,
            inverse=False,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        z = torch.cat([x1, z2], dim=1)
        self.block_det = logabsdet.sum(dim=1)  # per sample
        return z

    def reverse(self, z: torch.Tensor, y: torch.Tensor):
        z1, z2 = torch.split(
            z,
            [self.num_identity, self.num_transform],
            dim=1
        )

        z1_cond = torch.cat([z1, y], dim=1)
        uw, uh, ud = self._get_spline_params(z1_cond)

        x2, _ = rational_quadratic_spline_extrapolated(
            inputs=z2,
            unnormalized_widths=uw,
            unnormalized_heights=uh,
            unnormalized_derivatives=ud,
            inverse=True,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )

        x = torch.cat([z1, x2], dim=1)
        return x


class SplineFlow(nn.Module):
    def __init__(self, input_size, hidden_size, blocks,
                 condition_size=0, num_bins=8, tail_bound=3.0):
        super().__init__()
        self.input_size = input_size
        self.condition_size = condition_size
        self.tail_bound = tail_bound

        self.coupling_blocks = nn.ModuleList()
        self.linear_blocks = nn.ModuleList()   # replaces rotation matrices
        self.actnorm_layers = nn.ModuleList()   

        for _ in range(blocks - 1):
            self.coupling_blocks.append(SplineCouplingBlock(
                input_size, hidden_size, condition_size, num_bins, tail_bound))
            self.linear_blocks.append(InvertibleLinear(input_size))
            self.actnorm_layers.append(ActNorm(input_size))  

        self.coupling_blocks.append(SplineCouplingBlock(
            input_size, hidden_size, condition_size, num_bins, tail_bound))

    def forward(self, x, y=torch.ones((0,), device=device)):
        total_logdet = 0.0
        for block, linear, actnorm in zip(self.coupling_blocks, self.linear_blocks, self.actnorm_layers):
            x = block(x, y)
            total_logdet += block.block_det
            x, ld = linear(x)
            total_logdet += ld
            x, ld = actnorm(x)
            total_logdet += ld
        x = self.coupling_blocks[-1](x, y)
        total_logdet += self.coupling_blocks[-1].block_det
        return x, total_logdet

    def reverse(self, z, y=torch.ones((0,), device=device)):
        x = self.coupling_blocks[-1].reverse(z, y)
        for block, linear, actnorm in zip(
                reversed(self.coupling_blocks[:-1]),
                reversed(self.linear_blocks),
                reversed(self.actnorm_layers)):
            x = actnorm.inverse(x)
            x = linear.reverse(x)
            x = block.reverse(x, y)
        return x

    def sample(self, num_samples: int, conditions: Optional[torch.Tensor] = None):
        effective_num_samples = num_samples if conditions is None else num_samples * conditions.size(0)
        gaussians = torch.randn(effective_num_samples, self.input_size, device=device)

        if conditions is not None:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
        else:
            if self.condition_size == 0:
                conditions = torch.ones((num_samples, 0), device=device)
            else:
                raise ValueError("Conditional model requires explicit conditions.")

        return self.reverse(gaussians, conditions)
    
class SplineFlowSummary(nn.Module):
    def __init__(
        self,
        foot_input_size,
        theta_dim=8,
        condition_size=16,
        s_hidden=512,
        s_layers=4,
        f_hidden=256,
        f_blocks=6,
        num_bins=8,
        tail_bound=3.0,
    ):
        super().__init__()

        self.summary = SummaryNetwork(
            input_size=foot_input_size,
            hidden_size=s_hidden,
            layers=s_layers,
            output_size=condition_size,
        )

        self.flow = SplineFlow(
            input_size=theta_dim,
            hidden_size=f_hidden,
            blocks=f_blocks,
            condition_size=condition_size,
            num_bins=num_bins,
            tail_bound=tail_bound,
        )

    def forward(self, x, theta):
        hx = self.summary(x)
        z, logdet = self.flow(theta, hx)
        return z, logdet

    def reverse(self, z, x):
        hx = self.summary(x)
        theta = self.flow.reverse(z, hx)
        return theta

    def sample(self, x, n):
        hx = self.summary(x)
        return self.flow.sample(n, hx)

#----------Loss and training function---------------------------
import math
def calculate_loss(z, logdet):
    D = z.shape[1]
    log_base = 0.5 * z.pow(2).sum(dim=1) + 0.5 * math.log(2 * math.pi) * D
    return (log_base - logdet).mean()

def train_inn_cond(model, train_dataset, test_dataset, optim, epochs,
                   batch_size=128, shuffle=True, lr_scheduler=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, persistent_workers=True)

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    history = {"train_loss": [], "test_loss": []}
    pbar = tqdm(range(epochs), desc="Training", leave=True)

    for epoch in pbar:
        model.train()
        train_epoch_loss = 0.0

        for X, theta in train_loader:
            X, theta = X.to(device), theta.to(device)
            optim.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                z, logdet = model(X, theta)
                loss = calculate_loss(z, logdet)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_epoch_loss += loss.item() * X.size(0)

        train_epoch_loss /= len(train_dataset)

        model.eval()
        test_epoch_loss = 0.0

        with torch.no_grad():
            for X, theta in test_loader:
                X, theta = X.to(device), theta.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    z, logdet = model(X, theta)
                    loss = calculate_loss(z, logdet)
                test_epoch_loss += loss.item() * X.size(0)

        test_epoch_loss /= len(test_dataset)
        history["train_loss"].append(train_epoch_loss)
        history["test_loss"].append(test_epoch_loss)
        pbar.set_postfix(train_loss=train_epoch_loss, test_loss=test_epoch_loss)

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(test_epoch_loss)
            else:
                lr_scheduler.step()

    return history





class SplineCouplingBlock3D(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size=0,
                 num_bins=8, tail_bound=3.0, min_bin_width=1e-3,
                 min_bin_height=1e-3, min_derivative=1e-3, split_idx=0):
        super().__init__()
        self.input_size = input_size
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.num_transform = 1
        self.num_identity = input_size - 1
        self.params_per_dim = 3 * num_bins + 1

        perm = list(range(input_size))
        perm.remove(split_idx)
        perm.append(split_idx)
        self.register_buffer("perm", torch.tensor(perm))
        inv_perm = [0] * input_size
        for i, p in enumerate(perm):
            inv_perm[p] = i
        self.register_buffer("inv_perm", torch.tensor(inv_perm))

        self.param_net = nn.Sequential(
            nn.Linear(self.num_identity + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_transform * self.params_per_dim),
        )
        self.block_det = None

    def _get_spline_params(self, x1_cond):
        batch_size = x1_cond.shape[0]
        params = self.param_net(x1_cond)
        params = params.view(batch_size, self.num_transform, self.params_per_dim)
        return (params[..., :self.num_bins],
                params[..., self.num_bins:2*self.num_bins],
                params[..., 2*self.num_bins:])

    def forward(self, x, y=None):
        if y is None:
            y = torch.ones((x.shape[0], 0), device=x.device)
        x_perm = x[:, self.perm]
        x1, x2 = x_perm[:, :-1], x_perm[:, -1:]
        uw, uh, ud = self._get_spline_params(torch.cat([x1, y], dim=1))
        z2, logabsdet = rational_quadratic_spline_extrapolated(
            inputs=x2, unnormalized_widths=uw, unnormalized_heights=uh,
            unnormalized_derivatives=ud, inverse=False,
            left=-self.tail_bound, right=self.tail_bound,
            bottom=-self.tail_bound, top=self.tail_bound,
            min_bin_width=self.min_bin_width, min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        z_perm = torch.cat([x1, z2], dim=1)
        self.block_det = logabsdet.sum(dim=1)
        return z_perm[:, self.inv_perm]

    def reverse(self, z, y=None):
        if y is None:
            y = torch.ones((z.shape[0], 0), device=z.device)
        z_perm = z[:, self.perm]
        z1, z2 = z_perm[:, :-1], z_perm[:, -1:]
        uw, uh, ud = self._get_spline_params(torch.cat([z1, y], dim=1))
        x2, _ = rational_quadratic_spline_extrapolated(
            inputs=z2, unnormalized_widths=uw, unnormalized_heights=uh,
            unnormalized_derivatives=ud, inverse=True,
            left=-self.tail_bound, right=self.tail_bound,
            bottom=-self.tail_bound, top=self.tail_bound,
            min_bin_width=self.min_bin_width, min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        return torch.cat([z1, x2], dim=1)[:, self.inv_perm]


class SplineFlow3D(nn.Module):
    def __init__(self, input_size, hidden_size, blocks,
                 condition_size=0, num_bins=8, tail_bound=3.0):
        super().__init__()
        self.input_size = input_size
        self.condition_size = condition_size

        self.coupling_blocks = nn.ModuleList()
        self.linear_blocks = nn.ModuleList()

        for i in range(blocks):
            self.coupling_blocks.append(SplineCouplingBlock3D(
                input_size=input_size, hidden_size=hidden_size,
                condition_size=condition_size, num_bins=num_bins,
                tail_bound=tail_bound, split_idx=i % input_size,
            ))
            if i < blocks - 1:
                self.linear_blocks.append(InvertibleLinear(input_size))  

    def forward(self, x, y=None):
        total_logdet = torch.zeros(x.shape[0], device=x.device)
        for i, block in enumerate(self.coupling_blocks):
            x = block(x, y)
            total_logdet += block.block_det                 
            if i < len(self.linear_blocks):
                x, ld = self.linear_blocks[i](x)
                total_logdet += ld                          
        return x, total_logdet                               
    
    def reverse(self, z, y=None):
        for i in range(len(self.coupling_blocks) - 1, -1, -1):
            if i < len(self.linear_blocks):
                z = self.linear_blocks[i].reverse(z)        
            z = self.coupling_blocks[i].reverse(z, y)
        return z

    def sample(self, num_samples, conditions=None):
        device = next(self.parameters()).device
        if conditions is not None:
            effective = num_samples * conditions.shape[0]
            conditions = conditions.repeat_interleave(num_samples, dim=0)
        else:
            effective = num_samples
            conditions = torch.ones((effective, 0), device=device)
        z = torch.randn(effective, self.input_size, device=device)
        return self.reverse(z, conditions)


class SplineFlowWindow(nn.Module):
    def __init__(self, window_dim, hidden_size=256, blocks=4, num_bins=8, tail_bound=3.0):
        super().__init__()
        self.flow = SplineFlow3D(
            input_size=3, hidden_size=hidden_size, blocks=blocks,
            condition_size=window_dim, num_bins=num_bins, tail_bound=tail_bound,
        )

    def forward(self, window, com):         
        z, logdet = self.flow(com, window)
        return z, logdet                    
    def reverse(self, z, window):
        return self.flow.reverse(z, window)

    def sample(self, window, n=1):
        return self.flow.sample(n, window)

class SplineFlowWindowEncoder(nn.Module):
    def __init__(self, window_dim=90, hidden_size=256, blocks=4, num_bins=10, tail_bound=6.0):
        super().__init__()
        
        self.window_encoder = nn.Sequential(
            nn.Linear(window_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        self.flow = SplineFlow3D(
            input_size=3,
            hidden_size=hidden_size,
            blocks=blocks,
            condition_size=128,
            num_bins=num_bins,
            tail_bound=tail_bound,
        )
    
    def forward(self, x, y=None):
        x_enc = self.window_encoder(x)
        z = self.flow(y, x_enc)
        return z
    
    def reverse(self, z, x):
        x_enc = self.window_encoder(x)
        return self.flow.reverse(z, x_enc)
    
    def sample(self, x, n=1):
        x_enc = self.window_encoder(x)
        return self.flow.sample(n, x_enc)
    
import math

def calculate_loss3D(z, logdet):
    D = z.shape[1]
    log_base = 0.5 * z.pow(2).sum(dim=1) + 0.5 * math.log(2 * math.pi) * D
    return (log_base - logdet).mean()


def train_inn_cond3D(
    model,
    train_dataset,
    test_dataset,
    optim,
    epochs,
    batch_size=128,
    shuffle=True,
    lr_scheduler=None,
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, persistent_workers=True)

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    history = {"train_loss": [], "test_loss": []}
    pbar = tqdm(range(epochs), desc="Training", leave=True)

    for epoch in pbar:
        model.train()
        train_epoch_loss = 0.0

        for X, theta in train_loader:
            X, theta = X.to(device), theta.to(device)
            optim.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                z, logdet = model(X, theta)          # unpack tuple
                loss = calculate_loss3D(z, logdet)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            train_epoch_loss += loss.item() * X.size(0)

        train_epoch_loss /= len(train_dataset)

        model.eval()
        test_epoch_loss = 0.0

        with torch.no_grad():
            for X, theta in test_loader:
                X, theta = X.to(device), theta.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    z, logdet = model(X, theta)      # unpack tuple
                    loss = calculate_loss3D(z, logdet)
                test_epoch_loss += loss.item() * X.size(0)

        test_epoch_loss /= len(test_dataset)
        history["train_loss"].append(train_epoch_loss)
        history["test_loss"].append(test_epoch_loss)
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'train_loss': train_epoch_loss,
                'test_loss': test_epoch_loss,
                'history': history,
            }
            torch.save(checkpoint, f'checkpointwindow_epoch_{epoch + 1}.pt')
        pbar.set_postfix(train_loss=train_epoch_loss, test_loss=test_epoch_loss)

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(test_epoch_loss)
            else:
                lr_scheduler.step()

    return history