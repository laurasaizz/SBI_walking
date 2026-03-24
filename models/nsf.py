import torch
import torch.nn as nn
from typing import Optional
from nflows.transforms.splines.rational_quadratic import rational_quadratic_spline

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_onehot(y, num_classes: int = 2):
    return torch.nn.functional.one_hot(y, num_classes=num_classes)


def get_orthonormal_matrix(d: int) -> torch.Tensor:
    A = torch.randn((d, d))
    Q, _ = torch.linalg.qr(A)
    return Q.to(device)


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

        x1_cond = torch.cat([x1, y], dim=1)
        uw, uh, ud = self._get_spline_params(x1_cond)

        z2, logabsdet = rational_quadratic_spline(
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

        x2, _ = rational_quadratic_spline(
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
    condition_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        blocks: int,
        condition_size: int = 0,
        num_bins: int = 8,
        tail_bound: float = 3.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.condition_size = condition_size
        self.blocks = nn.ModuleList()
        self.rotation_matrices = []

        for _ in range(blocks - 1):
            self.blocks.append(
                SplineCouplingBlock(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    condition_size=condition_size,
                    num_bins=num_bins,
                    tail_bound=tail_bound,
                )
            )
            R = get_orthonormal_matrix(input_size)
            R.requires_grad = False
            self.rotation_matrices.append(R)

        self.blocks.append(
            SplineCouplingBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                condition_size=condition_size,
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
        )

    def forward(self, x, y=torch.ones((0,), device=device)):
        for block, R in zip(self.blocks, self.rotation_matrices):
            x = block(x, y)
            x = x @ R
        x = self.blocks[-1](x, y)
        return x

    def reverse(self, z: torch.Tensor, y: torch.Tensor = torch.ones((0,), device=device)):
        x = self.blocks[-1].reverse(z, y)
        for block, R in zip(reversed(self.blocks[:-1]), reversed(self.rotation_matrices)):
            x = x @ R.T
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
        z = self.flow(theta, hx)
        return z

    def reverse(self, z, x):
        hx = self.summary(x)
        theta = self.flow.reverse(z, hx)
        return theta

    def sample(self, x, n):
        hx = self.summary(x)
        return self.flow.sample(n, hx)