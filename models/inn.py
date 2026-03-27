# adapted from our solution lecture exercise 4
import torch
import torch.nn as nn
from typing import Optional
from torch.nn.functional import one_hot
from .regressionNetwork import RegressionNetwork

from torch.amp.grad_scaler import GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_orthonormal_matrix(d: int) -> torch.Tensor:
    A = torch.randn((d, d))
    Q, _R = torch.linalg.qr(A)
    return Q.to(device)


class CouplingBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, condition_size: int = 0,drop = 0.1, invert_rounding = True):
        super(CouplingBlock, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.1],device=device))
        self.a.requires_grad=False  # a is hyperparam
        
        self.unchanged_size = input_size // 2 if invert_rounding else (input_size - input_size // 2)  # x1 size   # is half input_size, rounded down
        self.changed_size = input_size - self.unchanged_size # x2 size  # half, rounded up

        self.scale_net = nn.Sequential(
            nn.Linear(self.unchanged_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.changed_size),
            nn.Tanh(),
            # remember to exp this
        )
        self.cond_drop = nn.Dropout(drop)
        self.t_net = nn.Sequential(
            nn.Linear(self.unchanged_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.changed_size),
        )
        # self.block_det = 0

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = torch.ones((0,), device=device)
    ):
        # assume x is of shape (batch_size, input_size)
        x1, x2 = torch.split(x, [self.unchanged_size, self.changed_size], dim=1)
        # creates views!
        y_drop = y #self.cond_drop(y)
        x1_cond = torch.cat([x1, y_drop], dim=1)
        scaled_x1 = self.scale_net(x1_cond) * self.a
        z2 = x2 * torch.exp(scaled_x1) + self.t_net(x1_cond)
        z = torch.cat([x1, z2], dim=1)
        block_det = torch.sum(scaled_x1,dim=1)
        return z, block_det

    def reverse(self, z, y):
        z1, z2 = torch.split(z, [self.unchanged_size, self.changed_size], dim=1)
        x1 = torch.cat([z1, y], dim=1)
        x2 = (z2 - self.t_net(x1)) / torch.exp(self. a * self.scale_net(x1))
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
        self.n_blocks = blocks
        self.blocks: nn.ModuleList[CouplingBlock] = nn.ModuleList()
        self.rotation_matrices = nn.ParameterList([])
        for i in range(blocks): # blocks -1 
            self.blocks.append(CouplingBlock(input_size, hidden_size, condition_size, invert_rounding= i %2 ==0))
            R = get_orthonormal_matrix(input_size)  # R means Rotational matrix,
            # could also be named Q.
            R.requires_grad = False  # prof said its not worth to learn (probably)
            # but it seams to decrease train loss. # but reconstruction is not possible if R becomes not Orthonormal
            self.rotation_matrices.append(R)
        
        #!TODO: readd
        #self.blocks.append(CouplingBlock(input_size, hidden_size, condition_size))

    def forward(self, x, y=torch.ones((0,), device=device)):
        """
        y should be a onehot
        """
        sum_log_det = 0
        for i in range(self.n_blocks):
            x,log_det = self.blocks[i].forward(x, y)
            x = x @ self.rotation_matrices[i]
            sum_log_det = sum_log_det + log_det
        
        return x , sum_log_det

    def reverse(self, z: torch.Tensor, y: torch.Tensor = torch.ones((0,))):
        #!TODO: readd
        x = z
        # x = self.blocks[-1].reverse(z, y)
        for block, R in zip(
            reversed(self.blocks), reversed(self.rotation_matrices) # self.blocks[:-1]
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

class RealNVPSummary(nn.Module):
    def __init__(self,input_size = 200, condition_size = 100 , reduced_condition_size = 8,s_hidden = 1000,s_layers=6, r_hidden = 500,r_blocks=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary = RegressionNetwork(condition_size,hidden_size= s_hidden, layers=s_layers, output_size=reduced_condition_size)
        self.realNVP = RealNVP(input_size=input_size,hidden_size=r_hidden,blocks=r_blocks,condition_size=reduced_condition_size)

    def forward(self,x,y = None):
        """
        x is observation, y is [lam,mu,i0]
        """
        hy = self.summary(y)
        codes,log_det = self.realNVP.forward(x,hy)
        return codes, log_det
    def reverse(self,z,y):
        hy = self.summary(y)
        x = self.realNVP.reverse(z,hy)
        return x
    
    def sample(self,x,n):
        """
        samples n per condition
        """
        hx = self.summary(x)
        return self.realNVP.sample(n,hx)


class CouplingBlockSingle(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, condition_size: int = 0):
        super(CouplingBlockSingle, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
            # remember to exp this
        )
        self.cond_drop = nn.Dropout(0.1)
        self.t_net = nn.Sequential(
            nn.Linear(condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        self.block_det = 0

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = torch.ones((0,), device=device)
    ):
        # assume x is of shape (batch_size, input_size)
    
        y_drop = self.cond_drop(y)

        
        scaled_x1 = self.scale_net(y_drop)
        x2 = x * torch.exp(scaled_x1) + self.t_net(y_drop)
        self.block_det = torch.sum(scaled_x1)
        return x2

    def reverse(self, z, y):
        x = (z - self.t_net(y)) / torch.exp(self.scale_net(y))
        return x

class RealNVPsingle(nn.Module):
    condition_size: int

    def __init__(
        self, input_size: int = 1, hidden_size: int = 32, blocks: int = 8, condition_size: int = 0
    ):
        super(RealNVPsingle, self).__init__()
        self.input_size = input_size
        self.condition_size = condition_size
        self.blocks: nn.ModuleList[CouplingBlockSingle] = nn.ModuleList() #TODO:
        for i in range(blocks):
            self.blocks.append(CouplingBlockSingle(input_size, hidden_size, condition_size))


    def forward(self, x, y=torch.ones((0,), device=device)):
        z = x 
        for block in self.blocks: 
            block: CouplingBlockSingle
            z = block.forward(z, y)
        return z

    def reverse(self, z: torch.Tensor, y: torch.Tensor = torch.ones((0,))):
        x = z
        for block in reversed(self.blocks):
            block: CouplingBlockSingle
            x = block.reverse(z = x, y = y)
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


class RealNVPSummarySingle(nn.Module):
    def __init__(self,input_size = 1, condition_size = 100 , reduced_condition_size = 8,s_hidden = 1000,s_layers=6, r_hidden = 500,r_blocks=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary = RegressionNetwork(condition_size,hidden_size= s_hidden, layers=s_layers, output_size=reduced_condition_size)
        self.realNVP = RealNVPsingle(input_size=input_size,hidden_size=r_hidden,blocks=r_blocks,condition_size=reduced_condition_size)

    def forward(self,x,y = None):
        """
        x is observation, y is [lam,mu,i0]
        """
        hy = self.summary(y)
        codes = self.realNVP(x,hy)
        return codes
    def reverse(self,z,y):
        hy = self.summary(y)
        x = self.realNVP.reverse(z,hy)
        return x
    
    def sample(self,x,n):
        """
        samples n per condition
        """
        hx = self.summary(x)
        return self.realNVP.sample(n,hx)