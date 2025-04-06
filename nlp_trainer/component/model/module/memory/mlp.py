from torch import nn, torch
from typing import Optional


class TitansMlpMemory(nn.Module):
    """
    Memory module which implements Titans Neural Memory
    
    The concepts of the memory are
    - returning value if key is given
    - update parameter using surprise metric
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        ff_dim: int,
    ):
        super(TitansMlpMemory, self).__init__()
        assert num_layers > 1, "num_layers must be greater than 1"
        
        self.neural_net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            *[
                nn.Sequential(
                    nn.Linear(ff_dim, ff_dim),
                    nn.GELU(),
                )
                for _ in range(num_layers - 2)
            ],
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(
        self, 
        x: torch.Tensor
    ):
        """
        x: could be key or query
        - used for expecting value
        """
        expected_value = self.neural_net(x)
        
        return expected_value
        
