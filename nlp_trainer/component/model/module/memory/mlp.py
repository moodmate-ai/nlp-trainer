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
        x: torch.Tensor,
        value: Optional[torch.Tensor] = None,
    ):
        """
        x: could be key or query
        - used for expecting value
        
        if value is given, then returns surprise, which can be used for updating memory parameter.
        """
        expected_value = self.neural_net(x)
        if value is None:
            return expected_value
        
        surprise = torch.norm(expected_value - value, p=2, dim=-1)
        surprise = surprise.sum()
        return surprise
        
