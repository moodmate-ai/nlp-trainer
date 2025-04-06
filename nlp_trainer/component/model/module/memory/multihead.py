from torch import nn, torch
from typing import Optional


class MultiHeadMemory(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super(MultiHeadMemory, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.neural_network = nn.Sequential()

    def forward(
        self,
        q: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ):
        """
        returns expected value
        """
        assert len(q.shape) == 4
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HIDDEN_DIM = q.shape

        assert NUM_HEADS == HIDDEN_DIM
        expected_value = self.neural_network(q)

        if v is None:
            return expected_value

        memory_loss = torch.norm(expected_value - v, p=2, dim=-1)
        memory_loss = memory_loss.sum()

        return memory_loss
