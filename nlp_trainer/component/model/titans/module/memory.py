from torch import nn, torch
from typing import Optional


class LongTermMemory(nn.Module):
    """
    This is a key-value neural network memory store.

    Retrieve
    - query -> output

    Update
    - key, output -> update

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        hidden_dim: int,
    ):
        super(LongTermMemory, self).__init__()

        self.neural_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)

        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001
        )
        ## 결국 SGD에서 momentum과 weight decay를 쓰는건,
        ## past surprise 반영과
        ## forgetting mechanism 사용하는것과 동일

    def retrieve_memory(self, query_seq: torch.Tensor) -> torch.Tensor:
        query_seq = self.w_q(query_seq)
        output = self.neural_net(query_seq)
        return output

    def update_memory(self, kv_seq: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            kv_seq = kv_seq.detach()

            self.optimizer.zero_grad()

            k = self.w_k(kv_seq)
            v = self.w_v(kv_seq)

            output = self.neural_net(k)
            surprise = torch.norm(output - v, p=2, dim=-1)
            surprise = surprise.sum()
            surprise.backward()

            self.optimizer.step()

        return self.retrieve_memory(kv_seq)

    def forward(self, x: torch.Tensor, is_update: bool = True) -> torch.Tensor:
        if is_update:
            return self.update_memory(x)
        else:
            return self.retrieve_memory(x)


class MultiHeadMemory(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super(MultiHeadMemory, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

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
        expected_value = self.network(q)

        if v is None:
            return expected_value

        memory_loss = torch.norm(expected_value - v, p=2, dim=-1)
        memory_loss = memory_loss.sum()

        return memory_loss
