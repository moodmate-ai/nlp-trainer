import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_dim, eps=1e-05):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.eps = eps

        self.r = torch.nn.Parameter(torch.ones(hidden_dim))
        self.b = torch.nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        result = x - torch.mean(x, dim=-1, keepdim=True)
        result /= torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps
        )
        result = result * self.r + self.b

        assert x.shape == result.shape

        return result
