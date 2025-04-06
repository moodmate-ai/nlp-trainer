import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Given a tensor of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM]
    applies Rotary Positional Encoding.
    offset allows to apply rotary to sequnce part by part by telling how much tokens preecede the input in the sequence.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        ## Theta := theta^( -(2i / head_dim) ) where i = 0, 1, 2, ..., head_dim / 2
        self.theta = (
            1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / head_dim))
        )[None, :]  # [1, head_dim / 2]

        rot_seq = max_seq_len
        m_theta = torch.arange(rot_seq)[:, None].float()  # [max_seq_len, 1]
        m_theta = (m_theta @ self.theta)[
            :, :, None, None
        ]  # [max_seq_len, head_dim / 2, 1, 1]

        m_sin = m_theta.sin()
        m_cos = m_theta.cos()

        row0 = torch.cat((m_cos, -m_sin), dim=-1)  # [max_seq_len, head_dim / 2, 1, 2]
        row1 = torch.cat((m_sin, m_cos), dim=-1)  # [max_seq_len, head_dim / 2, 1, 2]

        self.rotation_matrix = torch.cat((row0, row1), dim=-2)[None, :, None, :, :, :]
        """
        [1, max_seq_len, 1, head_dim / 2, 2, 2]
        """

    def forward(self, x, offset: int = 0):
        assert len(x.shape) == 4, f"x.shape: {x.shape}"
        # torch tensor of shape [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM]
        assert offset >= 0
        
        ## reshape
        BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM = x.shape
        y = x.reshape(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM // 2, 2, 1)

        ## rotate
        start, end = offset, offset + SEQ_LEN
        
        y = self.rotation_matrix[:, start:end].to(x.device) @ y

        ## reshape
        y = y.reshape(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM)

        assert y.shape == x.shape

        return y
