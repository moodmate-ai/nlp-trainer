from nlp_trainer.component.model.module.attention.multihead import MultiHeadAttention
import torch
import pytest

@pytest.fixture
def mock_input_size():
    return {
        "batch_size" : 2,
        "seq_length" : 3,
        "memory_length" : 3,
        "cache_length": 12,
        "num_heads" : 5,
        "head_dim" : 7
    }


@pytest.fixture
def mock_inputs(mock_input_size):
    
    size = (
        mock_input_size["batch_size"],
        mock_input_size["num_heads"],
        mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["head_dim"]
    )
    size_w_cache = (
        mock_input_size["batch_size"],
        mock_input_size["num_heads"],
        mock_input_size["cache_length"] + mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["head_dim"]
    )
    
    q = torch.rand(size)
    k = torch.rand(size_w_cache)
    v = torch.rand(size_w_cache)
    
    yield q, k, v, mock_input_size["memory_length"]
    
@pytest.fixture
def mock_model(mock_input_size):
    model = MultiHeadAttention(
        num_heads=mock_input_size["num_heads"],
        hidden_dim=mock_input_size["num_heads"] * mock_input_size["head_dim"]
    )
    
    yield model
    
def test_forward(mock_input_size, mock_inputs, mock_model):
    
    q, k, v, mem_length = mock_inputs
    
    attn, mask = mock_model(q, k, v, mem_length)
    
    ### Shape check
    
    assert attn.shape == (
        mock_input_size["batch_size"],
        mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["num_heads"] * mock_input_size["head_dim"]
    )
    
    assert mask.shape == (
        mock_input_size["batch_size"],
        mock_input_size["num_heads"],
        mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["cache_length"] + mock_input_size["memory_length"] + mock_input_size["seq_length"]
    )
    
    ### masking check
    
    triangel_mask = torch.triu(
        torch.ones((mock_input_size["batch_size"], mock_input_size["num_heads"], mock_input_size["seq_length"], mock_input_size["seq_length"])),
        diagonal=1
    )
    
    assert torch.equal(
        mask[:, :, mock_input_size["memory_length"] : , mock_input_size["cache_length"] + mock_input_size["memory_length"]: ], 
        triangel_mask
    )
    
    memory_mask = torch.ones(
        (mock_input_size["batch_size"], mock_input_size["num_heads"], mock_input_size["memory_length"], mock_input_size["seq_length"])
    )
    
    assert torch.equal(
        mask[:, :, :mock_input_size["memory_length"], mock_input_size["cache_length"] + mock_input_size["memory_length"]:],
        memory_mask
    )