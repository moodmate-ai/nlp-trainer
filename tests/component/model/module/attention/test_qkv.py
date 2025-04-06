from nlp_trainer.component.model.module.attention.qkv import MultiHeadQKVCreator
from unittest.mock import Mock, patch, MagicMock, call
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
        mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["num_heads"] * mock_input_size["head_dim"]
    )
    
    size_cache = (
        mock_input_size["batch_size"],
        mock_input_size["cache_length"],
        mock_input_size["num_heads"],
        mock_input_size["head_dim"]
    )
    
    x = torch.rand(size)
    k_cahce = torch.rand(size_cache)
    v_cache = torch.rand(size_cache)
    
    yield x, k_cahce, v_cache
    

@pytest.fixture
def mock_model(mock_input_size):
    
    def mock_side_effect(*args, **kwargs):
        return args[0]
    
    mock_encoder = Mock(side_effect=mock_side_effect)
    
    model = MultiHeadQKVCreator(
        num_heads=mock_input_size["num_heads"],
        hidden_dim=mock_input_size["num_heads"] * mock_input_size["head_dim"],
        positional_encoder=mock_encoder
    )

    yield model
        
def test_forward(mock_input_size, mock_inputs, mock_model):
    x, cache_key, cache_value = mock_inputs
     
    q, k, v = mock_model(x, cache_key, cache_value)
    
    assert q.shape == (
        mock_input_size["batch_size"],
        mock_input_size["num_heads"],
        mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["head_dim"]
    )
    
    assert k.shape == (
        mock_input_size["batch_size"],
        mock_input_size["num_heads"],
        mock_input_size["cache_length"] + mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["head_dim"]
    )
    
    assert v.shape == (
        mock_input_size["batch_size"],
        mock_input_size["num_heads"],
        mock_input_size["cache_length"] + mock_input_size["memory_length"] + mock_input_size["seq_length"],
        mock_input_size["head_dim"]
    )
    