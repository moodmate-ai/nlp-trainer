import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class LanguageModelingUsecase:
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_loader: DataLoader, loss_fn: nn.Module):
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()
            self.model.forward(batch)
            self.optimizer.step()

    def evaluate(self, val_loader: DataLoader):
        pass

    def test(self, test_loader: DataLoader):
        pass