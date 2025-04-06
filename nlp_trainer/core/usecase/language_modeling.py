import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from nlp_trainer.core.port.model import NLPModel


class LanguageModelingUsecase:
    def __init__(self, model: NLPModel, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_loader: DataLoader, loss_fn: nn.Module):
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()

            # output = self.model.train_step(batch)

            ## need converter: output -> loss fn input
            self.optimizer.step()

    def evaluate(self, val_loader: DataLoader):
        pass

    def test(self, test_loader: DataLoader):
        pass
