import torch
from typing import List
from torch.optim import Optimizer
from nlp_trainer.core.domain.port.dataloader import Dataloader
from nlp_trainer.core.domain.port.model import NLPModel
from nlp_trainer.core.domain.registry.train import TrainRegistry
from nlp_trainer.core.domain.strategy.model import ModelInputStrategy
from nlp_trainer.core.domain.port.loss import LossFunction
from logging import getLogger

logger = getLogger(__name__)


class TrainUsecase:
    def __init__(
        self,
        model: NLPModel,
        optimizer: Optimizer,
        loss_fn_list: List[LossFunction],
        train_registry: TrainRegistry,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn_list = loss_fn_list
        self.train_registry = train_registry
        self.model_input_strategy: ModelInputStrategy = (
            train_registry.get_model_input_strategy()
        )

    def execute(self, train_loader: Dataloader):
        for batch in train_loader:
            self.optimizer.zero_grad()

            x = self.model_input_strategy.execute(batch)
            y = self.model.train_step(x)

            loss_list = []
            for loss_fn in self.loss_fn_list:
                strategy = self.train_registry.get_loss_input_strategy(
                    loss_fn.get_type()
                )
                loss_input = strategy.execute(batch, y)

                loss = loss_fn.calculate_loss(loss_input)
                loss_list.append(loss)

            loss = torch.sum(torch.stack(loss_list))
            loss.backward()
            self.optimizer.step()
