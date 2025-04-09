from torch.optim import Optimizer
from nlp_trainer.core.domain.port.dataloader import Dataloader
from nlp_trainer.core.domain.port.model import NLPModel
from nlp_trainer.core.domain.registry.strategy import StrategyRegistry
from nlp_trainer.core.domain.entity.strategy import StrategyType
from nlp_trainer.core.domain.strategy.model import ModelInputStrategy
from nlp_trainer.core.domain.strategy.loss import LossInputStrategy
from nlp_trainer.core.domain.port.loss import LossFunction


class TrainUsecase:
    def __init__(
        self,
        model: NLPModel,
        optimizer: Optimizer,
        loss_fn: LossFunction,
        strategy_registry: StrategyRegistry,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model_input_strategy: ModelInputStrategy = strategy_registry.get_strategy(
            StrategyType.MODEL_INPUT
        )
        self.loss_input_strategy: LossInputStrategy = strategy_registry.get_strategy(
            StrategyType.LOSS_INPUT
        )

    def execute(self, train_loader: Dataloader):
        for batch in train_loader:
            self.optimizer.zero_grad()

            x = self.model_input_strategy.execute(batch)
            y = self.model.train_step(x)

            y = self.loss_input_strategy.execute(batch, y)

            loss = self.loss_fn.calculate_loss(y)

            loss.backward()
            self.optimizer.step()
