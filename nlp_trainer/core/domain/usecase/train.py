from torch.optim import Optimizer
from nlp_trainer.core.domain.port.dataloader import Dataloader
from nlp_trainer.core.domain.port.model import NLPModel
from nlp_trainer.core.domain.registry.strategy import StrategyRegistry
from nlp_trainer.core.domain.entity.strategy import StrategyType
from nlp_trainer.core.domain.strategy.data_to_model import DataToModelStrategy
from nlp_trainer.core.domain.strategy.model_to_loss import ModelToLossStrategy
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
        self.input_strategy: DataToModelStrategy = strategy_registry.get_strategy(
            StrategyType.DATA_TO_MODEL_INPUT
        )
        self.output_strategy: ModelToLossStrategy = strategy_registry.get_strategy(
            StrategyType.MODEL_OUTPUT_TO_LOSS_INPUT
        )

    def execute(self, train_loader: Dataloader):
        for batch in train_loader:
            self.optimizer.zero_grad()

            x = self.input_strategy.execute(batch)
            y = self.model.train_step(x)

            y = self.output_strategy.execute(y)

            loss = self.loss_fn.calculate_loss(y)

            loss.backward()
            self.optimizer.step()
