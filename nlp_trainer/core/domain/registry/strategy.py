from abc import ABC, abstractmethod
from nlp_trainer.core.domain.entity.strategy import StrategyType
from nlp_trainer.core.domain.strategy.data_to_model import DataToModelStrategy
from nlp_trainer.core.domain.strategy.model_to_loss import ModelToLossStrategy


class StrategyRegistry(ABC):
    @abstractmethod
    def get_strategy(
        self, strategy_type: StrategyType
    ) -> DataToModelStrategy | ModelToLossStrategy:
        pass
