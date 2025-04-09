from abc import ABC, abstractmethod
from nlp_trainer.core.domain.entity.strategy import StrategyType
from nlp_trainer.core.domain.strategy.model import ModelInputStrategy
from nlp_trainer.core.domain.strategy.loss import LossInputStrategy


class StrategyRegistry(ABC):
    @abstractmethod
    def get_strategy(
        self, strategy_type: StrategyType
    ) -> ModelInputStrategy | LossInputStrategy:
        pass
