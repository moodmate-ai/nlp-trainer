from abc import ABC, abstractmethod
from nlp_trainer.core.domain.strategy.model import ModelInputStrategy
from nlp_trainer.core.domain.strategy.loss import LossInputStrategy
from nlp_trainer.core.domain.entity.loss import LossFunctionType


class TrainRegistry(ABC):
    @abstractmethod
    def get_model_input_strategy(self) -> ModelInputStrategy:
        pass

    @abstractmethod
    def get_loss_input_strategy(
        self, loss_fn_type: LossFunctionType
    ) -> LossInputStrategy:
        pass
