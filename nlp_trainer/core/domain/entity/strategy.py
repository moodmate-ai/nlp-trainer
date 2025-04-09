from nlp_trainer.core.type.base import StrEnum


class StrategyType(StrEnum):
    DATA_TO_MODEL_INPUT = "data_to_model_input"
    MODEL_OUTPUT_TO_LOSS_INPUT = "model_output_to_loss_input"
