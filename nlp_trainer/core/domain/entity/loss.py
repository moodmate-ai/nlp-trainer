from nlp_trainer.core.type.base import StrEnum


class LossFunctionType(StrEnum):
    PERPLEXITY = "perplexity"
    MSE = "mean_squared_error"
    ZERO = "zero"
