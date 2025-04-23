from nlp_trainer.core.logging.handler import WandbHandler
from nlp_trainer.di.config import BaseConfig
from nlp_trainer.core.logging.base import setup_logging
import logging
import sys


class Application:
    def __init__(self):
        config = BaseConfig()
        ### Logging
        wandb_handler = WandbHandler(
            config.wandb_key,
            config.wandb_project_name,
            config.wandb_entity,
            config.wandb_run_name,
            config.wandb_notes,
        )
        console_handler = logging.StreamHandler(sys.stdout)

        setup_logging(
            level=config.logging_level, handlers=[wandb_handler, console_handler]
        )

    def run(self):
        raise NotImplementedError("Subclasses must implement this method")
