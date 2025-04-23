from dependency_injector import containers, providers
from nlp_trainer.core.logging.handler import WandbHandler
from nlp_trainer.di.config import Config
from nlp_trainer.core.logging.base import setup_logging
import logging
import sys


class Container(containers.DeclarativeContainer):
    config = Config()

    wandb_logging_handler = providers.Singleton(
        WandbHandler,
        config.wandb_key,
        config.wandb_project_name,
        config.wandb_entity,
        config.wandb_run_name,
        config.wandb_notes,
    )

    console_logging_handler = providers.Singleton(
        logging.StreamHandler, stream=sys.stdout
    )

    _init_logging = providers.Callable(
        setup_logging,
        level=config.logging_level,
        handlers=[wandb_logging_handler, console_logging_handler],
    )

    @classmethod
    def initialize(cls):
        container = cls()
        container._init_logging()
        return container
