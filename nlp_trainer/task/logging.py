from nlp_trainer.core.app import Application
from logging import getLogger

logger = getLogger(__name__)


class LoggingTask(Application):
    def run(self):
        logger.info({"msg": "LoggingTask", "wandb": {"loss": 1.23}})
        logger.info({"msg": "LoggingTask", "wandb": {"loss": 1.5}})
        logger.info({"msg": "LoggingTask", "wandb": {"loss": 1.00005}})


if __name__ == "__main__":
    LoggingTask().run()
