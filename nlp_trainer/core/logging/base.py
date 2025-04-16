import logging
import logging.handlers
import queue
import sys
from typing import Optional

import wandb
from logging.handlers import QueueHandler, QueueListener


class WandbHandler(logging.Handler):
    def __init__(self, project_name: str, run_name: Optional[str] = None):
        super().__init__()
        # Initialize wandb if not already initialized
        if wandb.run is None:
            wandb.init(project=project_name, name=run_name)

    def emit(self, record):
        try:
            # Format the record
            msg = self.format(record)
            # Log to wandb
            wandb.log(
                {
                    "log_level": record.levelname,
                    "message": msg,
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "timestamp": record.created,
                }
            )
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str = "INFO",
    project_name: str = "my_project",
    run_name: Optional[str] = None,
) -> None:
    """
    Set up logging configuration with queue handler, console and wandb outputs.

    Args:
        level: Logging level (default: "INFO")
        project_name: Name of the wandb project
        run_name: Optional name for the wandb run
    """
    # Create queue
    log_queue = queue.Queue()

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    wandb_handler = WandbHandler(project_name, run_name)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    wandb_handler.setFormatter(formatter)

    # Get root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Create queue handler and add it to root logger
    queue_handler = QueueHandler(log_queue)
    root.addHandler(queue_handler)

    # Create and start queue listener
    listener = QueueListener(
        log_queue, console_handler, wandb_handler, respect_handler_level=True
    )
    listener.start()

    # Store listener in root logger for future access
    root.listener = listener  # type: ignore


def shutdown_logging() -> None:
    """
    Properly shutdown logging system.
    Should be called when the application exits.
    """
    root = logging.getLogger()
    if hasattr(root, "listener"):
        root.listener.stop()  # type: ignore
