import logging
from typing import Optional
from .formatter import JSONFormatter
import json
import wandb
import os


class WandbHandler(logging.Handler):
    """
    A logging handler that logs to wandb.
    """

    def __init__(
        self,
        wandb_key: str,
        project_name: str,
        entity: str,
        run_name: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        super().__init__()
        self.setFormatter(JSONFormatter())
        os.system(f"wandb login {wandb_key}")
        # if wandb.run is None:
        self.wandb_run = wandb.init(
            project=project_name, entity=entity, name=run_name, notes=notes
        )

    def format(self, record):
        data = record.getMessage()
        data = json.loads(data)

        return data.get("wandb", None)

    def emit(self, record):
        try:
            data = self.format(record)

            if data is not None:
                assert isinstance(data, dict)
                self.wandb_run.log(data)

        except Exception:
            self.handleError(record)
