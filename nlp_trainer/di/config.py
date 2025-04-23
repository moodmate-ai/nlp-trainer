from pydantic_settings import BaseSettings


class BaseConfig(BaseSettings):
    wandb_key: str
    wandb_project_name: str
    wandb_entity: str
    wandb_run_name: str
    wandb_notes: str
    logging_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
