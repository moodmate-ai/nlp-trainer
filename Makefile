exec-torchinfo:
	poetry run python -m nlp_trainer.task.torchinfo

exec-pipeline:
	poetry run python -m nlp_trainer.task.pipeline

exec-inference:
	poetry run python -m nlp_trainer.task.inference

exec-model-size:
	poetry run python -m nlp_trainer.task.model_size

exec-language-modeling:
	poetry run python -m nlp_trainer.task.titans.language_modeling

exec-logging:
	poetry run python -m nlp_trainer.task.logging
