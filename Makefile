exec-torchinfo:
	poetry run python -m nlp_trainer.task.torchinfo

exec-pipeline:
	poetry run python -m nlp_trainer.task.pipeline

exec-inference:
	poetry run python -m nlp_trainer.task.inference