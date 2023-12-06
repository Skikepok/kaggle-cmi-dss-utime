all: install test

install:
	pipenv install

test:
	PYTHONPATH=$$(pwd) pipenv run python scripts/test_prediction.py

fine_tune:
	PYTHONPATH=$$(pwd) pipenv run python scripts/fine_tuning.py
