.PHONY: setup data eda train eval serve dashboard lint test

setup:
	python -m pip install -r requirements.txt

data:
	python scripts/download_data.py

eda:
	python scripts/eda.py

train:
	python scripts/train.py

eval:
	python scripts/evaluate.py

serve:
	python scripts/serve.py

dashboard:
	python scripts/dashboard.py

lint:
	ruff check src app scripts
	black --check src app scripts

test:
	pytest -q
