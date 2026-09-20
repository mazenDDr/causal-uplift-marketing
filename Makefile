.PHONY: check test foundation naive-baselines propensity-diagnostics matching dag

check:
	python -m ruff check .
	python -m ruff format --check .
	PYTHONPATH=src python -m pytest

test:
	PYTHONPATH=src python -m pytest

foundation:
	PYTHONPATH=src python scripts/run_foundation.py --config configs/base.yaml

naive-baselines:
	PYTHONPATH=src python scripts/run_naive_baselines.py --config configs/base.yaml

propensity-diagnostics:
	PYTHONPATH=src python scripts/run_propensity_diagnostics.py --config configs/base.yaml

matching:
	PYTHONPATH=src python scripts/run_matching.py --config configs/base.yaml

dag:
	dot -Tsvg docs/causal_dag.dot -o docs/causal_dag.svg
