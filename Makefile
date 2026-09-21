.PHONY: check test foundation naive-baselines propensity-diagnostics matching linear-dml causal-forest uplift-models uplift-metrics business-policies confounding-ablation overlap-stress dag

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

linear-dml:
	PYTHONPATH=src python scripts/run_linear_dml.py --config configs/base.yaml

causal-forest:
	PYTHONPATH=src python scripts/run_causal_forest.py --config configs/base.yaml

uplift-models:
	PYTHONPATH=src python scripts/run_uplift_models.py --config configs/base.yaml

uplift-metrics: naive-baselines causal-forest uplift-models
	PYTHONPATH=src python scripts/run_uplift_metrics.py --config configs/base.yaml

business-policies: matching linear-dml uplift-metrics
	PYTHONPATH=src python scripts/run_business_policies.py --config configs/base.yaml

confounding-ablation: matching linear-dml
	PYTHONPATH=src python scripts/run_confounding_ablation.py --config configs/base.yaml

overlap-stress:
	PYTHONPATH=src python scripts/run_overlap_stress.py --config configs/base.yaml

dag:
	dot -Tsvg docs/causal_dag.dot -o docs/causal_dag.svg
