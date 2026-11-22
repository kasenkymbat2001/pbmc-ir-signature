.PHONY: install test demo
install:
	pip install -e ".[dev]"
test:
	pytest -q
demo:
	irsig demo --out results/demo --n-perm 200
