build:
	python -m build

install:
	pip install -e .

run_tests:
	python -m unittest tests/test_attention_archs.py 