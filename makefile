build:
	python -m build

install:
	pip install -e .

test:
	python -m unittest tests/test_attention_archs.py
	
upload:
	twine upload dist/*