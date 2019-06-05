.PHONY: init install clean-pyc clean-dist clean build test publish docs-init docs

init:
	conda install --file requirements.txt

build:
	python setup.py build_ext --inplace

test:
	pytest

install:
	pip install -e .

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

clean-dist:
	rm -rf build/
	rm -rf dist/

clean: clean-pyc clean-dist

dist-build: clean-dist
	python setup.py sdist
	python setup.py bdist_wheel

publish: build
	twine upload dist/*

# docs-init:
# 	conda install --file docs/requirements.txt

# docs:
# 	cd docs && make html
