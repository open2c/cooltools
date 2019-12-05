.PHONY: build install test docs clean clean-pyc clean-dist build-dist publish-test publish


build:
	python setup.py build_ext --inplace

install:
	pip install -e .

test:
	pytest

docs:
	cd docs && make html


clean-pyc:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force  {} +

clean-dist:
	rm -rf build/
	rm -rf dist/

clean: clean-pyc clean-dist


build-dist: clean-dist
	python setup.py sdist
	# python setup.py bdist_wheel

publish-test: build-dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish: build-dist
	twine upload dist/*
