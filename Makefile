.PHONY: doc test mypy_check style_check pre_commit

doc:
	poetry install --with doc
	poetry run sphinx-build -b html doc/ documentation

test:
	tox

mypy_check:
	tox -e mypy

style_check:
	tox -e style_check

pre_commit:
	tox -e pre_commit
