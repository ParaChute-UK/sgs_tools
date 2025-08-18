.PHONY: test doc mypy_check style_check

doc:
	poetry install --with doc
	poetry run sphinx-build -b html doc/ documentation

test:
	tox

mypy_check:
	tox -e mypy

style_check:
	tox -e style_check

staged_fix:
	tox -e pre_commit
