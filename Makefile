.PHONY: test doc format mypy

test:
	python3 -m tox

doc:
	cd doc && make html

mypy:
	python3 -m tox -e mypy

style_check:
	python3 -m tox -e style_check

staged_check:
	python3 -m tox -e pre-commit
