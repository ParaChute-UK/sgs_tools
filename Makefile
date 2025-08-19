.PHONY: doc test mypy_check style_check pre_commit

doc:
	@echo "Generating documentation..."
	@if command -v poetry >/dev/null 2>&1; then \
 		poetry install --with doc; \
		poetry run sphinx-build -b html doc/ documentation/html; \
  else \
    echo "Poetry not found. Attempting pip-based build..."; \
    pip install .[doc]; \
    sphinx-build -b html doc/ documentation/html; \
  fi

test:
	tox

mypy_check:
	tox -e mypy

style_check:
	tox -e style_check

pre_commit:
	tox -e pre_commit
