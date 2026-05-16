build:
	uv build

publish:
	uv publish

run:
	uv run

lint:
	uv run ruff check src
	uv run ruff format --check src
	uv run ty check src
