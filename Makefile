.PHONY: build publish run lint install uninstall clean

# pipx 的默认解释器未必满足 pyproject 里锁定的 requires-python，所以按 pyproject 的
# 版本约束挑一个 uv 管理的解释器给它。--system 是为了绕开当前目录的 .venv，否则 pipx
# 建出来的 venv 会依赖项目里的 .venv。
PYTHON_SPEC := $(shell sed -n 's/^requires-python *= *"\(.*\)"/\1/p' pyproject.toml)
PYTHON := $(shell uv python find --system '$(PYTHON_SPEC)')

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

# 先清 dist，避免残留的旧版本 wheel 让下面的通配符匹配到多个文件。
# 先卸载再装（而不是 --force）：--force 会复用已存在的 venv 并忽略 --python，
# 于是老 venv 的解释器版本会一直沿用下去。
install: clean build
	pipx uninstall transvideo || true
	pipx install --python "$(PYTHON)" dist/*.whl

uninstall:
	pipx uninstall transvideo

clean:
	rm -rf dist
