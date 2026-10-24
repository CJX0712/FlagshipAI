# FlagshipAI —— Makefile
# 一键复现全链路: 环境 → 验证 → 测试 → 基准 → 训练 → 文档
# 作者: 晨星 (MorningStar)

PY      ?= python
VENV    ?= .venv
PYPY    := $(VENV)/Scripts/python.exe
PYPY_LN := $(VENV)/bin/python
P       := $(shell if [ -f "$(PYPY)" ]; then echo "$(PYPY)"; else echo "$(PYPY_LN)"; fi)
OUT     ?= results

.DEFAULT_GOAL := help
.PHONY: help venv install dev test verify bench train example clean all ci

help: ## 显示所有可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv: ## 创建隔离虚拟环境
	$(PY) -m venv $(VENV)
	@echo "已创建 $(VENV)"

install: ## 安装核心依赖 (numpy)
	$(P) -m pip install -U pip
	$(P) -m pip install -e .

dev: ## 安装全量开发依赖 (含 torch 交叉验证 / pytest / scipy / sklearn)
	$(P) -m pip install -U pip
	$(P) -m pip install -e ".[dev]"

test: ## 运行单元测试
	$(P) -m pytest tests -q

verify: ## 运行全部硬不变量并落盘 results/verification.json
	$(P) -m morningstar.verify.report --out $(OUT)

bench: ## 运行基准评测并落盘 results/benchmark.json
	$(P) -m morningstar.verify.benchmark --out $(OUT)

train: ## 训练旗舰 GPT (600 步, CPU 可跑)
	$(P) examples/train_tiny.py --steps 600 --out $(OUT)

example: ## 逐个运行 examples/ 下的示例
	@for f in examples/*.py; do \
		echo "=== $$f ==="; \
		$(P) $$f || exit 1; \
	done

all: install test verify bench train ## 完整复现链路

ci: test verify ## CI 等价目标 (测试 + 不变量)

clean: ## 清理构建产物与缓存
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
