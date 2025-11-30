# Makefile for Fuel Prediction Project
# Windows-compatible (uses PowerShell)

.PHONY: help install clean train predict pipeline test

help:
	@echo "Available commands:"
	@echo "  make install          - Install package in development mode"
	@echo "  make clean            - Clean generated files"
	@echo "  make train            - Train all models"
	@echo "  make predict          - Run inference (stacking)"
	@echo "  make pipeline         - Run full training + inference pipeline"
	@echo "  make test             - Run tests (when implemented)"

install:
	pip install -e .

install-dev:
	pip install -e ."[dev]"

clean:
	@echo "Cleaning generated files..."
	@if exist models rmdir /s /q models
	@if exist submissions rmdir /s /q submissions
	@if exist src\fuel_prediction\__pycache__ rmdir /s /q src\fuel_prediction\__pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@echo "Clean complete"

train:
	@echo "Training all models..."
	python pipelines/01_train_gbm.py
	python pipelines/02_train_lstm.py
	python pipelines/05_lstm_oof.py
	python pipelines/06_stacking_train.py

predict:
	@echo "Running inference..."
	python pipelines/07_stacking_inference.py

pipeline:
	@echo "Running full pipeline..."
	fuel-predict pipeline

test:
	@echo "Running tests..."
	pytest tests/ -v

lint:
	@echo "Linting code..."
	flake8 src/ pipelines/

format:
	@echo "Formatting code..."
	black src/ pipelines/
	isort src/ pipelines/
