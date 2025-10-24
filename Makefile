.PHONY: setup test run docker-build docker-run lint fmt

setup:
	python -m pip install --upgrade pip
	pip install -e .[dev]

test:
	pytest -q

run:
	uvicorn tinyvision.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t tinyvision:latest .

docker-run:
	docker run -p 8000:8000 tinyvision:latest

lint:
	ruff check tinyvision tests

fmt:
	black tinyvision tests