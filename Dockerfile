FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY tinyvision /app/tinyvision
COPY tests /app/tests

RUN pip install --upgrade pip && pip install .[dev]

EXPOSE 8000
CMD ["uvicorn", "tinyvision.app:app", "--host", "0.0.0.0", "--port", "8000"]