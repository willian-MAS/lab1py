# =============================================================================
# Imagem da API de predicao de diabetes (Kedro + FastAPI, servida por uvicorn)
# =============================================================================
# Instalacao em duas camadas, de proposito: as dependencias sao instaladas
# antes do codigo, entao editar um node nao invalida o cache do `uv sync`.

FROM python:3.11-slim

# O uv vem da imagem oficial da Astral -- nada de pip install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Camada 1: dependencias (muda pouco -> cache aproveitado entre builds)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Camada 2: configuracao, codigo e dado bruto
COPY conf/ conf/
COPY src/ src/
COPY data/01_raw/ data/01_raw/

# As demais camadas de dados sao criadas vazias: o pipeline as preenche.
RUN mkdir -p data/02_intermediate data/03_primary data/04_feature \
    data/05_model_input data/06_models data/07_model_output \
    data/08_reporting data/tmp

RUN uv sync --frozen --no-dev

EXPOSE 8000

# Usa o proprio endpoint /health para o Docker saber se o servico esta de pe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# --no-dev: sem isso o `uv run` tentaria instalar pytest/ruff ao subir o container.
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "diabetes.api:app", "--host", "0.0.0.0", "--port", "8000"]
