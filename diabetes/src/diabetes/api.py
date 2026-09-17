"""API FastAPI que expoe as pipelines Kedro do projeto diabetes como REST.

Tres familias de endpoint:

* **Dados** -- ``GET /datasets`` e ``GET /datasets/{name}`` servem qualquer
  dataset do Data Catalog como JSON, sem que o cliente precise saber se aquilo
  esta em CSV, Parquet ou num bucket.
* **Treino** -- ``POST /train`` dispara data_engineering + modelling + refit
  em segundo plano e devolve um ``run_id``; ``GET /train/{run_id}`` acompanha.
* **Inferencia** -- ``POST /inference`` escora registros enviados no corpo da
  requisicao (sincrono) e ``POST /batch-inference`` roda a pipeline sobre o
  arquivo declarado no catalogo (assincrono).

O ponto central do desenho: a inferencia online roda **a mesma pipeline** do
modo batch. O que muda e so a origem dos dados -- o catalogo recebe
``MemoryDataset`` no lugar dos arquivos, e nada toca o disco.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from kedro.framework.project import configure_project
from kedro.framework.project import pipelines as kedro_pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_PATH = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "diabetes"

# Pipelines disparadas por POST /train, na ordem.
TRAIN_PIPELINES = ["data_engineering", "modelling", "refit"]

# Datasets substituidos por MemoryDataset na inferencia online: a requisicao
# entra e sai pela memoria, sem escrever nos arquivos do modo batch.
ONLINE_OVERRIDES = [
    "raw_inference_data",
    "cleaned_inference_data",
    "imputed_inference_data",
    "clipped_inference_data",
    "featured_inference_data",
    "encoded_inference_data",
    "scaled_inference_data",
    "inference_predictions",
]

# Artefatos sem os quais nao ha inferencia possivel.
PRODUCTION_ARTIFACTS = [
    "production_imputers",
    "production_outlier_thresholds",
    "production_encoders",
    "production_scalers",
    "production_model",
]

_bootstrap_lock = threading.Lock()
_bootstrapped = False


def _ensure_bootstrap() -> None:
    """Inicializa o projeto Kedro uma unica vez (seguro entre threads)."""
    global _bootstrapped  # noqa: PLW0603
    if _bootstrapped:
        return
    with _bootstrap_lock:
        if not _bootstrapped:
            bootstrap_project(PROJECT_PATH)
            configure_project(PACKAGE_NAME)
            _bootstrapped = True


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _get_catalog(session: KedroSession):
    """Devolve o Data Catalog da sessao (compativel com versoes do Kedro)."""
    context = session.load_context()
    if hasattr(context, "catalog"):
        return context.catalog
    return context._get_catalog()  # noqa: SLF001


def _jsonable(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converte um DataFrame em registros JSON (NaN vira null)."""
    return json.loads(df.to_json(orient="records"))


def _ensure_json_safe(name: str, data: Any) -> Any:
    """Recusa com 415 o que nao couber em JSON (um modelo em pickle, p. ex.)."""
    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=415,
            detail=(
                f"O dataset '{name}' nao pode ser serializado como JSON "
                f"({type(data).__name__} contendo objetos Python, por exemplo um "
                "modelo em pickle). Use os endpoints de inferencia."
            ),
        ) from exc
    return data


# ---------------------------------------------------------------------------
# Contratos da API (Pydantic) -- viram documentacao automatica em /docs
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    project: str
    pipelines: list[str]
    production_artifacts_ready: bool
    missing_artifacts: list[str] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    datasets: list[str]


class DatasetResponse(BaseModel):
    name: str
    rows: int | None = None
    returned: int
    offset: int
    columns: list[str] | None = None
    records: list[Any] | dict[str, Any]


class RunResponse(BaseModel):
    run_id: str
    status: str


class RunStatus(BaseModel):
    run_id: str
    status: str
    pipeline: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: Any | None = None


class InferenceRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Registros a escorar. Cada um deve trazer as 8 medidas do exame: "
            "Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, "
            "DiabetesPedigreeFunction e Age."
        ),
        json_schema_extra={
            "example": [
                {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                }
            ]
        },
    )


class InferenceResponse(BaseModel):
    predictions: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Registro das execucoes em segundo plano
# ---------------------------------------------------------------------------

runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()


def _set_run(run_id: str, **fields: Any) -> None:
    with _runs_lock:
        runs.setdefault(run_id, {}).update(fields)


def _get_run(run_id: str) -> dict[str, Any] | None:
    with _runs_lock:
        entry = runs.get(run_id)
        return entry.copy() if entry else None


def _run_pipelines_background(run_id: str, pipeline_names: list[str]) -> None:
    """Executa uma ou mais pipelines Kedro em sequencia, numa thread."""
    _set_run(run_id, status="running", started_at=_now())
    try:
        _ensure_bootstrap()
        for name in pipeline_names:
            with KedroSession.create(project_path=PROJECT_PATH) as session:
                session.run(pipeline_name=name)
        _set_run(run_id, status="completed", finished_at=_now())
    except Exception as exc:
        logger.exception("Execucao %s falhou", run_id)
        _set_run(
            run_id,
            status="failed",
            finished_at=_now(),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _start_background_run(
    pipeline_label: str, pipeline_names: list[str]
) -> RunResponse:
    run_id = str(uuid.uuid4())
    _set_run(run_id, status="pending", pipeline=pipeline_label, result=None, error=None)
    threading.Thread(
        target=_run_pipelines_background,
        args=(run_id, pipeline_names),
        daemon=True,
    ).start()
    return RunResponse(run_id=run_id, status="pending")


# ---------------------------------------------------------------------------
# Aplicacao
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_bootstrap()
    yield


app = FastAPI(
    title="Diabetes ML API",
    description=(
        "Pipelines Kedro de predicao de diabetes expostas como API REST. "
        "Documentacao interativa em /docs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Manda quem abre a raiz direto para a documentacao interativa."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Diz se o servico esta de pe e se ja existem artefatos de producao."""
    _ensure_bootstrap()
    missing: list[str] = []
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = _get_catalog(session)
        for name in PRODUCTION_ARTIFACTS:
            try:
                if not catalog.exists(name):
                    missing.append(name)
            except Exception:  # noqa: BLE001 - dataset ausente conta como faltante
                missing.append(name)

    return HealthResponse(
        status="ok",
        project=PACKAGE_NAME,
        pipelines=sorted(kedro_pipelines),
        production_artifacts_ready=not missing,
        missing_artifacts=missing,
    )


# ---- Dados ----------------------------------------------------------------


@app.get("/datasets", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """Lista os datasets declarados no Data Catalog."""
    _ensure_bootstrap()
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = _get_catalog(session)
        names = sorted(catalog.keys()) if hasattr(catalog, "keys") else catalog.list()

    return DatasetListResponse(
        datasets=[name for name in names if not name.startswith("param")]
    )


@app.get("/datasets/{name}", response_model=DatasetResponse)
def read_dataset(
    name: str,
    limit: int = Query(100, ge=1, le=10_000, description="Maximo de registros"),
    offset: int = Query(0, ge=0, description="Registros a pular"),
) -> DatasetResponse:
    """Serve um dataset do catalogo como JSON.

    O cliente pede pelo nome logico (``master_table``, ``baseline_metrics``,
    ...) e nao precisa saber onde nem em que formato aquilo esta guardado --
    e a abstracao do catalogo chegando ate o HTTP.
    """
    _ensure_bootstrap()
    try:
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            catalog = _get_catalog(session)
            names = (
                sorted(catalog.keys()) if hasattr(catalog, "keys") else catalog.list()
            )
            if name not in names:
                raise HTTPException(
                    status_code=404, detail=f"Dataset '{name}' nao existe no catalogo"
                )
            data = catalog.load(name)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Falha ao carregar o dataset %s", name)
        raise HTTPException(
            status_code=404,
            detail=(
                f"Nao foi possivel carregar '{name}': {type(exc).__name__}: {exc}. "
                "O dataset pode ainda nao ter sido gerado -- rode POST /train."
            ),
        ) from exc

    if isinstance(data, pd.DataFrame):
        page = data.iloc[offset : offset + limit]
        return DatasetResponse(
            name=name,
            rows=int(len(data)),
            returned=int(len(page)),
            offset=offset,
            columns=[str(col) for col in data.columns],
            records=_jsonable(page),
        )

    if isinstance(data, list):
        _ensure_json_safe(name, data[:1])
        page = data[offset : offset + limit]
        return DatasetResponse(
            name=name,
            rows=len(data),
            returned=len(page),
            offset=offset,
            records=page,
        )

    if isinstance(data, dict):
        _ensure_json_safe(name, data)
        return DatasetResponse(name=name, returned=1, offset=0, records=data)

    raise HTTPException(
        status_code=415,
        detail=(
            f"O dataset '{name}' e do tipo {type(data).__name__} e nao pode ser "
            "serializado como JSON (por exemplo, um modelo em pickle)."
        ),
    )


# ---- Treino ---------------------------------------------------------------


@app.post("/train", response_model=RunResponse)
def start_training() -> RunResponse:
    """Dispara data_engineering + modelling + refit em segundo plano."""
    return _start_background_run("train", TRAIN_PIPELINES)


@app.get("/train/{run_id}", response_model=RunStatus)
def get_training_status(run_id: str) -> RunStatus:
    """Acompanha o andamento de uma execucao de treino."""
    entry = _get_run(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Execucao nao encontrada")
    return RunStatus(run_id=run_id, **entry)


# ---- Inferencia em lote ---------------------------------------------------


@app.post("/batch-inference", response_model=RunResponse)
def start_batch_inference() -> RunResponse:
    """Roda a pipeline de inferencia sobre o arquivo declarado no catalogo."""
    return _start_background_run("batch-inference", ["inference"])


@app.get("/batch-inference/{run_id}", response_model=RunStatus)
def get_batch_inference_status(run_id: str) -> RunStatus:
    """Acompanha o andamento de uma inferencia em lote."""
    entry = _get_run(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Execucao nao encontrada")
    return RunStatus(run_id=run_id, **entry)


# ---- Inferencia online ----------------------------------------------------


@app.post("/inference", response_model=InferenceResponse)
def run_inference(request: InferenceRequest) -> InferenceResponse:
    """Escora registros enviados no corpo da requisicao, de forma sincrona.

    Monta o catalogo a partir do contexto Kedro, troca as entradas e saidas da
    inferencia por ``MemoryDataset`` e roda a pipeline ``inference`` com o
    ``SequentialRunner``. E exatamente a mesma pipeline do modo batch: mesmas
    funcoes de limpeza, imputacao, features, encoding e escalonamento, com os
    mesmos artefatos de producao.
    """
    if not request.instances:
        raise HTTPException(status_code=422, detail="A lista 'instances' esta vazia")

    _ensure_bootstrap()
    try:
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            catalog = _get_catalog(session)

            for dataset_name in ONLINE_OVERRIDES:
                catalog[dataset_name] = MemoryDataset()
            catalog["raw_inference_data"] = MemoryDataset(
                data=pd.DataFrame(request.instances)
            )

            SequentialRunner().run(kedro_pipelines["inference"], catalog)
            predictions = catalog.load("inference_predictions")

        return InferenceResponse(predictions=predictions)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inferencia online falhou")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
