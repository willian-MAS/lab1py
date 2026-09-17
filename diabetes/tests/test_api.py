"""Testes da API FastAPI.

Usam o TestClient do FastAPI, que sobe a aplicacao em memoria -- nao e preciso
rodar o uvicorn para testar. Os testes de dados e de inferencia dependem dos
artefatos de producao; se eles ainda nao existirem (projeto recem-clonado, sem
`kedro run`), o teste e pulado em vez de falhar.
"""

import pytest
from fastapi.testclient import TestClient

from diabetes import api
from diabetes.api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def artefatos_prontos(client: TestClient) -> bool:
    return client.get("/health").json()["production_artifacts_ready"]


class TestHealth:
    def test_responde_ok_e_lista_as_pipelines(self, client: TestClient):
        resposta = client.get("/health")

        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["status"] == "ok"
        assert corpo["project"] == "diabetes"
        assert {"data_engineering", "modelling", "refit", "inference"} <= set(
            corpo["pipelines"]
        )

    def test_raiz_redireciona_para_a_documentacao(self, client: TestClient):
        resposta = client.get("/", follow_redirects=False)

        assert resposta.status_code in (307, 302)
        assert resposta.headers["location"] == "/docs"


class TestDatasets:
    def test_lista_datasets_do_catalogo(self, client: TestClient):
        resposta = client.get("/datasets")

        assert resposta.status_code == 200
        datasets = resposta.json()["datasets"]
        assert "master_table" in datasets
        assert "production_model" in datasets
        # parametros nao sao dados: nao entram na listagem
        assert not [name for name in datasets if name.startswith("param")]

    def test_serve_um_dataset_tabular_paginado(
        self, client: TestClient, artefatos_prontos: bool
    ):
        if not artefatos_prontos:
            pytest.skip("rode `kedro run` antes: faltam artefatos")

        resposta = client.get("/datasets/master_table", params={"limit": 5})

        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["name"] == "master_table"
        assert corpo["returned"] == 5
        assert corpo["rows"] >= corpo["returned"]
        assert "Outcome" in corpo["columns"]
        assert len(corpo["records"]) == 5

    def test_paginacao_anda_para_frente(
        self, client: TestClient, artefatos_prontos: bool
    ):
        if not artefatos_prontos:
            pytest.skip("rode `kedro run` antes: faltam artefatos")

        primeira = client.get("/datasets/master_table", params={"limit": 2}).json()
        segunda = client.get(
            "/datasets/master_table", params={"limit": 2, "offset": 2}
        ).json()

        assert primeira["records"] != segunda["records"]
        assert segunda["offset"] == 2

    def test_dataset_inexistente_da_404(self, client: TestClient):
        assert client.get("/datasets/nao_existe").status_code == 404

    def test_modelo_em_pickle_da_415(self, client: TestClient, artefatos_prontos: bool):
        if not artefatos_prontos:
            pytest.skip("rode `kedro run` antes: faltam artefatos")

        resposta = client.get("/datasets/production_model")

        assert resposta.status_code == 415
        assert "serializado" in resposta.json()["detail"]


class TestInferenciaOnline:
    def test_escora_registros_enviados_no_corpo(
        self, client: TestClient, artefatos_prontos: bool
    ):
        if not artefatos_prontos:
            pytest.skip("rode `kedro run` antes: faltam artefatos")

        payload = {
            "instances": [
                {
                    "Pregnancies": 6,
                    "Glucose": 148,
                    "BloodPressure": 72,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.627,
                    "Age": 50,
                },
                {
                    "Pregnancies": 1,
                    "Glucose": 85,
                    "BloodPressure": 66,
                    "SkinThickness": 29,
                    "Insulin": 0,
                    "BMI": 26.6,
                    "DiabetesPedigreeFunction": 0.351,
                    "Age": 31,
                },
            ]
        }

        resposta = client.post("/inference", json=payload)

        assert resposta.status_code == 200
        predicoes = resposta.json()["predictions"]
        assert len(predicoes) == 2
        assert {"index", "prediction", "probability"} == set(predicoes[0])
        assert all(p["prediction"] in (0, 1) for p in predicoes)
        assert all(0.0 <= p["probability"] <= 1.0 for p in predicoes)

    def test_lista_vazia_e_recusada(self, client: TestClient):
        assert client.post("/inference", json={"instances": []}).status_code == 422

    def test_corpo_sem_instances_e_recusado(self, client: TestClient):
        assert client.post("/inference", json={}).status_code == 422


class TestExecucoesEmSegundoPlano:
    """O disparo em segundo plano e testado com o executor trocado por um dublê.

    Assim validamos a mecanica do endpoint (run_id, rastreamento, 404) sem
    rodar o pipeline de verdade e sobrescrever os dados do projeto.
    """

    @pytest.fixture(autouse=True)
    def _executor_falso(self, monkeypatch):
        def _executor(run_id: str, pipeline_names: list[str]) -> None:
            api._set_run(run_id, status="completed", finished_at="2026-01-01T00:00:00Z")

        monkeypatch.setattr(api, "_run_pipelines_background", _executor)

    def test_train_devolve_run_id_rastreavel(self, client: TestClient):
        resposta = client.post("/train")

        assert resposta.status_code == 200
        run_id = resposta.json()["run_id"]
        assert resposta.json()["status"] == "pending"

        status = client.get(f"/train/{run_id}")
        assert status.status_code == 200
        assert status.json()["pipeline"] == "train"
        assert status.json()["status"] in {"pending", "running", "completed", "failed"}

    def test_batch_inference_devolve_run_id_rastreavel(self, client: TestClient):
        resposta = client.post("/batch-inference")

        assert resposta.status_code == 200
        run_id = resposta.json()["run_id"]

        status = client.get(f"/batch-inference/{run_id}")
        assert status.status_code == 200
        assert status.json()["pipeline"] == "batch-inference"

    def test_run_id_inexistente_da_404(self, client: TestClient):
        assert client.get("/train/nao-existe").status_code == 404
        assert client.get("/batch-inference/nao-existe").status_code == 404
