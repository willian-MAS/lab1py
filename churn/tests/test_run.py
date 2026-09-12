"""Teste de fumaça (smoke test) do projeto.

Garante que o projeto Kedro carrega e que o DAG registrado tem os nós
esperados. É o teste que pega erro de digitação em nome de dataset ou
parâmetro antes de rodar o pipeline inteiro.
"""

from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

# 6 nós de engenharia de dados + 4 de modelagem + 3 de refit + 5 de inferência
TOTAL_DE_NOS = 18


class TestKedroProject:
    def test_pipelines_registradas(self):
        bootstrap_project(Path.cwd())

        assert set(pipelines) == {
            "__default__",
            "data_engineering",
            "modelling",
            "refit",
            "inference",
        }

    def test_default_tem_todos_os_nos(self):
        bootstrap_project(Path.cwd())

        nomes = {node.name for node in pipelines["__default__"].nodes}

        assert "clean_data" in nomes
        assert "predict" in nomes
        assert len(nomes) == TOTAL_DE_NOS
