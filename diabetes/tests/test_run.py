"""Testes de fumaca do projeto.

Garantem que o projeto Kedro carrega e que o DAG registrado tem a forma
esperada. Pegam erro de digitacao em nome de dataset ou de parametro antes de
rodar o pipeline inteiro.
"""

from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project

NOS_POR_PIPELINE = {
    "data_engineering": 11,
    "modelling": 4,
    "refit": 8,
    "inference": 8,
}
TOTAL_DE_NOS = sum(NOS_POR_PIPELINE.values())


class TestProjetoKedro:
    def test_pipelines_registradas(self):
        bootstrap_project(Path.cwd())

        assert set(pipelines) == {"__default__", *NOS_POR_PIPELINE}

    def test_quantidade_de_nos_por_pipeline(self):
        bootstrap_project(Path.cwd())

        for nome, esperado in NOS_POR_PIPELINE.items():
            assert len(pipelines[nome].nodes) == esperado, nome

    def test_default_reune_todas_as_pipelines(self):
        bootstrap_project(Path.cwd())

        nomes = {node.name for node in pipelines["__default__"].nodes}

        assert len(nomes) == TOTAL_DE_NOS
        assert {"clean_data", "refit_model", "predict"} <= nomes

    def test_inferencia_so_consome_artefatos_de_producao(self):
        """A inferencia nunca pode depender de artefato de modelagem."""
        bootstrap_project(Path.cwd())

        entradas = set(pipelines["inference"].inputs())

        assert {"production_model", "production_encoders"} <= entradas
        assert not {name for name in entradas if name.startswith("modelling_")}
