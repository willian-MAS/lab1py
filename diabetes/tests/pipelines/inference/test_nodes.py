"""Testes unitarios dos nos de inferencia."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from diabetes.pipelines.inference.nodes import predict, to_dataframe


def _artefato_de_modelo() -> dict:
    estimator = LogisticRegression()
    estimator.fit(
        pd.DataFrame({"Glucose": [80.0, 200.0, 90.0, 190.0], "Age": [20, 60, 25, 55]}),
        [0, 1, 0, 1],
    )
    return {
        "estimator": estimator,
        "target_column": "Outcome",
        "feature_columns": ["Glucose", "Age"],
    }


class TestToDataFrame:
    def test_lista_de_dicionarios_vira_dataframe(self):
        df = to_dataframe([{"Glucose": 100, "Age": 30}])

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 2)

    def test_dataframe_passa_intacto(self):
        df_in = pd.DataFrame({"Glucose": [100]})

        assert to_dataframe(df_in) is df_in


class TestPredict:
    def test_formato_da_resposta(self):
        resultado = predict(
            _artefato_de_modelo(),
            pd.DataFrame({"Glucose": [85.0, 195.0], "Age": [21, 58], "Extra": [1, 2]}),
        )

        assert [r["index"] for r in resultado] == [0, 1]
        assert all(r["prediction"] in (0, 1) for r in resultado)
        assert all(0.0 <= r["probability"] <= 1.0 for r in resultado)

    def test_erro_claro_quando_falta_coluna(self):
        with pytest.raises(ValueError, match="Faltam colunas"):
            predict(_artefato_de_modelo(), pd.DataFrame({"Glucose": [100.0]}))
