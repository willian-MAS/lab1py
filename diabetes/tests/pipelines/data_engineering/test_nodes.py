"""Testes unitarios dos nos de engenharia de dados.

Nos sao funcoes puras: da para testa-los com DataFrames sinteticos, sem Kedro,
sem catalogo e sem arquivo nenhum.
"""

import numpy as np
import pandas as pd
import pytest

from diabetes.pipelines.data_engineering.nodes import (
    UNKNOWN_CATEGORY_CODE,
    add_engineered_features,
    add_split_column,
    clean_data,
    clip_outliers,
    fit_encoders,
    fit_imputers,
    fit_outlier_thresholds,
    fit_scalers,
    impute_missing_values,
    transform_encoders,
    transform_scalers,
)

COLUMNS = {
    "target": "Outcome",
    "numerical": ["Glucose", "BMI", "Age"],
    "categorical": [],
    "zero_as_missing": ["Glucose", "BMI"],
    "engineered_numerical": ["NEW_GLUCOSE_AGE"],
    "engineered_categorical": ["NEW_AGE_CAT"],
}

FEATURE_ENGINEERING = {
    "binned": {
        "NEW_AGE_CAT": {
            "source": "Age",
            "bins": [0, 50, 200],
            "labels": ["mature", "senior"],
            "right": False,
        }
    },
    "ranged": {
        "NEW_BMI_SCORE": {
            "source": "BMI",
            "low": 18.5,
            "high": 30.0,
            "inside": "Normal",
            "outside": "Abnormal",
        }
    },
    "combined": {"NEW_AGE_BMI": ["NEW_AGE_CAT", "NEW_BMI_SCORE"]},
    "products": {"NEW_GLUCOSE_AGE": ["Glucose", "Age"]},
}


def _df_com_split() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "train", "train", "test", "validate"],
            "Outcome": [1, 0, 1, 0, 1],
            "Glucose": [100.0, 120.0, 140.0, 900.0, 110.0],
            "BMI": [25.0, 30.0, 35.0, 31.0, 28.0],
            "Age": [30, 45, 60, 22, 51],
        }
    )


class TestCleanData:
    def test_zeros_impossiveis_viram_nulos(self):
        df_in = pd.DataFrame(
            {
                "Outcome": [1, 0],
                "Glucose": [0, 120],
                "BMI": [33.6, 0],
                "Age": [50, 31],
                "ColunaIgnorada": ["x", "y"],
            }
        )

        df_out = clean_data(df_in, COLUMNS)

        assert "ColunaIgnorada" not in df_out.columns
        assert df_out["Glucose"].isna().tolist() == [True, False]
        assert df_out["BMI"].isna().tolist() == [False, True]
        # Age nao esta em zero_as_missing: um zero ali continua sendo zero
        assert df_out["Age"].tolist() == [50, 31]

    def test_funciona_sem_a_coluna_alvo(self):
        """Dados de inferencia podem chegar sem o Outcome."""
        df_in = pd.DataFrame({"Glucose": [120], "BMI": [30.0], "Age": [40]})

        df_out = clean_data(df_in, COLUMNS)

        assert "Outcome" not in df_out.columns
        assert len(df_out) == 1


class TestSplit:
    def test_e_reprodutivel(self):
        df_in = pd.DataFrame({"Glucose": range(500)})
        split = {"train": 0.7, "test": 0.15, "validate": 0.15, "random_state": 42}

        primeira = add_split_column(df_in, split)
        segunda = add_split_column(df_in, split)

        assert set(primeira["split"]) <= {"train", "test", "validate"}
        assert primeira["split"].tolist() == segunda["split"].tolist()

    def test_rejeita_proporcoes_invalidas(self):
        df_in = pd.DataFrame({"Glucose": [1, 2, 3]})
        split = {"train": 0.7, "test": 0.15, "validate": 0.30, "random_state": 42}

        with pytest.raises(ValueError, match="somar 1.0"):
            add_split_column(df_in, split)


class TestImputacao:
    def test_mediana_vem_apenas_do_split_de_treino(self):
        """O coracao da disciplina: nada de vazar teste/validacao para o treino."""
        df_in = pd.DataFrame(
            {
                "split": ["train", "train", "train", "test"],
                "Glucose": [100.0, 200.0, 300.0, 9999.0],
                "BMI": [20.0, 30.0, 40.0, 50.0],
                "Age": [30, 40, 50, 60],
            }
        )
        params = {
            "columns": ["numerical"],
            "strategy": "median",
            "split_to_fit": ["train"],
        }

        imputers = fit_imputers(df_in, COLUMNS, params)

        # mediana de [100, 200, 300] = 200; o 9999 do teste nao entra na conta
        assert imputers["Glucose"] == 200.0

    def test_preenche_os_nulos(self):
        df_in = pd.DataFrame({"Glucose": [np.nan, 150.0], "BMI": [np.nan, 30.0]})

        df_out = impute_missing_values(df_in, {"Glucose": 200.0, "BMI": 25.0})

        assert df_out["Glucose"].tolist() == [200.0, 150.0]
        assert not df_out.isna().to_numpy().any()

    def test_estrategia_desconhecida_falha(self):
        params = {
            "columns": ["numerical"],
            "strategy": "moda",
            "split_to_fit": ["train"],
        }

        with pytest.raises(ValueError, match="Estrategia"):
            fit_imputers(_df_com_split(), COLUMNS, params)


class TestOutliers:
    def test_limites_vem_do_treino_e_truncam_o_extremo(self):
        df_in = _df_com_split()
        params = {
            "columns": ["numerical"],
            "q_low": 0.05,
            "q_high": 0.95,
            "iqr_factor": 1.5,
            "split_to_fit": ["train"],
        }

        thresholds = fit_outlier_thresholds(df_in, COLUMNS, params)
        df_out = clip_outliers(df_in, thresholds)

        assert thresholds["Glucose"]["up"] < 900.0
        assert df_out["Glucose"].max() == thresholds["Glucose"]["up"]
        assert df_out["Glucose"].min() >= thresholds["Glucose"]["low"]


class TestFeatureEngineering:
    def test_cria_as_quatro_familias_de_feature(self):
        df_in = pd.DataFrame(
            {"Glucose": [100.0, 150.0], "BMI": [22.0, 35.0], "Age": [30, 60]}
        )

        df_out = add_engineered_features(df_in, FEATURE_ENGINEERING)

        assert df_out["NEW_AGE_CAT"].tolist() == ["mature", "senior"]  # binned
        assert df_out["NEW_BMI_SCORE"].tolist() == ["Normal", "Abnormal"]  # ranged
        assert df_out["NEW_AGE_BMI"].tolist() == [  # combined
            "mature_Normal",
            "senior_Abnormal",
        ]
        assert df_out["NEW_GLUCOSE_AGE"].tolist() == [3000.0, 9000.0]  # products

    def test_ignora_regra_cuja_coluna_de_origem_nao_existe(self):
        df_in = pd.DataFrame({"Age": [30]})

        df_out = add_engineered_features(df_in, FEATURE_ENGINEERING)

        assert "NEW_AGE_CAT" in df_out.columns
        assert "NEW_BMI_SCORE" not in df_out.columns


class TestEncodersEScalers:
    def test_encoder_nao_aprende_categoria_exclusiva_do_teste(self):
        df_in = pd.DataFrame(
            {
                "split": ["train", "train", "test"],
                "NEW_AGE_CAT": ["mature", "senior", "centenario"],
            }
        )
        params = {"columns": ["engineered_categorical"], "split_to_fit": ["train"]}

        encoders = fit_encoders(df_in, COLUMNS, params)

        assert set(encoders["NEW_AGE_CAT"].classes_) == {"mature", "senior"}

    def test_categoria_desconhecida_nao_derruba_a_inferencia(self):
        """Em producao a API pode receber qualquer valor: vira -1, nao excecao."""
        treino = pd.DataFrame(
            {"split": ["train", "train"], "NEW_AGE_CAT": ["mature", "senior"]}
        )
        params = {"columns": ["engineered_categorical"], "split_to_fit": ["train"]}
        encoders = fit_encoders(treino, COLUMNS, params)

        novo = pd.DataFrame({"NEW_AGE_CAT": ["mature", "desconhecido"]})
        df_out = transform_encoders(novo, encoders)

        assert df_out["NEW_AGE_CAT"].tolist() == [0, UNKNOWN_CATEGORY_CODE]

    def test_scaler_vem_do_yaml_e_padroniza(self):
        df_in = _df_com_split()
        params = {
            "columns": ["numerical"],
            "class_path": "sklearn.preprocessing.RobustScaler",
            "split_to_fit": ["train"],
        }

        scalers = fit_scalers(df_in, COLUMNS, params)
        df_out = transform_scalers(df_in, scalers)

        assert type(scalers["Glucose"]).__name__ == "RobustScaler"
        # a mediana do treino vai para zero
        assert df_out.loc[1, "Glucose"] == pytest.approx(0.0)
