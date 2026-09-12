"""Testes unitários dos nós de engenharia de dados.

Nós são funções puras: dá para testá-los com DataFrames sintéticos, sem
Kedro, sem catálogo e sem arquivos.
"""

import pandas as pd
import pytest

from churn.pipelines.data_engineering.nodes import (
    add_split_column,
    clean_data,
    fit_encoders,
    transform_encoders,
)

COLUMNS = {
    "target": "Churn",
    "numerical": ["tenure", "MonthlyCharges"],
    "categorical": ["Contract"],
}


def test_clean_data_seleciona_colunas_e_converte_numericos():
    df_in = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "Churn": ["Yes", "No"],
            "tenure": ["12", "nao-numerico"],
            "MonthlyCharges": [10.5, 20.0],
            "Contract": [" Two year ", "Month-to-month"],
        }
    )

    df_out = clean_data(df_in, COLUMNS)

    # coluna fora dos grupos configurados é descartada
    assert "customerID" not in df_out.columns
    # valor não numérico vira 0 em vez de explodir
    assert df_out["tenure"].tolist() == [12.0, 0.0]
    # categórica vem sem espaços nas pontas
    assert df_out["Contract"].tolist() == ["Two year", "Month-to-month"]


def test_clean_data_funciona_sem_a_coluna_target():
    """Dados de inferência não têm target - a mesma função deve servir."""
    df_in = pd.DataFrame(
        {
            "tenure": [1],
            "MonthlyCharges": [2.0],
            "Contract": ["Two year"],
        }
    )

    df_out = clean_data(df_in, COLUMNS)

    assert "Churn" not in df_out.columns
    assert len(df_out) == 1


def test_add_split_column_e_reprodutivel():
    df_in = pd.DataFrame({"tenure": range(1000)})
    split = {"train": 0.7, "test": 0.15, "validate": 0.15, "random_state": 42}

    primeira = add_split_column(df_in, split)
    segunda = add_split_column(df_in, split)

    assert set(primeira["split"].unique()) <= {"train", "test", "validate"}
    # mesma semente, mesmo resultado: determinismo
    assert primeira["split"].tolist() == segunda["split"].tolist()


def test_add_split_column_rejeita_proporcoes_invalidas():
    df_in = pd.DataFrame({"tenure": [1, 2, 3]})
    split = {"train": 0.7, "test": 0.15, "validate": 0.30, "random_state": 42}

    with pytest.raises(ValueError, match="must sum to 1.0"):
        add_split_column(df_in, split)


def test_encoders_sao_ajustados_somente_no_split_configurado():
    """O coração da aula: sem data leakage."""
    df_in = pd.DataFrame(
        {
            "split": ["train", "train", "test"],
            "Churn": ["Yes", "No", "Yes"],
            "Contract": ["Two year", "One year", "Month-to-month"],
        }
    )

    encoders = fit_encoders(
        df_in,
        COLUMNS,
        {"columns": ["target", "categorical"], "split_to_fit": ["train"]},
    )

    # a categoria que só aparece no split de teste não foi aprendida
    assert set(encoders["Contract"].classes_) == {"One year", "Two year"}
    assert set(encoders["Churn"].classes_) == {"No", "Yes"}


def test_transform_encoders_substitui_categorias_por_inteiros():
    df_in = pd.DataFrame(
        {
            "split": ["train", "train"],
            "Churn": ["Yes", "No"],
            "Contract": ["Two year", "One year"],
        }
    )
    encoders = fit_encoders(
        df_in,
        COLUMNS,
        {"columns": ["target", "categorical"], "split_to_fit": ["train"]},
    )

    df_out = transform_encoders(df_in, encoders)

    assert df_out["Churn"].tolist() == [1, 0]
    assert df_out["Contract"].tolist() == [1, 0]
