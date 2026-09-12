"""Nós da pipeline de engenharia de dados.

Padrão central da aula: separar `fit` de `transform`.

* `fit_*`  aprende os artefatos (encoders/scalers) SÓ nos splits configurados
  em ``split_to_fit`` — é isso que evita data leakage;
* `transform_*` aplica artefatos já ajustados a qualquer conjunto de dados.

Todas as funções são puras: recebem e devolvem objetos em memória, nunca leem
nem escrevem arquivos (isso é responsabilidade do Data Catalog).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def clean_data(
    raw_churn_data: pd.DataFrame,
    columns: dict[str, Any],
) -> pd.DataFrame:
    """Seleciona, converte e imputa as colunas de um DataFrame bruto.

    Colunas listadas em ``columns`` que não existirem no DataFrame são
    simplesmente ignoradas, então a mesma função serve para os dados de
    treino (que têm a coluna target) e para os de inferência (que não têm).
    Colunas numéricas são convertidas para float com NaN preenchido por 0;
    colunas categóricas são convertidas para string sem espaços nas pontas,
    com NaN preenchido por ``"Unknown"``.

    Args:
        raw_churn_data: DataFrame de entrada bruto.
        columns: Grupos de colunas a processar. Chaves reconhecidas:
            - ``target`` (str, opcional): nome da coluna alvo.
            - ``numerical`` (list[str]): colunas numéricas.
            - ``categorical`` (list[str]): colunas categóricas.

    Returns:
        DataFrame limpo contendo apenas as colunas especificadas que
        existem na entrada.
    """
    df_raw = raw_churn_data.copy()

    all_columns = [
        item
        for v in columns.values()
        for item in (v if isinstance(v, list) else [v])
        if item in df_raw.columns
    ]

    df_flt = df_raw[all_columns].copy()

    for c in columns["numerical"]:
        df_flt[c] = pd.to_numeric(df_flt[c], errors="coerce").fillna(0)

    for c in columns["categorical"]:
        df_flt[c] = df_flt[c].astype(str).str.strip().fillna("Unknown")

    logger.info("Cleaned data: %d rows, %d columns", len(df_flt), len(df_flt.columns))

    return df_flt


def add_split_column(
    df_in: pd.DataFrame,
    split: dict[str, Any],
) -> pd.DataFrame:
    """Sorteia cada linha para os splits train, test ou validate.

    Args:
        df_in: DataFrame limpo de entrada.
        split: Configuração do split, com as chaves:
            - ``train`` (float): proporção de linhas de treino.
            - ``test`` (float): proporção de linhas de teste.
            - ``validate`` (float): proporção de linhas de validação.
            - ``random_state`` (int): semente para sorteio reprodutível.

    Returns:
        DataFrame de entrada com a coluna ``split`` adicionada, cujos
        valores são ``"train"``, ``"test"`` ou ``"validate"``.

    Raises:
        ValueError: Se ``train + test + validate`` não somar 1.0.
    """
    total = split["train"] + split["test"] + split["validate"]
    probs = [split["train"], split["test"], split["validate"]]

    if not np.isclose(total, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0, got {total}")

    rng = np.random.default_rng(split["random_state"])

    labels = rng.choice(
        a=["train", "test", "validate"],
        size=len(df_in),
        p=probs,
        replace=True,
    )

    df_out = df_in.assign(split=labels)

    logger.info(
        "Split counts - train: %d, test: %d, validate: %d",
        (labels == "train").sum(),
        (labels == "test").sum(),
        (labels == "validate").sum(),
    )

    return df_out


def fit_encoders(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, LabelEncoder]:
    """Ajusta um LabelEncoder por coluna categórica nos splits indicados.

    Só as linhas dos splits em ``params["split_to_fit"]`` são usadas, o que
    impede vazamento de rótulos dos dados de teste ou validação.

    Args:
        df_in: DataFrame com a coluna ``split`` e as colunas de features.
        columns: Dicionário de grupos de colunas.
        params: Configuração dos encoders, com as chaves:
            - ``columns`` (list[str]): grupos de ``columns`` a encodar.
            - ``split_to_fit`` (list[str]): splits usados no ajuste.

    Returns:
        Mapa de nome da coluna para o ``LabelEncoder`` ajustado.
    """
    encoders: dict[str, LabelEncoder] = {}

    cols_to_encode = [
        item
        for p in params["columns"]
        for item in (columns[p] if isinstance(columns[p], list) else [columns[p]])
    ]

    for col in cols_to_encode:
        le = LabelEncoder()
        le.fit(df_in.loc[df_in["split"].isin(params["split_to_fit"]), col].astype(str))
        encoders[col] = le

    logger.info("Fitted label encoders for %d columns", len(encoders))
    return encoders


def transform_encoders(
    df_in: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Aplica LabelEncoders ajustados, trocando categorias por inteiros.

    Args:
        df_in: DataFrame com as colunas a encodar.
        encoders: Mapa de coluna para ``LabelEncoder`` ajustado.

    Returns:
        DataFrame com cada coluna encodada substituída pelos inteiros.
    """
    df_out = df_in.copy()

    for col, en in encoders.items():
        if col in df_out.columns:
            df_out[col] = en.transform(df_out[col].astype(str))

    return df_out


def fit_scalers(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, StandardScaler]:
    """Ajusta um StandardScaler por coluna numérica nos splits indicados.

    Só as linhas dos splits em ``params["split_to_fit"]`` são usadas, o que
    impede vazamento de estatísticas (média/desvio) de teste ou validação.

    Args:
        df_in: DataFrame com a coluna ``split`` e as colunas de features.
        columns: Dicionário de grupos de colunas.
        params: Configuração dos scalers, com as chaves:
            - ``columns`` (list[str]): grupos de ``columns`` a escalar.
            - ``split_to_fit`` (list[str]): splits usados no ajuste.

    Returns:
        Mapa de nome da coluna para o ``StandardScaler`` ajustado.
    """
    scalers: dict[str, StandardScaler] = {}

    cols_to_scale = [
        item
        for p in params["columns"]
        for item in (columns[p] if isinstance(columns[p], list) else [columns[p]])
    ]

    for col in cols_to_scale:
        scaler = StandardScaler()
        scaler.fit(df_in.loc[df_in["split"].isin(params["split_to_fit"]), [col]])
        scalers[col] = scaler

    logger.info("Fitted scalers for %d columns", len(scalers))
    return scalers


def transform_scalers(
    df_in: pd.DataFrame,
    scalers: dict[str, StandardScaler],
) -> pd.DataFrame:
    """Aplica StandardScalers ajustados, padronizando as colunas numéricas.

    Args:
        df_in: DataFrame com as colunas a escalar.
        scalers: Mapa de coluna para ``StandardScaler`` ajustado.

    Returns:
        DataFrame com cada coluna escalada substituída pelos valores
        padronizados.
    """
    df_out = df_in.copy()

    for col, scaler in scalers.items():
        if col in df_out.columns:
            df_out[col] = scaler.transform(df_out[[col]])

    return df_out
