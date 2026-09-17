"""Nos da pipeline de engenharia de dados.

Cada funcao aqui e a traducao de um trecho do notebook
``diabetes-prediction.ipynb`` para codigo de producao, com duas mudancas de
fundo em relacao ao original:

1. **Separacao fit/transform.** No notebook, mediana de imputacao, limites de
   outlier, encoders e scaler sao calculados sobre o dataset inteiro e so
   depois os dados sao separados em treino e teste. Isso vaza informacao do
   teste para o treino. Aqui, cada ``fit_*`` aprende somente nas linhas dos
   splits declarados em ``split_to_fit`` e devolve um artefato; cada
   ``transform_*`` apenas aplica esse artefato.
2. **Funcoes puras.** Nenhuma le ou escreve arquivo (isso e papel do Data
   Catalog) e nenhuma altera o DataFrame de entrada (sempre ``.copy()``).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from diabetes.common import as_list, columns_from_groups, load_class

logger = logging.getLogger(__name__)

# Valor atribuido a uma categoria que o encoder nunca viu no treino. Acontece
# em producao (a API pode receber qualquer coisa) e nao pode derrubar o servico.
UNKNOWN_CATEGORY_CODE = -1


def clean_data(
    raw_diabetes_data: pd.DataFrame,
    columns: dict[str, Any],
) -> pd.DataFrame:
    """Seleciona colunas, converte tipos e marca zeros impossiveis como nulos.

    O dataset Pima usa ``0`` como marcador de ausencia em varias medidas de
    exame: nenhum paciente vivo tem glicose, pressao, espessura de pele,
    insulina ou IMC igual a zero (no arquivo de modelagem sao 314 zeros em
    ``Insulin`` e 192 em ``SkinThickness``). Esses zeros viram ``NaN`` aqui e
    sao imputados mais adiante, pela mediana aprendida no treino.

    Colunas listadas em ``columns`` que nao existam no DataFrame sao ignoradas,
    entao a mesma funcao serve para os dados de treino e para os de inferencia.

    Args:
        raw_diabetes_data: DataFrame bruto de entrada.
        columns: Grupos de colunas. Chaves usadas: ``target``, ``numerical``,
            ``categorical`` e ``zero_as_missing``.

    Returns:
        DataFrame apenas com as colunas declaradas que existem na entrada,
        com numericas convertidas e zeros impossiveis como ``NaN``.
    """
    df_raw = raw_diabetes_data.copy()

    selected = [
        col
        for group in ("target", "numerical", "categorical")
        for col in as_list(columns.get(group))
        if col in df_raw.columns
    ]
    df_flt = df_raw[selected].copy()

    for col in as_list(columns.get("numerical")):
        if col in df_flt.columns:
            df_flt[col] = pd.to_numeric(df_flt[col], errors="coerce")

    for col in as_list(columns.get("categorical")):
        if col in df_flt.columns:
            df_flt[col] = df_flt[col].astype(str).str.strip()

    zeroed = 0
    for col in as_list(columns.get("zero_as_missing")):
        if col in df_flt.columns:
            zeroed += int((df_flt[col] == 0).sum())
            df_flt[col] = df_flt[col].replace(0, np.nan)

    logger.info(
        "Dados limpos: %d linhas, %d colunas, %d zeros impossiveis marcados como nulos",
        len(df_flt),
        len(df_flt.columns),
        zeroed,
    )
    return df_flt


def add_split_column(
    df_in: pd.DataFrame,
    split: dict[str, Any],
) -> pd.DataFrame:
    """Sorteia cada linha para os splits train, test ou validate.

    Args:
        df_in: DataFrame limpo de entrada.
        split: Configuracao com ``train``, ``test``, ``validate`` (proporcoes
            que devem somar 1.0) e ``random_state`` (semente).

    Returns:
        DataFrame de entrada com a coluna ``split`` adicionada.

    Raises:
        ValueError: Se as proporcoes nao somarem 1.0.
    """
    total = split["train"] + split["test"] + split["validate"]
    probs = [split["train"], split["test"], split["validate"]]

    if not np.isclose(total, 1.0):
        raise ValueError(f"As proporcoes do split devem somar 1.0, e somam {total}")

    rng = np.random.default_rng(split["random_state"])
    labels = rng.choice(
        a=["train", "test", "validate"],
        size=len(df_in),
        p=probs,
        replace=True,
    )
    df_out = df_in.assign(split=labels)

    logger.info(
        "Splits - train: %d, test: %d, validate: %d",
        (labels == "train").sum(),
        (labels == "test").sum(),
        (labels == "validate").sum(),
    )
    return df_out


def fit_imputers(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, float]:
    """Aprende o valor de imputacao de cada coluna nos splits configurados.

    Args:
        df_in: DataFrame com a coluna ``split``.
        columns: Dicionario de grupos de colunas.
        params: Configuracao com ``columns`` (grupos a imputar),
            ``strategy`` (``median`` ou ``mean``) e ``split_to_fit``.

    Returns:
        Mapa de nome da coluna para o valor a usar na imputacao.

    Raises:
        ValueError: Se ``strategy`` nao for ``median`` nem ``mean``.
    """
    strategy = params.get("strategy", "median")
    if strategy not in {"median", "mean"}:
        raise ValueError(f"Estrategia de imputacao desconhecida: {strategy}")

    mask = df_in["split"].isin(params["split_to_fit"])
    imputers: dict[str, float] = {}

    for col in columns_from_groups(columns, params["columns"]):
        if col not in df_in.columns:
            continue
        serie = df_in.loc[mask, col]
        value = serie.median() if strategy == "median" else serie.mean()
        imputers[col] = float(value)

    logger.info(
        "Imputadores (%s) ajustados para %d colunas nos splits %s",
        strategy,
        len(imputers),
        params["split_to_fit"],
    )
    return imputers


def impute_missing_values(
    df_in: pd.DataFrame,
    imputers: dict[str, float],
) -> pd.DataFrame:
    """Preenche os nulos com os valores aprendidos por ``fit_imputers``.

    Args:
        df_in: DataFrame com nulos a preencher.
        imputers: Mapa de coluna para valor de imputacao.

    Returns:
        DataFrame sem nulos nas colunas imputadas.
    """
    df_out = df_in.copy()
    filled = 0

    for col, value in imputers.items():
        if col in df_out.columns:
            filled += int(df_out[col].isna().sum())
            df_out[col] = df_out[col].fillna(value)

    logger.info("Imputados %d valores nulos", filled)
    return df_out


def fit_outlier_thresholds(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Calcula os limites de outlier de cada coluna nos splits configurados.

    Replica ``outlier_thresholds`` do notebook: o intervalo interquantil e
    medido entre os quantis ``q_low`` e ``q_high`` e os limites ficam a
    ``iqr_factor`` intervalos de distancia.

    Args:
        df_in: DataFrame com a coluna ``split``.
        columns: Dicionario de grupos de colunas.
        params: Configuracao com ``columns`` (grupos), ``q_low``, ``q_high``,
            ``iqr_factor`` e ``split_to_fit``.

    Returns:
        Mapa de coluna para ``{"low": float, "up": float}``.
    """
    mask = df_in["split"].isin(params["split_to_fit"])
    q_low = params.get("q_low", 0.05)
    q_high = params.get("q_high", 0.95)
    factor = params.get("iqr_factor", 1.5)

    thresholds: dict[str, dict[str, float]] = {}

    for col in columns_from_groups(columns, params["columns"]):
        if col not in df_in.columns:
            continue
        serie = df_in.loc[mask, col]
        quartile_low = serie.quantile(q_low)
        quartile_high = serie.quantile(q_high)
        iqr = quartile_high - quartile_low
        thresholds[col] = {
            "low": float(quartile_low - factor * iqr),
            "up": float(quartile_high + factor * iqr),
        }

    logger.info("Limites de outlier ajustados para %d colunas", len(thresholds))
    return thresholds


def clip_outliers(
    df_in: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Trunca os valores extremos nos limites aprendidos.

    Args:
        df_in: DataFrame a tratar.
        thresholds: Mapa de coluna para ``{"low", "up"}``.

    Returns:
        DataFrame com os valores fora da faixa substituidos pelo limite.
    """
    df_out = df_in.copy()
    clipped = 0

    for col, limits in thresholds.items():
        if col not in df_out.columns:
            continue
        below = (df_out[col] < limits["low"]).sum()
        above = (df_out[col] > limits["up"]).sum()
        clipped += int(below + above)
        df_out[col] = df_out[col].clip(lower=limits["low"], upper=limits["up"])

    logger.info("Truncados %d valores extremos", clipped)
    return df_out


def add_engineered_features(
    df_in: pd.DataFrame,
    feature_engineering: dict[str, Any],
) -> pd.DataFrame:
    """Cria as features derivadas descritas no YAML.

    Quatro tipos de regra, aplicados nesta ordem:

    * ``binned``: faixas de uma coluna numerica (``pd.cut``), por exemplo
      IMC em Underweight/Healthy/Overweight/Obese;
    * ``ranged``: dentro ou fora de uma faixa de referencia, por exemplo
      insulina normal entre 16 e 166;
    * ``combined``: concatenacao de duas categoricas ja criadas, por exemplo
      faixa de IMC com faixa etaria;
    * ``products``: produto de duas numericas (interacoes).

    Args:
        df_in: DataFrame ja imputado e com outliers truncados.
        feature_engineering: Bloco ``feature_engineering`` dos parametros.

    Returns:
        DataFrame com as colunas derivadas adicionadas.
    """
    df_out = df_in.copy()
    created: list[str] = []

    for name, spec in (feature_engineering.get("binned") or {}).items():
        source = spec["source"]
        if source not in df_out.columns:
            continue
        df_out[name] = (
            pd.cut(
                df_out[source],
                bins=spec["bins"],
                labels=spec["labels"],
                right=spec.get("right", True),
            )
            .astype("object")
            .fillna("Unknown")
            .astype(str)
        )
        created.append(name)

    for name, spec in (feature_engineering.get("ranged") or {}).items():
        source = spec["source"]
        if source not in df_out.columns:
            continue
        inside = df_out[source].between(spec["low"], spec["high"])
        df_out[name] = np.where(inside, spec["inside"], spec["outside"])
        created.append(name)

    for name, sources in (feature_engineering.get("combined") or {}).items():
        if not all(col in df_out.columns for col in sources):
            continue
        df_out[name] = df_out[sources[0]].astype(str)
        for col in sources[1:]:
            df_out[name] = df_out[name] + "_" + df_out[col].astype(str)
        created.append(name)

    for name, sources in (feature_engineering.get("products") or {}).items():
        if not all(col in df_out.columns for col in sources):
            continue
        product = df_out[sources[0]].astype(float)
        for col in sources[1:]:
            product = product * df_out[col].astype(float)
        df_out[name] = product
        created.append(name)

    logger.info("Criadas %d features derivadas: %s", len(created), ", ".join(created))
    return df_out


def fit_encoders(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, LabelEncoder]:
    """Ajusta um LabelEncoder por coluna categorica nos splits configurados.

    So as linhas dos splits em ``params["split_to_fit"]`` sao usadas, o que
    impede vazamento de rotulos de teste ou validacao.

    Args:
        df_in: DataFrame com a coluna ``split`` e as categoricas ja criadas.
        columns: Dicionario de grupos de colunas.
        params: Configuracao com ``columns`` (grupos) e ``split_to_fit``.

    Returns:
        Mapa de coluna para o ``LabelEncoder`` ajustado.
    """
    mask = df_in["split"].isin(params["split_to_fit"])
    encoders: dict[str, LabelEncoder] = {}

    for col in columns_from_groups(columns, params["columns"]):
        if col not in df_in.columns:
            continue
        encoder = LabelEncoder()
        encoder.fit(df_in.loc[mask, col].astype(str))
        encoders[col] = encoder

    logger.info("Encoders ajustados para %d colunas", len(encoders))
    return encoders


def transform_encoders(
    df_in: pd.DataFrame,
    encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    """Aplica os encoders ajustados, trocando categorias por inteiros.

    Categorias que o encoder nunca viu (possivel em producao, quando a API
    recebe um registro novo) viram ``-1`` em vez de derrubar a execucao.

    Args:
        df_in: DataFrame com as colunas a encodar.
        encoders: Mapa de coluna para ``LabelEncoder`` ajustado.

    Returns:
        DataFrame com as colunas encodadas como inteiros.
    """
    df_out = df_in.copy()
    unknown_total = 0

    for col, encoder in encoders.items():
        if col not in df_out.columns:
            continue
        mapping = {label: code for code, label in enumerate(encoder.classes_)}
        codes = df_out[col].astype(str).map(mapping)
        unknown_total += int(codes.isna().sum())
        df_out[col] = codes.fillna(UNKNOWN_CATEGORY_CODE).astype(int)

    if unknown_total:
        logger.warning(
            "%d valores categoricos nao vistos no treino foram codificados como %d",
            unknown_total,
            UNKNOWN_CATEGORY_CODE,
        )
    return df_out


def fit_scalers(
    df_in: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Ajusta um scaler por coluna numerica nos splits configurados.

    A classe do scaler vem do YAML (``class_path``); o notebook usa
    ``RobustScaler``, que e menos sensivel a valores extremos.

    Args:
        df_in: DataFrame com a coluna ``split``.
        columns: Dicionario de grupos de colunas.
        params: Configuracao com ``columns`` (grupos), ``class_path`` e
            ``split_to_fit``.

    Returns:
        Mapa de coluna para o scaler ajustado.
    """
    mask = df_in["split"].isin(params["split_to_fit"])
    scaler_class = load_class(
        params.get("class_path", "sklearn.preprocessing.RobustScaler")
    )
    scalers: dict[str, Any] = {}

    for col in columns_from_groups(columns, params["columns"]):
        if col not in df_in.columns:
            continue
        scaler = scaler_class()
        scaler.fit(df_in.loc[mask, [col]])
        scalers[col] = scaler

    logger.info(
        "Scalers (%s) ajustados para %d colunas",
        scaler_class.__name__,
        len(scalers),
    )
    return scalers


def transform_scalers(
    df_in: pd.DataFrame,
    scalers: dict[str, Any],
) -> pd.DataFrame:
    """Aplica os scalers ajustados, padronizando as colunas numericas.

    Args:
        df_in: DataFrame com as colunas a escalar.
        scalers: Mapa de coluna para scaler ajustado.

    Returns:
        DataFrame com as colunas escaladas.
    """
    df_out = df_in.copy()

    for col, scaler in scalers.items():
        if col in df_out.columns:
            df_out[col] = scaler.transform(df_out[[col]])

    return df_out
