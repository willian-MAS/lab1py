"""Nos da pipeline de refit.

Por que refit? Durante a modelagem, tudo e ajustado somente no split de treino,
para que a avaliacao seja honesta. Depois de validada a abordagem, os artefatos
que vao para producao sao reajustados com TODOS os dados disponiveis: mesma
classe de modelo, mesmos hiperparametros vencedores, mais dados.
"""

import logging
from typing import Any

import pandas as pd

from diabetes.common import columns_from_groups

logger = logging.getLogger(__name__)


def refit_model(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    optimized_model: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Retreina o modelo otimizado em todos os splits, para producao.

    Extrai a classe e os hiperparametros vencedores do artefato
    ``optimized_model`` e ajusta uma instancia nova nos splits indicados.

    Args:
        master_table: DataFrame processado, com a coluna ``split``.
        columns: Dicionario de grupos de colunas.
        optimized_model: Artefato produzido por ``optimize_hyperparameters``.
        params: Configuracao com ``feature_groups`` e ``train_splits``.

    Returns:
        Artefato de producao com ``estimator``, ``target_column``,
        ``feature_columns`` e ``trained_on``.
    """
    source_estimator = optimized_model["estimator"]
    target = optimized_model["target_column"]
    feature_cols = columns_from_groups(columns, params["feature_groups"])

    train_df = master_table[master_table["split"].isin(params["train_splits"])]
    x_train = train_df[feature_cols]
    y_train = train_df[target]

    estimator = type(source_estimator)(**source_estimator.get_params())
    estimator.fit(x_train, y_train)

    logger.info(
        "Modelo de producao: %s reajustado com %d amostras e %d features (splits %s)",
        type(source_estimator).__name__,
        len(x_train),
        x_train.shape[1],
        params["train_splits"],
    )

    return {
        "estimator": estimator,
        "target_column": target,
        "feature_columns": feature_cols,
        "trained_on": params["train_splits"],
    }
