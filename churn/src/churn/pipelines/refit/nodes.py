"""Nós da pipeline de refit.

Por que refit? Durante a modelagem treinamos só no split de treino, para que a
avaliação seja honesta. Depois de validar a abordagem, o modelo que vai para
produção é retreinado com TODOS os dados disponíveis — mesma classe, mesmos
hiperparâmetros vencedores, mais dados.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def refit_model(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    optimized_model: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Retreina o modelo otimizado em todos os splits, para produção.

    Extrai a classe do estimador e os melhores hiperparâmetros do artefato
    ``optimized_model`` e ajusta uma instância nova nos splits indicados em
    ``params["train_splits"]`` (normalmente todos).

    Args:
        master_table: DataFrame processado, com a coluna ``split`` e todas as
            colunas de features e alvo.
        columns: Grupos de colunas, com as chaves ``numerical`` e
            ``categorical`` definindo as features.
        optimized_model: Artefato produzido por ``optimize_hyperparameters``,
            cujo ``estimator`` ajustado define classe e hiperparâmetros.
        params: Configuração do refit, com a chave:
            - ``train_splits`` (list[str]): splits usados no treino.

    Returns:
        Artefato de modelo de produção com as chaves ``estimator``,
        ``target_column`` e ``feature_columns``.
    """
    source_estimator = optimized_model["estimator"]
    target = optimized_model["target_column"]
    feature_cols = columns["numerical"] + columns["categorical"]

    train_df = master_table[master_table["split"].isin(params["train_splits"])]
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    estimator = type(source_estimator)(**source_estimator.get_params())
    estimator.fit(X_train, y_train)

    logger.info(
        "Refitted %s on %d samples (%d features) using all splits",
        type(source_estimator).__name__,
        len(X_train),
        X_train.shape[1],
    )

    return {
        "estimator": estimator,
        "target_column": target,
        "feature_columns": feature_cols,
    }
