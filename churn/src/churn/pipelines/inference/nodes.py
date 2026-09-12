"""Nós da pipeline de inferência.

Reaproveitamento máximo: limpeza, encoding e scaling são exatamente as mesmas
funções da data_engineering. Aqui só existem duas funções novas —
``to_dataframe`` (adapta a entrada, que pode vir de um CSV ou de um JSON da
API) e ``predict``. Nada é ajustado nesta pipeline: só se aplicam os artefatos
de produção.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def to_dataframe(
    raw_data: pd.DataFrame | list[dict[str, Any]],
) -> pd.DataFrame:
    """Converte a entrada bruta em DataFrame, se ainda não for um.

    Args:
        raw_data: Dados de entrada, como DataFrame (por exemplo de um
            CSVDataset) ou como lista de dicionários (por exemplo de um
            JSONDataset ou do corpo de uma requisição HTTP).

    Returns:
        DataFrame construído a partir da entrada.
    """
    if isinstance(raw_data, pd.DataFrame):
        return raw_data
    return pd.DataFrame(raw_data)


def predict(
    model_artifact: dict[str, Any],
    inference_data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Escora um DataFrame processado e devolve predições com probabilidades.

    Args:
        model_artifact: Artefato de modelo com as chaves ``estimator`` e
            ``feature_columns``.
        inference_data: DataFrame já encodado e escalado, pronto para escoragem.

    Returns:
        Lista de dicionários, cada um com ``index`` (int, número da linha),
        ``prediction`` (int) e ``probability`` (float, score da classe
        positiva).
    """
    estimator = model_artifact["estimator"]
    feature_cols = model_artifact["feature_columns"]

    X = inference_data[feature_cols]
    predictions = estimator.predict(X)
    probabilities = estimator.predict_proba(X)[:, 1]

    result = [
        {"index": idx, "prediction": int(p), "probability": float(prob)}
        for idx, (p, prob) in enumerate(zip(predictions, probabilities, strict=False))
    ]

    logger.info("Generated predictions for %d samples", len(result))
    return result
