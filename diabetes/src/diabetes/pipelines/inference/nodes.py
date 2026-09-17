"""Nos da pipeline de inferencia.

Reaproveitamento maximo: limpeza, imputacao, tratamento de outliers,
engenharia de features, encoding e escalonamento sao exatamente as mesmas
funcoes da data_engineering, so que alimentadas pelos artefatos de PRODUCAO.
Nada e ajustado aqui -- so duas funcoes sao novas: ``to_dataframe`` (adapta a
entrada, que pode vir de um CSV ou do corpo JSON de uma requisicao HTTP) e
``predict``.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def to_dataframe(
    raw_data: pd.DataFrame | list[dict[str, Any]],
) -> pd.DataFrame:
    """Converte a entrada bruta em DataFrame, se ainda nao for um.

    Args:
        raw_data: DataFrame (por exemplo de um CSVDataset) ou lista de
            dicionarios (por exemplo do corpo de uma requisicao da API).

    Returns:
        DataFrame construido a partir da entrada.
    """
    if isinstance(raw_data, pd.DataFrame):
        return raw_data
    return pd.DataFrame(raw_data)


def predict(
    model_artifact: dict[str, Any],
    inference_data: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Escora um DataFrame processado e devolve predicoes com probabilidades.

    Args:
        model_artifact: Artefato de modelo com ``estimator`` e
            ``feature_columns``.
        inference_data: DataFrame ja imputado, tratado, encodado e escalado.

    Returns:
        Lista de dicionarios com ``index`` (numero da linha), ``prediction``
        (0 ou 1) e ``probability`` (score da classe positiva).

    Raises:
        ValueError: Se faltarem colunas exigidas pelo modelo.
    """
    estimator = model_artifact["estimator"]
    feature_cols = model_artifact["feature_columns"]

    missing = [col for col in feature_cols if col not in inference_data.columns]
    if missing:
        raise ValueError(
            "Faltam colunas exigidas pelo modelo na entrada de inferencia: "
            f"{', '.join(missing)}"
        )

    x = inference_data[feature_cols]
    predictions = estimator.predict(x)
    probabilities = estimator.predict_proba(x)[:, 1]

    result = [
        {"index": idx, "prediction": int(pred), "probability": float(prob)}
        for idx, (pred, prob) in enumerate(
            zip(predictions, probabilities, strict=False)
        )
    ]

    logger.info("Geradas predicoes para %d registros", len(result))
    return result
