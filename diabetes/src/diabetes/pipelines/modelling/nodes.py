"""Nos da pipeline de modelagem.

O modelo nao esta escrito no codigo: ele vem do YAML, em ``class_path``.
Trocar LogisticRegression por RandomForest, XGBoost ou qualquer estimador
compativel com a API do scikit-learn e mudanca de configuracao.

Os nos devolvem um "artefato de modelo" (dict) em vez de so o estimador. Com
isso o no de avaliacao recebe junto tudo o que precisa -- coluna alvo, colunas
de feature e splits de avaliacao -- e a mesma funcao avalia o baseline e o
modelo otimizado.
"""

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from diabetes.common import columns_from_groups, load_class

logger = logging.getLogger(__name__)


def train_model(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Treina um classificador nos splits indicados e devolve o artefato.

    Args:
        master_table: DataFrame processado, com a coluna ``split`` e todas as
            colunas de feature e alvo.
        columns: Dicionario de grupos de colunas.
        params: Configuracao de treino, com as chaves:
            - ``class_path`` (str): classe do estimador;
            - ``feature_groups`` (list[str]): grupos de ``columns`` usados
              como features;
            - ``init_args`` (dict, opcional): argumentos do construtor;
            - ``target_column`` (str): nome da coluna alvo;
            - ``train_splits`` (list[str]): splits usados no ajuste;
            - ``eval_splits`` (list[str], opcional): splits de avaliacao.

    Returns:
        Artefato com ``estimator``, ``target_column``, ``feature_columns`` e
        ``eval_splits``.
    """
    target = params["target_column"]
    feature_cols = columns_from_groups(columns, params["feature_groups"])
    train_df = master_table[master_table["split"].isin(params["train_splits"])]

    x_train = train_df[feature_cols]
    y_train = train_df[target]

    estimator_class = load_class(params["class_path"])
    estimator = estimator_class(**params.get("init_args", {}))
    estimator.fit(x_train, y_train)

    logger.info(
        "Treinado %s com %d amostras e %d features",
        params["class_path"].rsplit(".", 1)[-1],
        len(x_train),
        x_train.shape[1],
    )

    return {
        "estimator": estimator,
        "target_column": target,
        "feature_columns": feature_cols,
        "eval_splits": params.get("eval_splits", []),
    }


def optimize_hyperparameters(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Roda um grid search com validacao cruzada e devolve o melhor modelo.

    O artefato devolvido tem a mesma estrutura de ``train_model``, entao pode
    ser passado direto para ``evaluate_model``.

    Args:
        master_table: DataFrame processado, com a coluna ``split``.
        columns: Dicionario de grupos de colunas.
        params: Alem das chaves de ``train_model``, aceita ``param_grid``
            (grade de hiperparametros), ``cv`` (numero de folds) e
            ``scoring`` (metrica da busca).

    Returns:
        Artefato com ``estimator``, ``target_column``, ``feature_columns``,
        ``eval_splits``, ``best_params`` e ``cv_best_score``.
    """
    target = params["target_column"]
    feature_cols = columns_from_groups(columns, params["feature_groups"])
    train_df = master_table[master_table["split"].isin(params["train_splits"])]

    x_train = train_df[feature_cols]
    y_train = train_df[target]

    estimator_class = load_class(params["class_path"])
    base_estimator = estimator_class(**params.get("init_args", {}))

    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=params["param_grid"],
        cv=params.get("cv", 5),
        scoring=params.get("scoring", "roc_auc"),
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)

    logger.info(
        "Grid search concluido - melhor %s: %.4f, parametros: %s",
        params.get("scoring", "roc_auc"),
        search.best_score_,
        search.best_params_,
    )

    return {
        "estimator": search.best_estimator_,
        "target_column": target,
        "feature_columns": feature_cols,
        "eval_splits": params["eval_splits"],
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
    }


def evaluate_model(
    model_artifact: dict[str, Any],
    master_table: pd.DataFrame,
) -> dict[str, Any]:
    """Avalia um modelo ajustado em cada split listado no artefato.

    As metricas sao as mesmas que o notebook compara entre modelos:
    acuracia, recall, precisao, F1 e AUC.

    Args:
        model_artifact: Dict produzido por ``train_model`` ou
            ``optimize_hyperparameters``.
        master_table: DataFrame processado, com a coluna ``split``.

    Returns:
        Dict indexado pelo nome do split, com ``accuracy``, ``recall``,
        ``precision``, ``f1``, ``roc_auc``, ``classification_report`` e
        ``n_samples``.
    """
    estimator = model_artifact["estimator"]
    target = model_artifact["target_column"]
    feature_cols = model_artifact["feature_columns"]
    eval_splits = model_artifact["eval_splits"]

    all_metrics: dict[str, Any] = {}

    for split_name in eval_splits:
        split_df = master_table[master_table["split"] == split_name]
        x_split = split_df[feature_cols]
        y_split = split_df[target]

        y_pred = estimator.predict(x_split)
        y_proba = estimator.predict_proba(x_split)[:, 1]

        split_metrics = {
            "accuracy": float(accuracy_score(y_split, y_pred)),
            "recall": float(recall_score(y_split, y_pred, zero_division=0)),
            "precision": float(precision_score(y_split, y_pred, zero_division=0)),
            "f1": float(f1_score(y_split, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_split, y_proba)),
            "classification_report": classification_report(
                y_split, y_pred, output_dict=True, zero_division=0
            ),
            "n_samples": int(len(y_split)),
        }

        logger.info(
            "Avaliacao '%s' - accuracy: %.4f, recall: %.4f, precision: %.4f, "
            "f1: %.4f, roc_auc: %.4f",
            split_name,
            split_metrics["accuracy"],
            split_metrics["recall"],
            split_metrics["precision"],
            split_metrics["f1"],
            split_metrics["roc_auc"],
        )
        all_metrics[split_name] = split_metrics

    return all_metrics
