"""Nós da pipeline de modelagem.

O modelo não está escrito no código: ele vem do YAML em ``class_path``. Trocar
LogisticRegression por CatBoost, XGBoost ou qualquer estimador compatível com a
API do scikit-learn é mudança de configuração, não de código.

Os nós devolvem um "artefato de modelo" (dict) em vez de só o estimador. Assim
o nó de avaliação recebe junto tudo o que precisa (coluna alvo, features,
splits de avaliação) e serve tanto para o baseline quanto para o otimizado.
"""

import importlib
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


def _load_class(class_path: str) -> type:
    """Importa dinamicamente uma classe a partir do caminho completo.

    Args:
        class_path: Caminho completo da classe, por exemplo
            ``"sklearn.linear_model.LogisticRegression"``.

    Returns:
        O objeto de classe importado.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def train_model(
    master_table: pd.DataFrame,
    columns: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Treina um classificador nos splits indicados e devolve o artefato.

    Args:
        master_table: DataFrame processado, com a coluna ``split`` e todas as
            colunas de features e alvo.
        columns: Grupos de colunas, com as chaves ``numerical`` e
            ``categorical`` definindo as features.
        params: Configuração de treino, com as chaves:
            - ``class_path`` (str): classe do estimador.
            - ``init_args`` (dict, opcional): argumentos do construtor.
            - ``target_column`` (str): nome da coluna alvo.
            - ``train_splits`` (list[str]): splits usados no ajuste.
            - ``eval_splits`` (list[str], opcional): splits de avaliação.

    Returns:
        Artefato de modelo com as chaves ``estimator``, ``target_column``,
        ``feature_columns`` e ``eval_splits``.
    """
    target = params["target_column"]
    train_df = master_table[master_table["split"].isin(params["train_splits"])]
    feature_cols = columns["numerical"] + columns["categorical"]

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    cls = _load_class(params["class_path"])
    estimator = cls(**params.get("init_args", {}))
    estimator.fit(X_train, y_train)

    logger.info(
        "Trained %s on %d samples (%d features)",
        params["class_path"].rsplit(".", 1)[-1],
        len(X_train),
        X_train.shape[1],
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
    """Roda um grid search com validação cruzada e devolve o melhor modelo.

    O artefato devolvido tem a mesma estrutura de ``train_model``, então pode
    ser passado direto para ``evaluate_model``.

    Args:
        master_table: DataFrame processado, com a coluna ``split`` e todas as
            colunas de features e alvo.
        columns: Grupos de colunas, com as chaves ``numerical`` e
            ``categorical`` definindo as features.
        params: Configuração da otimização, com as chaves:
            - ``class_path`` (str): classe do estimador.
            - ``init_args`` (dict, opcional): argumentos base do construtor.
            - ``param_grid`` (dict): grade de hiperparâmetros do GridSearchCV.
            - ``cv`` (int): número de folds da validação cruzada.
            - ``scoring`` (str): métrica usada na busca.
            - ``target_column`` (str): nome da coluna alvo.
            - ``train_splits`` (list[str]): splits usados no ajuste.
            - ``eval_splits`` (list[str]): splits de avaliação.

    Returns:
        Artefato de modelo com as chaves ``estimator``, ``target_column``,
        ``feature_columns``, ``eval_splits``, ``best_params`` e
        ``cv_best_score``.
    """
    target = params["target_column"]
    train_df = master_table[master_table["split"].isin(params["train_splits"])]
    feature_cols = columns["numerical"] + columns["categorical"]

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    cls = _load_class(params["class_path"])
    base_estimator = cls(**params.get("init_args", {}))

    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=params["param_grid"],
        cv=params.get("cv", 5),
        scoring=params.get("scoring", "roc_auc"),
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Grid search complete - best %s: %.4f, params: %s",
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

    Args:
        model_artifact: Dict produzido por ``train_model`` ou
            ``optimize_hyperparameters``, contendo ``estimator``,
            ``target_column``, ``feature_columns`` e ``eval_splits``.
        master_table: DataFrame processado, com a coluna ``split`` e todas as
            colunas de features e alvo.

    Returns:
        Dict indexado pelo nome do split. Cada valor contém ``accuracy``,
        ``roc_auc``, ``f1_macro``, ``classification_report`` e ``n_samples``.
    """
    estimator = model_artifact["estimator"]
    target = model_artifact["target_column"]
    feature_cols = model_artifact["feature_columns"]
    eval_splits = model_artifact["eval_splits"]

    all_metrics: dict[str, Any] = {}

    for split_name in eval_splits:
        split_df = master_table[master_table["split"] == split_name]
        X_split = split_df[feature_cols]
        y_split = split_df[target]

        y_pred = estimator.predict(X_split)
        y_proba = estimator.predict_proba(X_split)[:, 1]

        split_metrics = {
            "accuracy": float(accuracy_score(y_split, y_pred)),
            "roc_auc": float(roc_auc_score(y_split, y_proba)),
            "f1_macro": float(f1_score(y_split, y_pred, average="macro")),
            "classification_report": classification_report(
                y_split, y_pred, output_dict=True
            ),
            "n_samples": int(len(y_split)),
        }

        logger.info(
            "Evaluation '%s' - accuracy: %.4f, roc_auc: %.4f, f1_macro: %.4f",
            split_name,
            split_metrics["accuracy"],
            split_metrics["roc_auc"],
            split_metrics["f1_macro"],
        )

        all_metrics[split_name] = split_metrics

    return all_metrics
