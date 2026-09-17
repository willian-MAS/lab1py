"""Pipeline de modelagem.

DAG:

    master_table -> [train_baseline_model]      -> baseline_model
    baseline     -> [evaluate_baseline_model]   -> baseline_metrics
    master_table -> [optimize_hyperparameters]  -> optimized_model
    optimized    -> [evaluate_optimized_model]  -> optimized_metrics

O mesmo ``evaluate_model`` avalia os dois modelos: o artefato carrega tudo o
que a avaliacao precisa.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import evaluate_model, optimize_hyperparameters, train_model


def create_pipeline(**kwargs) -> Pipeline:
    """Monta a pipeline de treino, avaliacao e otimizacao."""
    return Pipeline(
        [
            Node(
                func=train_model,
                inputs=["master_table", "params:columns", "params:modelling_baseline"],
                outputs="baseline_model",
                name="train_baseline_model",
            ),
            Node(
                func=evaluate_model,
                inputs=["baseline_model", "master_table"],
                outputs="baseline_metrics",
                name="evaluate_baseline_model",
            ),
            Node(
                func=optimize_hyperparameters,
                inputs=[
                    "master_table",
                    "params:columns",
                    "params:modelling_optimization",
                ],
                outputs="optimized_model",
                name="optimize_hyperparameters",
            ),
            Node(
                func=evaluate_model,
                inputs=["optimized_model", "master_table"],
                outputs="optimized_metrics",
                name="evaluate_optimized_model",
            ),
        ]
    )
