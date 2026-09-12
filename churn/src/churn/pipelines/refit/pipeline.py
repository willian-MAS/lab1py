"""Pipeline de refit (preparação dos artefatos de produção).

Repare que os nós de encoder/scaler importam as MESMAS funções da
data_engineering: o que muda são só os parâmetros (`split_to_fit` com todos os
splits). Mesma lógica, outra configuração.

DAG:
    split_churn_data -> [refit_encoders] -> production_encoders
    split_churn_data -> [refit_scalers]  -> production_scalers
    master_table + optimized_model -> [refit_model] -> production_model
"""

from kedro.pipeline import Node, Pipeline

from churn.pipelines.data_engineering.nodes import fit_encoders, fit_scalers

from .nodes import refit_model


def create_pipeline(**kwargs) -> Pipeline:
    """Monta a pipeline de refit dos artefatos de produção."""
    return Pipeline(
        [
            Node(
                func=fit_encoders,
                inputs=[
                    "split_churn_data",
                    "params:columns",
                    "params:refit_encoders",
                ],
                outputs="production_encoders",
                name="refit_encoders",
            ),
            Node(
                func=fit_scalers,
                inputs=[
                    "split_churn_data",
                    "params:columns",
                    "params:refit_scalers",
                ],
                outputs="production_scalers",
                name="refit_scalers",
            ),
            Node(
                func=refit_model,
                inputs=[
                    "master_table",
                    "params:columns",
                    "optimized_model",
                    "params:refit_model",
                ],
                outputs="production_model",
                name="refit_model",
            ),
        ]
    )
