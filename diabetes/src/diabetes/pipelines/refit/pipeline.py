"""Pipeline de refit (preparacao dos artefatos de producao).

Repare que todos os nos, menos o ultimo, importam as MESMAS funcoes da
data_engineering: o que muda sao os parametros (``split_to_fit`` com todos os
splits). Mesma logica, outra configuracao.

DAG:

    split_diabetes_data  -> [refit_imputers]           -> production_imputers
    split + imputers     -> [refit_impute]             -> refit_imputed_data
    refit_imputed        -> [refit_outlier_thresholds] -> production_outlier_thresholds
    refit_imputed + thr  -> [refit_clip]               -> refit_clipped_data
    refit_clipped        -> [refit_features]           -> refit_featured_data
    refit_featured       -> [refit_encoders]           -> production_encoders
    refit_featured       -> [refit_scalers]            -> production_scalers
    master_table + otim. -> [refit_model]              -> production_model
"""

from kedro.pipeline import Node, Pipeline

from diabetes.pipelines.data_engineering.nodes import (
    add_engineered_features,
    clip_outliers,
    fit_encoders,
    fit_imputers,
    fit_outlier_thresholds,
    fit_scalers,
    impute_missing_values,
)

from .nodes import refit_model


def create_pipeline(**kwargs) -> Pipeline:
    """Monta a pipeline de refit dos artefatos de producao."""
    return Pipeline(
        [
            Node(
                func=fit_imputers,
                inputs=[
                    "split_diabetes_data",
                    "params:columns",
                    "params:refit_imputers",
                ],
                outputs="production_imputers",
                name="refit_imputers",
            ),
            Node(
                func=impute_missing_values,
                inputs=["split_diabetes_data", "production_imputers"],
                outputs="refit_imputed_data",
                name="refit_impute_missing_values",
            ),
            Node(
                func=fit_outlier_thresholds,
                inputs=[
                    "refit_imputed_data",
                    "params:columns",
                    "params:refit_outliers",
                ],
                outputs="production_outlier_thresholds",
                name="refit_outlier_thresholds",
            ),
            Node(
                func=clip_outliers,
                inputs=["refit_imputed_data", "production_outlier_thresholds"],
                outputs="refit_clipped_data",
                name="refit_clip_outliers",
            ),
            Node(
                func=add_engineered_features,
                inputs=["refit_clipped_data", "params:feature_engineering"],
                outputs="refit_featured_data",
                name="refit_add_engineered_features",
            ),
            Node(
                func=fit_encoders,
                inputs=[
                    "refit_featured_data",
                    "params:columns",
                    "params:refit_encoders",
                ],
                outputs="production_encoders",
                name="refit_encoders",
            ),
            Node(
                func=fit_scalers,
                inputs=[
                    "refit_featured_data",
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
