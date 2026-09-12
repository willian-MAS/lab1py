"""Pipeline de engenharia de dados.

DAG:
    raw_churn_data      -> [clean_data]                   -> cleaned_churn_data
    cleaned_churn_data  -> [add_split_column]             -> split_churn_data
    split_churn_data    -> [fit_encoders]                 -> modelling_encoders
    split + encoders    -> [encode_categorical_features]  -> encoded_churn_data
    encoded_churn_data  -> [fit_scalers]                  -> modelling_scalers
    encoded + scalers   -> [scale_numerical_features]     -> master_table
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    add_split_column,
    clean_data,
    fit_encoders,
    fit_scalers,
    transform_encoders,
    transform_scalers,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Monta a pipeline de engenharia de dados de treino."""
    return Pipeline(
        [
            Node(
                func=clean_data,
                inputs=["raw_churn_data", "params:columns"],
                outputs="cleaned_churn_data",
                name="clean_data",
            ),
            Node(
                func=add_split_column,
                inputs=["cleaned_churn_data", "params:split"],
                outputs="split_churn_data",
                name="add_split_column",
            ),
            Node(
                func=fit_encoders,
                inputs=[
                    "split_churn_data",
                    "params:columns",
                    "params:modelling_encoders",
                ],
                outputs="modelling_encoders",
                name="fit_encoders",
            ),
            Node(
                func=transform_encoders,
                inputs=["split_churn_data", "modelling_encoders"],
                outputs="encoded_churn_data",
                name="encode_categorical_features",
            ),
            Node(
                func=fit_scalers,
                inputs=[
                    "encoded_churn_data",
                    "params:columns",
                    "params:modelling_scalers",
                ],
                outputs="modelling_scalers",
                name="fit_scalers",
            ),
            Node(
                func=transform_scalers,
                inputs=["encoded_churn_data", "modelling_scalers"],
                outputs="master_table",
                name="scale_numerical_features",
            ),
        ]
    )
