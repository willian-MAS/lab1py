"""Pipeline de engenharia de dados.

DAG (o Kedro infere a ordem pelos nomes de entrada e saida):

    raw_diabetes_data      -> [clean_data]                  -> cleaned_diabetes_data
    cleaned_diabetes_data  -> [add_split_column]            -> split_diabetes_data
    split_diabetes_data    -> [fit_imputers]                -> modelling_imputers
    split + imputers       -> [impute_missing_values]       -> imputed_diabetes_data
    imputed                -> [fit_outlier_thresholds]      -> modelling_outlier_thresholds
    imputed + thresholds   -> [clip_outliers]               -> clipped_diabetes_data
    clipped                -> [add_engineered_features]     -> featured_diabetes_data
    featured               -> [fit_encoders]                -> modelling_encoders
    featured + encoders    -> [encode_categorical_features] -> encoded_diabetes_data
    encoded                -> [fit_scalers]                 -> modelling_scalers
    encoded + scalers      -> [scale_numerical_features]    -> master_table
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    add_engineered_features,
    add_split_column,
    clean_data,
    clip_outliers,
    fit_encoders,
    fit_imputers,
    fit_outlier_thresholds,
    fit_scalers,
    impute_missing_values,
    transform_encoders,
    transform_scalers,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Monta a pipeline de engenharia de dados de treino."""
    return Pipeline(
        [
            Node(
                func=clean_data,
                inputs=["raw_diabetes_data", "params:columns"],
                outputs="cleaned_diabetes_data",
                name="clean_data",
            ),
            Node(
                func=add_split_column,
                inputs=["cleaned_diabetes_data", "params:split"],
                outputs="split_diabetes_data",
                name="add_split_column",
            ),
            Node(
                func=fit_imputers,
                inputs=[
                    "split_diabetes_data",
                    "params:columns",
                    "params:modelling_imputers",
                ],
                outputs="modelling_imputers",
                name="fit_imputers",
            ),
            Node(
                func=impute_missing_values,
                inputs=["split_diabetes_data", "modelling_imputers"],
                outputs="imputed_diabetes_data",
                name="impute_missing_values",
            ),
            Node(
                func=fit_outlier_thresholds,
                inputs=[
                    "imputed_diabetes_data",
                    "params:columns",
                    "params:modelling_outliers",
                ],
                outputs="modelling_outlier_thresholds",
                name="fit_outlier_thresholds",
            ),
            Node(
                func=clip_outliers,
                inputs=["imputed_diabetes_data", "modelling_outlier_thresholds"],
                outputs="clipped_diabetes_data",
                name="clip_outliers",
            ),
            Node(
                func=add_engineered_features,
                inputs=["clipped_diabetes_data", "params:feature_engineering"],
                outputs="featured_diabetes_data",
                name="add_engineered_features",
            ),
            Node(
                func=fit_encoders,
                inputs=[
                    "featured_diabetes_data",
                    "params:columns",
                    "params:modelling_encoders",
                ],
                outputs="modelling_encoders",
                name="fit_encoders",
            ),
            Node(
                func=transform_encoders,
                inputs=["featured_diabetes_data", "modelling_encoders"],
                outputs="encoded_diabetes_data",
                name="encode_categorical_features",
            ),
            Node(
                func=fit_scalers,
                inputs=[
                    "encoded_diabetes_data",
                    "params:columns",
                    "params:modelling_scalers",
                ],
                outputs="modelling_scalers",
                name="fit_scalers",
            ),
            Node(
                func=transform_scalers,
                inputs=["encoded_diabetes_data", "modelling_scalers"],
                outputs="master_table",
                name="scale_numerical_features",
            ),
        ]
    )
