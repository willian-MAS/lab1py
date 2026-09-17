"""Utilidades compartilhadas pelas pipelines.

Duas funcoes pequenas que aparecem em mais de uma pipeline: resolver grupos de
colunas declarados no YAML e carregar uma classe a partir do seu caminho.
"""

import importlib
from typing import Any


def as_list(value: Any) -> list:
    """Envolve um valor solto numa lista (listas passam intactas)."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def columns_from_groups(
    columns: dict[str, Any],
    groups: list[str],
) -> list[str]:
    """Resolve nomes de colunas a partir de grupos declarados no YAML.

    Args:
        columns: Dicionario de grupos de colunas (``params:columns``).
        groups: Nomes dos grupos a expandir, por exemplo
            ``["numerical", "engineered_categorical"]``.

    Returns:
        Lista achatada com os nomes das colunas, na ordem dos grupos.
    """
    return [item for group in groups for item in as_list(columns.get(group, []))]


def load_class(class_path: str) -> type:
    """Importa dinamicamente uma classe a partir do caminho completo.

    E o que permite trocar de modelo ou de scaler mexendo so no YAML.

    Args:
        class_path: Caminho completo, por exemplo
            ``"sklearn.ensemble.RandomForestClassifier"``.

    Returns:
        O objeto de classe importado.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
