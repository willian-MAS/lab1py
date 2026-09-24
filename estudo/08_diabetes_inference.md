# Bloco 08 — Diabetes: inferência

Arquivos para abrir junto: `diabetes/src/diabetes/pipelines/inference/nodes.py` e `pipeline.py`.

---

## 8.1 A ideia

A inferência é "aplicar o que foi aprendido a um dado novo". Ela **não aprende nada**: não tem nenhum nó `fit`.

## 8.2 Os 8 nós

| Nó | Função | Origem | Artefato usado |
|---|---|---|---|
| `to_dataframe` | `to_dataframe` | **nova** | — |
| `clean_inference_data` | `clean_data` | data_engineering | — |
| `impute_inference_data` | `impute_missing_values` | data_engineering | `production_imputers` |
| `clip_inference_data` | `clip_outliers` | data_engineering | `production_outlier_thresholds` |
| `engineer_inference_features` | `add_engineered_features` | data_engineering | — |
| `encode_inference_data` | `transform_encoders` | data_engineering | `production_encoders` |
| `scale_inference_data` | `transform_scalers` | data_engineering | `production_scalers` |
| `predict` | `predict` | **nova** | `production_model` |

Só 2 funções novas. As outras 6 são exatamente as da engenharia de dados. Esse reuso é a garantia de que a paciente nova passa pelo **mesmo** tratamento que as pacientes de treino. Se a limpeza fosse reescrita na inferência, qualquer diferença entre as duas versões viraria erro silencioso em produção (o nome técnico é *training-serving skew*).

## 8.3 `to_dataframe`

```python
def to_dataframe(raw_data):
    if isinstance(raw_data, pd.DataFrame):
        return raw_data
    return pd.DataFrame(raw_data)
```

Parece inútil, e é o que permite a mesma pipeline servir a dois mundos:

- **batch**: `raw_inference_data` vem do CSV → já é DataFrame → passa direto;
- **API**: a requisição HTTP chega como lista de dicionários → vira DataFrame.

A saída, `raw_inference_dataframe`, **não está no catálogo**: é um `MemoryDataset`.

## 8.4 `predict`

```python
def predict(model_artifact, inference_data):
    estimator = model_artifact["estimator"]
    feature_cols = model_artifact["feature_columns"]

    missing = [c for c in feature_cols if c not in inference_data.columns]
    if missing:
        raise ValueError(f"Faltam colunas exigidas pelo modelo ...: {', '.join(missing)}")

    x = inference_data[feature_cols]
    predictions = estimator.predict(x)                  # 0 ou 1
    probabilities = estimator.predict_proba(x)[:, 1]    # probabilidade de ser 1

    return [{"index": i, "prediction": int(p), "probability": float(pr)}
            for i, (p, pr) in enumerate(zip(predictions, probabilities))]
```

- **`feature_cols` vem do artefato**: o modelo "sabe" quais colunas usou no treino e em que ordem. Colunas extras na entrada (como o `Outcome` do CSV de inferência) são simplesmente ignoradas.
- **`predict_proba(x)[:, 1]`**: devolve uma matriz com duas colunas (probabilidade de 0 e de 1). `[:, 1]` pega a segunda: a chance de ter diabetes.
- **`prediction`** é 1 quando a probabilidade passa de 50%.
- **`int(...)` e `float(...)`**: convertem os tipos do numpy para tipos Python, que o JSON sabe gravar.

## 8.5 O resultado

`data/07_model_output/inference_predictions.json`, com 116 entradas:

```json
[{"index": 0, "prediction": 0, "probability": 0.3...},
 {"index": 1, "prediction": 0, "probability": 0.2...},
 ...]
```

Como o CSV de inferência traz o `Outcome` verdadeiro (e o modelo nunca o usa), dá para medir:

| Métrica | Valor |
|---|---:|
| Accuracy | 0,741 |
| Recall | 0,750 |
| Precision | 0,600 |
| F1 | 0,667 |
| ROC AUC | 0,822 |

Das 40 pacientes com diabetes, o modelo encontrou 30. Ele marcou 50 como positivas, e 30 delas eram.

## 8.6 A ordem das pipelines no `kedro run`

A inferência precisa dos artefatos de produção, que o refit gera, que precisa do modelo otimizado, que a modelagem gera, que precisa da master table. O Kedro descobre essa cadeia sozinho pelo DAG, e é por isso que o `kedro run` sem argumentos roda tudo na ordem certa.

Se os artefatos já existem, a inferência pode rodar sozinha:

```powershell
uv run kedro run --pipeline=inference
```
