# churn — pipeline de predição de churn (Telco) com Kedro

Projeto das aulas 1 e 2 de *Deployment: Production-Ready Data Science*
(Prof. Donald Neumann, Insper), construído sobre o dataset
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

O passo a passo completo de como este projeto foi construído (e como refazê-lo
do zero) está em [`../PASSO_A_PASSO_AULAS_1_2.md`](../PASSO_A_PASSO_AULAS_1_2.md).

## Quick start

```bash
cd churn
uv sync                     # cria .venv e instala tudo a partir do uv.lock
uv run kedro run            # roda as 4 pipelines (18 nós) de ponta a ponta
uv run kedro viz run        # abre o DAG no navegador (http://127.0.0.1:4141)
```

Outros comandos úteis:

```bash
uv run kedro run --pipeline=data_engineering   # só uma pipeline
uv run kedro run --nodes=clean_data            # só um nó
uv run kedro run --to-nodes=fit_encoders       # até um nó
uv run kedro run --from-nodes=refit_model      # de um nó em diante
uv run pytest                                  # testes
uv run ruff check src/ && uv run ruff format src/
```

## As quatro pipelines

| Pipeline | O que faz | Saídas principais |
|---|---|---|
| `data_engineering` | limpa, sorteia splits, ajusta encoders/scalers **só no treino** e gera a master table | `master_table`, `modelling_encoders`, `modelling_scalers` |
| `modelling` | treina baseline (LogisticRegression), otimiza CatBoost com GridSearchCV e avalia os dois | `baseline_metrics`, `optimized_metrics`, `optimized_model` |
| `refit` | reajusta encoders, scalers e o modelo vencedor usando **todos** os splits | `production_encoders`, `production_scalers`, `production_model` |
| `inference` | reaproveita os nós de limpeza/encode/scale e escora novos dados | `inference_predictions` |

```
data_engineering            modelling                     refit                      inference
raw_churn_data              master_table                  split_churn_data           raw_inference_data
  → clean_data                → train_baseline_model        → refit_encoders            → to_dataframe
  → add_split_column          → evaluate_baseline_model     → refit_scalers             → clean_inference_data
  → fit_encoders              → optimize_hyperparameters    → refit_model               → encode_inference_data
  → encode_categorical…       → evaluate_optimized_model                                → scale_inference_data
  → fit_scalers                                                                         → predict
  → scale_numerical…
  ⇒ master_table                                            ⇒ production_model         ⇒ inference_predictions
```

## Estrutura

```
churn/
├── conf/
│   ├── base/                  # versionado: catalog.yml + parameters*.yml
│   └── local/                 # ignorado pelo git: credenciais e overrides locais
├── data/                      # 01_raw ... 08_reporting (só 01_raw vai para o git)
├── src/churn/
│   ├── pipelines/
│   │   ├── data_engineering/  # nodes.py + pipeline.py
│   │   ├── modelling/
│   │   ├── refit/
│   │   └── inference/
│   ├── pipeline_registry.py   # descobre as pipelines automaticamente
│   └── settings.py
├── tests/                     # espelha src/
└── pyproject.toml             # dependências + config de ruff/pytest
```

## Resultados da última execução

| Modelo | split | accuracy | roc_auc | f1_macro |
|---|---|---|---|---|
| baseline (LogisticRegression) | test | 0,7744 | 0,8145 | 0,7079 |
| otimizado (CatBoost) | test | 0,7831 | 0,8390 | 0,7137 |

Melhores hiperparâmetros do grid search: `depth=4`, `iterations=100`,
`learning_rate=0.1` (roc_auc de CV = 0,8451).

## Próximas aulas

Aula 3 (FastAPI) e aula 4 (Docker) entram em cima deste mesmo projeto: expor
`inference` como endpoint HTTP e empacotar tudo em container. O ponto de
injeção já está preparado — `raw_inference_dataframe` é um `MemoryDataset`
(não está no catálogo), então a API pode substituir a entrada em memória e
rodar exatamente a mesma pipeline de inferência.
