# Prevendo a incidencia de Diabetes — pipelines Kedro, API FastAPI e Docker

Exercicio de avaliacao da disciplina *Deployment: Production-Ready Data Science*
(Insper — Prof. Donald Neumann). O notebook `diabetes-prediction.ipynb` foi
transformado num projeto [Kedro](https://kedro.org) de quatro pipelines,
exposto como API REST com FastAPI e empacotado em container Docker.

**Grupo:**

- Willian Miranda
- Nicole Cristine

O desenho dos pipelines e as decisoes de projeto estao em
[`PLANEJAMENTO_PIPELINES.md`](./PLANEJAMENTO_PIPELINES.md) (item 1 do enunciado).

---

## Sumario

1. [Como rodar](#como-rodar)
2. [Os quatro pipelines](#os-quatro-pipelines)
3. [Estrutura do projeto](#estrutura-do-projeto)
4. [Dados](#dados)
5. [Configuracao](#configuracao)
6. [Resultados](#resultados)
7. [API REST](#api-rest)
8. [Docker](#docker)
9. [Testes e qualidade](#testes-e-qualidade)

---

## Como rodar

Pre-requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
# 1. instalar as dependencias (cria o .venv a partir do uv.lock)
uv sync

# 2. rodar as quatro pipelines de ponta a ponta (~40 s)
uv run kedro run

# 3. visualizar o DAG
uv run kedro viz run          # http://127.0.0.1:4141
```

Rodando por partes, durante o desenvolvimento:

```bash
uv run kedro run --pipeline=data_engineering
uv run kedro run --pipeline=modelling
uv run kedro run --pipeline=refit
uv run kedro run --pipeline=inference
uv run kedro run --nodes=optimize_hyperparameters   # so um no
uv run kedro run --from-nodes=refit_model           # de um no em diante
```

Isso so funciona porque os datasets intermediarios ficam persistidos no
catalogo, servindo de checkpoint.

## Os quatro pipelines

| Pipeline | Nos | O que faz | Saidas principais |
|---|---:|---|---|
| `data_engineering` | 11 | limpa (zeros impossiveis viram nulos), sorteia os splits, ajusta **somente no treino** imputadores, limites de outlier, encoders e scalers, cria as features derivadas e monta a master table | `master_table`, `modelling_*` |
| `modelling` | 4 | treina o baseline (LogisticRegression), otimiza um RandomForest com `GridSearchCV` (cv=5) e avalia os dois | `baseline_metrics`, `optimized_metrics`, `optimized_model` |
| `refit` | 8 | refaz todos os artefatos e o modelo campeao usando **todos** os splits | `production_*` (5 artefatos) |
| `inference` | 8 | reaproveita os nos de limpeza, imputacao, outliers, features, encoding e escala com os artefatos de producao e escora dados novos | `inference_predictions` |

Total: **31 nos**, 33 datasets e 14 blocos de parametros.

```
raw → clean → split → [fit imputers] → impute → [fit outliers] → clip
    → features → [fit encoders] → encode → [fit scalers] → scale → master_table
                                                                      ↓
                                       baseline + otimizado → metricas
                                                                      ↓
                                    refit em todos os splits → artefatos de producao
                                                                      ↓
dado novo (CSV ou JSON da API) → mesmas transformacoes → predicoes
```

## Estrutura do projeto

```
diabetes/
├── conf/
│   ├── base/                  # versionado: catalog.yml + parameters*.yml
│   └── local/                 # ignorado pelo git: credenciais e overrides
├── data/                      # camadas 01_raw ... 08_reporting
│   └── 01_raw/                #   os dois CSV do exercicio (versionados)
├── notebooks/
│   └── diabetes-prediction.ipynb   # notebook de origem (sem outputs)
├── src/diabetes/
│   ├── api.py                 # aplicacao FastAPI
│   ├── common.py              # utilidades compartilhadas
│   ├── pipeline_registry.py   # descobre as pipelines automaticamente
│   └── pipelines/
│       ├── data_engineering/  # nodes.py + pipeline.py
│       ├── modelling/
│       ├── refit/
│       └── inference/
├── tests/                     # 34 testes (espelha src/)
├── Dockerfile
├── docker-compose.yml
├── PLANEJAMENTO_PIPELINES.md
└── pyproject.toml + uv.lock
```

## Dados

| Arquivo | Linhas | Uso |
|---|---:|---|
| `data/01_raw/diabetes-dataset-modelling.csv` | 652 | treino, teste e validacao |
| `data/01_raw/diabetes-dataset-inference.csv` | 116 | inferencia em lote |

Oito features (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`,
`Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`) e o alvo `Outcome`.

Detalhe que move o projeto inteiro: **zeros impossiveis**. Sao 314 zeros em
`Insulin` e 192 em `SkinThickness` — ninguem tem insulina zero. Eles sao
marcados como nulos na limpeza e imputados pela mediana **aprendida no split
de treino**.

## Configuracao

Nenhuma lista de colunas, ponto de corte clinico ou hiperparametro aparece no
codigo Python.

| Arquivo | Conteudo |
|---|---|
| `conf/base/catalog.yml` | onde cada dado mora e em que formato |
| `conf/base/parameters.yml` | grupos de colunas e as regras de feature engineering |
| `conf/base/parameters_data_engineering.yml` | split, imputacao, outliers, encoders, scalers |
| `conf/base/parameters_modelling.yml` | baseline, grid de hiperparametros |
| `conf/base/parameters_refit.yml` | os mesmos blocos, agora com todos os splits |

O contraste que resume a disciplina:

```yaml
# conf/base/parameters_data_engineering.yml   -> avaliacao honesta
modelling_imputers:
  split_to_fit: [train]

# conf/base/parameters_refit.yml              -> producao usa tudo
refit_imputers:
  split_to_fit: [train, test, validate]
```

## Resultados

Metricas da ultima execucao (`uv run kedro run`):

| Modelo | split | accuracy | recall | f1 | roc_auc |
|---|---|---:|---:|---:|---:|
| baseline — LogisticRegression | test | 0,6422 | 0,4286 | 0,4800 | 0,7196 |
| baseline — LogisticRegression | validate | 0,8132 | 0,7097 | 0,7213 | 0,8812 |
| otimizado — RandomForest | test | 0,6330 | 0,5714 | 0,5455 | 0,7349 |
| otimizado — RandomForest | validate | **0,8352** | **0,7742** | **0,7619** | **0,8898** |

Melhores hiperparametros do grid search (roc_auc de CV = 0,8722):
`n_estimators=300`, `max_depth=None`, `min_samples_split=2`,
`min_samples_leaf=3`, `class_weight=balanced`.

**Inferencia nos 116 registros da base separada** (que trazem o `Outcome` real,
entao da para conferir): accuracy 0,7500, recall 0,7750, f1 0,6813,
roc_auc 0,8151 — na mesma faixa do que o notebook obtem, agora sem vazamento e
de forma reprodutivel.

## API REST

```bash
uv run uvicorn diabetes.api:app --host 0.0.0.0 --port 8000
```

Documentacao interativa (Swagger) em <http://localhost:8000/docs>.

| Metodo | Rota | O que faz |
|---|---|---|
| GET | `/health` | status do servico e se ja existem artefatos de producao |
| GET | `/datasets` | lista os datasets do Data Catalog |
| GET | `/datasets/{nome}` | **serve um dataset como JSON** (`?limit=`, `?offset=`) |
| POST | `/train` | roda data_engineering + modelling + refit em segundo plano |
| GET | `/train/{run_id}` | acompanha o treino |
| POST | `/batch-inference` | roda a inferencia sobre o arquivo do catalogo |
| GET | `/batch-inference/{run_id}` | acompanha a inferencia em lote |
| POST | `/inference` | escora registros enviados no corpo (sincrono) |

### Expondo um dataset

```bash
curl "http://localhost:8000/datasets/master_table?limit=5"
curl "http://localhost:8000/datasets/optimized_metrics"
```

O cliente pede pelo nome logico e nao precisa saber se aquilo e CSV, JSON ou
Parquet — e a abstracao do catalogo chegando ate o HTTP. Datasets que nao
cabem em JSON (um modelo em pickle) respondem `415` com mensagem explicita.

### Inferencia online

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"Pregnancies":6,"Glucose":148,"BloodPressure":72,
       "SkinThickness":35,"Insulin":0,"BMI":33.6,
       "DiabetesPedigreeFunction":0.627,"Age":50}]}'
```

```json
{"predictions":[{"index":0,"prediction":1,"probability":0.8501}]}
```

A requisicao roda **a mesma pipeline `inference`** do modo batch: o catalogo
recebe `MemoryDataset` no lugar dos arquivos, nada toca o disco, e as
transformacoes aplicadas sao exatamente as do treino.

## Docker

```bash
docker compose up --build         # sobe a API em http://localhost:8000
```

Se os artefatos de producao ainda nao existirem (repo recem-clonado), treine
uma vez pela propria API e acompanhe:

```bash
curl -X POST http://localhost:8000/train
curl http://localhost:8000/train/<run_id>
```

Como `./data` esta montado como volume, os modelos gerados dentro do container
ficam na sua maquina. Alternativamente, rode `uv run kedro run` antes de subir:
o volume ja leva os artefatos prontos.

O `Dockerfile` instala em duas camadas (dependencias e depois codigo), o que
preserva o cache do `uv sync` quando so o codigo muda, e declara um
`HEALTHCHECK` que usa o proprio `/health`.

## Testes e qualidade

```bash
uv run pytest                      # 34 testes
uv run ruff check src/ tests/      # lint
uv run ruff format src/ tests/     # formatacao
```

A suite cobre os nos (com DataFrames sinteticos) e a API (com `TestClient`).
Os testes mais importantes sao os que protegem o ponto central da disciplina:

- `test_mediana_vem_apenas_do_split_de_treino` — imputacao nao enxerga o teste;
- `test_encoder_nao_aprende_categoria_exclusiva_do_teste` — sem vazamento de rotulo;
- `test_categoria_desconhecida_nao_derruba_a_inferencia` — robustez em producao;
- `test_inferencia_so_consome_artefatos_de_producao` — a pipeline de inferencia
  nunca depende de artefato de modelagem.

---

## Checklist do enunciado

| Item da avaliacao | Onde esta |
|---|---|
| 1. Planejar os pipelines | [`PLANEJAMENTO_PIPELINES.md`](./PLANEJAMENTO_PIPELINES.md) |
| 2. Pipeline de engenharia de dados | `src/diabetes/pipelines/data_engineering/` (11 nos) |
| 2. Pipeline de treinamento | `src/diabetes/pipelines/modelling/` (4 nos) + `refit/` (8 nos) |
| 2. Pipeline de inferencia | `src/diabetes/pipelines/inference/` (8 nos) |
| Repo com os datasets | `data/01_raw/*.csv` versionados (excecao explicita no `.gitignore`) |
| Instalacao com uv | `pyproject.toml` + `uv.lock`; `uv sync` |
| `kedro run` e `kedro viz` | 31 nos executam de ponta a ponta; `uv run kedro viz run` |
| 3. [Extra] Dataset exposto como API | `GET /datasets` e `GET /datasets/{nome}` em `src/diabetes/api.py` |
| 4. [Extra] Container para inferencia via FastAPI | `Dockerfile` + `docker-compose.yml` |
