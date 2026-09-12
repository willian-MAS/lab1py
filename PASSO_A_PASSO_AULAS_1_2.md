# Aulas 1 e 2 — do notebook ao projeto Kedro (Telco Churn)

Reprodução passo a passo do que o professor fez nas duas primeiras aulas de
*Deployment: Production-Ready Data Science*, com base nos slides
`01_deployment_production_ready_data_science_vf.pdf` e
`02_kedro_concepts_vf.pdf`.

O resultado é o projeto [`churn/`](./churn) deste repositório: 4 pipelines,
18 nós, 19 datasets no catálogo e 9 blocos de parâmetros — o mesmo DAG que
aparece no `kedro viz` que o professor mostrou em aula (conferido nó a nó e
aresta a aresta contra o `code/churn-viz` do material).

> **Como usar este documento:** cada passo tem o comando exato para rodar no
> terminal do VSCode e o motivo pelo qual aquilo existe. Se você quiser só
> rodar o que já está pronto, pule para o passo 9.

---

## Mapa: o que cada slide pede

| Slide | Assunto | Passo aqui |
|---|---|---|
| 01, p. 7 e 18 | Preparando o ambiente (Python, uv, VSCode, Ruff, Git) | Passo 0 |
| 01, p. 19 | Quick start: `uv sync`, `uv run kedro run` | Passo 9 |
| 01, p. 20–21 | Por que notebook falha em produção / data leakage | Passo 5 (parâmetros `split_to_fit`) |
| 01, p. 25 | Scaffold do projeto (`kedro new --name churn`) | Passos 1 e 2 |
| 01, p. 26 | Planejamento das 4 pipelines | Passo 4 |
| 01, p. 27–28 | Pipeline de eng. de dados e padrão fit/transform | Passos 6 e 7 |
| 01, p. 29 | Wiring: nós, catálogo e parâmetros | Passos 5, 6 e 7 |
| 01, p. 30 | Modelagem: `class_path`, GridSearchCV | Passos 5 e 7 |
| 01, p. 31 | Refit e inferência (reuso de nós) | Passo 7 |
| 01, p. 36 | Ruff, pytest, logging | Passo 10 |
| 02, p. 3 | As 9 etapas do "production-ready" | estrutura geral deste guia |
| 02, p. 5 | Abordagem analítica em 3+1 pipelines | Passo 4 |
| 02, p. 6–15 | Kedro e os 12 fatores | seção "Por que cada pasta existe" |

Aulas 3 e 4 (FastAPI e Docker) entram depois, em cima deste mesmo projeto.

---

## Passo 0 — Preparar o ambiente (Part 0 dos slides)

### 0.1 Python e uv

```bash
python --version                              # 3.11+ serve; usamos 3.13
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
# Windows (PowerShell):
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

No Windows, os slides pedem também WSL2 (pré-requisito do Docker Desktop, que
só será usado na aula 4). Os tutoriais por sistema operacional estão em
`resources/tutorials/` do material da disciplina.

O `uv` substitui pip + virtualenv + poetry: ele cria o `.venv`, resolve as
dependências e grava o `uv.lock` com hash de cada pacote transitivo — é isso
que faz o ambiente ser reprodutível ("funciona na minha máquina" deixa de ser
argumento).

### 0.2 VSCode

Extensões (arquivo [`churn/.vscode/extensions.json`](./churn/.vscode/extensions.json)):
Python, Pylance, **Ruff**, YAML e Jupyter.

Configurações já versionadas em
[`churn/.vscode/settings.json`](./churn/.vscode/settings.json):

- interpretador apontando para `churn/.venv/bin/python` (no Windows, `.venv\Scripts\python.exe`);
- format on save e organize imports com Ruff;
- pytest habilitado apontando para `tests/`.

E [`churn/.vscode/launch.json`](./churn/.vscode/launch.json) para debugar com
F5 (`kedro run`, uma pipeline só, um nó só, ou `kedro viz`) — breakpoint dentro
de um nó em `nodes.py` funciona normalmente.

> **Importante:** abra a pasta `churn/` no VSCode (File → Open Folder), não a
> raiz do repositório. Assim o `${workspaceFolder}` das configurações acima
> aponta para o projeto Kedro.

### 0.3 Git

O `.gitignore` do projeto (gerado pelo `kedro new`) já ignora `conf/local/`,
`credentials.yml`, `data/` e `.venv/`. Neste repositório eu abri **uma**
exceção: `data/01_raw/*.csv` é versionado, porque a entrega do exercício pede
o repositório com os datasets.

---

## Passo 1 — Criar o projeto Kedro

```bash
cd /caminho/do/repo
uvx kedro new --name churn --tools=lint,test,log,data --example=n
```

`uvx` roda o Kedro sem instalar nada no sistema (ambiente temporário). As
`--tools` escolhidas são as que o professor usa: `lint` (Ruff), `test`
(pytest), `log` (`conf/logging.yml`) e `data` (as pastas `01_raw` a
`08_reporting`). `--example=n` porque não queremos o pipeline de exemplo
(spaceflights).

Estrutura criada:

```
churn/
├── conf/base/{catalog.yml,parameters.yml}   # configuração versionada
├── conf/local/                              # credenciais e overrides (git ignora)
├── conf/logging.yml
├── data/01_raw ... data/08_reporting         # camadas de maturidade do dado
├── notebooks/
├── src/churn/{__main__.py,settings.py,pipeline_registry.py}
├── tests/
└── pyproject.toml
```

### Por que cada pasta existe (aula 2: Kedro e os 12 fatores)

| Pasta | Fator / boa prática |
|---|---|
| `conf/base` vs `conf/local` | configuração no ambiente, não no código; `local` fica fora do git |
| `conf/base/catalog.yml` | serviços de apoio declarativos: o nó não sabe se é CSV, Parquet ou S3 |
| `data/01_raw … 08_reporting` | camadas de maturidade; `01_raw` é imutável; datasets intermediários são checkpoints |
| `src/churn/pipelines/` | modularidade: nós pequenos, puros e testáveis |
| `tests/` | robustez: pytest espelhando a estrutura de `src/` |
| `pyproject.toml` + `uv.lock` | dependências declaradas e isoladas, build/release/run separados |

---

## Passo 2 — Ambiente virtual e dependências (uv)

O `kedro new` cria o `pyproject.toml` com o mínimo. As dependências da nossa
solução entram com `uv add` (que já cria o `.venv`, resolve e grava o
`uv.lock`):

```bash
cd churn
echo "3.13" > .python-version      # fixa a versão do Python do projeto

uv add "kedro~=1.6.0" \
       "kedro-datasets[pandas-csvdataset,pickle-pickledataset,json-jsondataset]>=8.0" \
       "kedro-viz>=12.0" \
       "pandas>=2.2" "numpy>=1.26" "scikit-learn>=1.5" "catboost>=1.2" \
       "ipython>=8.10" "jupyterlab>=3.0" notebook

uv add --dev "pytest~=8.3" "pytest-cov>=5,<7" "pytest-mock>=3.14,<4.0" "ruff~=0.12.0"
```

Conferir:

```bash
uv run kedro info      # deve mostrar kedro 1.6.0 + plugin kedro_viz
```

Daí para frente **todo** comando roda com `uv run ...` (ou com o `.venv`
ativado). Em outra máquina, o ambiente inteiro volta com um `uv sync`.

O slide 7 da aula 2 cita `requirements.txt` **e** `pyproject.toml`: a fonte da
verdade é o `pyproject.toml` + `uv.lock`, e o `requirements.txt` é gerado a
partir dele, para quem precisar instalar com pip:

```bash
uv export --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt
```

---

## Passo 3 — Colocar os dados na camada 01_raw

```bash
cp <material>/data/telco-churn-modelling.csv churn/data/01_raw/
cp <material>/data/telco-churn-inference.csv churn/data/01_raw/
```

Os dois CSVs da disciplina usam `;` como separador e têm uma primeira coluna
de índice sem nome — isso vai aparecer no catálogo como `load_args: {sep: ";"}`.
`telco-churn-modelling.csv` tem 6.338 linhas (treino) e
`telco-churn-inference.csv` tem 705 (dados novos, sem a coluna `Churn`).

---

## Passo 4 — Planejar as pipelines e criá-las

A abordagem analítica do slide 5 (aula 2) e do slide 26 (aula 1) tem
**quatro** pipelines:

```
data_engineering : dado bruto  → limpeza → split → fit encoders/scalers → master_table
modelling        : master_table → baseline + otimização (GridSearchCV) → métricas
refit            : reajusta encoders/scalers/modelo em TODOS os splits → artefatos de produção
inference        : dado novo → mesmos transforms, com artefatos de produção → predições
```

```bash
uv run kedro pipeline create data_engineering
uv run kedro pipeline create modelling
uv run kedro pipeline create refit
uv run kedro pipeline create inference
```

Cada comando cria `src/churn/pipelines/<nome>/{__init__.py,nodes.py,pipeline.py}`
e `conf/base/parameters_<nome>.yml`. Não é preciso registrar nada à mão: o
`pipeline_registry.py` usa `find_pipelines()` e o `__default__` é a soma de
todas.

---

## Passo 5 — Preencher os parâmetros (conf/base/parameters*.yml)

Nenhuma lista de colunas e nenhum hiperparâmetro no código. Arquivos:

- [`conf/base/parameters.yml`](./churn/conf/base/parameters.yml) — `columns`
  (target, numéricas, categóricas), compartilhado por todas as pipelines;
- [`conf/base/parameters_data_engineering.yml`](./churn/conf/base/parameters_data_engineering.yml)
  — `split` (0.7/0.15/0.15, `random_state: 42`), `modelling_encoders`,
  `modelling_scalers`;
- [`conf/base/parameters_modelling.yml`](./churn/conf/base/parameters_modelling.yml)
  — `modelling_baseline` (LogisticRegression) e `modelling_optimization`
  (CatBoost + `param_grid`);
- [`conf/base/parameters_refit.yml`](./churn/conf/base/parameters_refit.yml)
  — `refit_encoders`, `refit_scalers`, `refit_model`.

O detalhe mais importante da aula inteira mora aqui:

```yaml
# conf/base/parameters_data_engineering.yml
modelling_encoders:
  columns: [target, categorical]
  split_to_fit: [train]          # ajusta SÓ no treino → sem data leakage

# conf/base/parameters_refit.yml
refit_encoders:
  columns: [target, categorical]
  split_to_fit: [train, test, validate]   # produção: usa tudo
```

Mesma função Python, dois comportamentos, zero `if` no código. É o que o slide
21 chama de `split → fit(train_only) → transform(all)`.

E no bloco de modelagem, o modelo é configuração:

```yaml
modelling_baseline:
  class_path: sklearn.linear_model.LogisticRegression
modelling_optimization:
  class_path: catboost.CatBoostClassifier
  param_grid:
    depth: [2, 4, 8]
    iterations: [10, 100, 300]
    learning_rate: [0.01, 0.05, 0.1]
```

Trocar de modelo (o exercício sugere XGBoost) é mexer só no YAML.

---

## Passo 6 — Preencher o catálogo (conf/base/catalog.yml)

[`conf/base/catalog.yml`](./churn/conf/base/catalog.yml) tem os 19 datasets,
cada um na sua camada. Exemplos:

```yaml
raw_churn_data:                       # 01_raw: imutável
  type: pandas.CSVDataset
  filepath: data/01_raw/telco-churn-modelling.csv
  load_args:
    sep: ";"

master_table:                         # 05_model_input: pronto para treino
  type: pandas.CSVDataset
  filepath: data/05_model_input/master_table_telco-churn-modelling.csv
  save_args:
    index: false

production_model:                     # 06_models: artefato serializado
  type: pickle.PickleDataset
  filepath: data/06_models/production_model.pkl

optimized_metrics:                    # 08_reporting: métricas
  type: json.JSONDataset
  filepath: data/08_reporting/optimized_metrics.json
```

Um dataset **não** está no catálogo de propósito: `raw_inference_dataframe`.
Sem entrada, o Kedro usa um `MemoryDataset` — e é exatamente esse o ponto onde
a API da aula 3 vai injetar o JSON recebido por HTTP para rodar a mesma
pipeline de inferência.

---

## Passo 7 — Escrever os nós (nodes.py)

Regra: função pura, sem I/O, sem caminho de arquivo, sem `print` (use
`logging`), e nunca mutar a entrada (`.copy()`).

| Arquivo | Funções |
|---|---|
| [`pipelines/data_engineering/nodes.py`](./churn/src/churn/pipelines/data_engineering/nodes.py) | `clean_data`, `add_split_column`, `fit_encoders`, `transform_encoders`, `fit_scalers`, `transform_scalers` |
| [`pipelines/modelling/nodes.py`](./churn/src/churn/pipelines/modelling/nodes.py) | `_load_class`, `train_model`, `optimize_hyperparameters`, `evaluate_model` |
| [`pipelines/refit/nodes.py`](./churn/src/churn/pipelines/refit/nodes.py) | `refit_model` |
| [`pipelines/inference/nodes.py`](./churn/src/churn/pipelines/inference/nodes.py) | `to_dataframe`, `predict` |

Três padrões que aparecem em todos eles:

1. **fit/transform separados** — `fit_encoders` filtra por
   `df["split"].isin(params["split_to_fit"])`; `transform_encoders` só aplica.
2. **artefato de modelo como dict** — `train_model` e
   `optimize_hyperparameters` devolvem
   `{"estimator", "target_column", "feature_columns", "eval_splits"}`, então o
   mesmo `evaluate_model` avalia baseline e otimizado.
3. **carregamento dinâmico de classe** — `_load_class("catboost.CatBoostClassifier")`
   via `importlib`, para o modelo vir do YAML.

O `add_split_column` também mostra o cuidado com robustez: levanta
`ValueError` se as proporções não somarem 1.0, e usa
`np.random.default_rng(random_state)` para o sorteio ser reprodutível.

---

## Passo 8 — Ligar os nós (pipeline.py)

```python
# src/churn/pipelines/data_engineering/pipeline.py
from kedro.pipeline import Node, Pipeline

from .nodes import clean_data, add_split_column  # ...

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
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
        # ...
    ])
```

O DAG não é declarado: ele é **inferido** dos nomes. `cleaned_churn_data` é
saída de um nó e entrada de outro, então o Kedro sabe a ordem. `params:` puxa
de `parameters*.yml` automaticamente.

O reuso fica explícito no `refit` e no `inference`, que importam funções da
`data_engineering`:

```python
# src/churn/pipelines/refit/pipeline.py
from churn.pipelines.data_engineering.nodes import fit_encoders, fit_scalers
```

```python
# src/churn/pipelines/inference/pipeline.py
from churn.pipelines.data_engineering.nodes import (
    clean_data, transform_encoders, transform_scalers,
)
```

Mesmas funções, `name=` diferente no nó (`refit_encoders`,
`clean_inference_data`, …) e outros parâmetros. Só duas funções realmente novas
na inferência: `to_dataframe` e `predict`.

Os quatro arquivos completos:
[data_engineering](./churn/src/churn/pipelines/data_engineering/pipeline.py) ·
[modelling](./churn/src/churn/pipelines/modelling/pipeline.py) ·
[refit](./churn/src/churn/pipelines/refit/pipeline.py) ·
[inference](./churn/src/churn/pipelines/inference/pipeline.py)

---

## Passo 9 — Rodar

```bash
cd churn
uv sync                        # se acabou de clonar o repo
uv run kedro run               # 18 nós, ~15 s nesta base
```

Por partes, enquanto desenvolve:

```bash
uv run kedro run --pipeline=data_engineering
uv run kedro run --pipeline=modelling
uv run kedro run --nodes=optimize_hyperparameters   # um nó só
uv run kedro run --to-nodes=fit_encoders            # até um nó
uv run kedro run --from-nodes=refit_model           # de um nó em diante
```

Isso só funciona porque os datasets intermediários estão persistidos no
catálogo (os *checkpoints* do slide 13 da aula 2).

Visualizar o DAG:

```bash
uv run kedro viz run           # http://127.0.0.1:4141
uv run kedro viz build         # versão estática (foi assim que o professor
                               # gerou a pasta code/churn-viz do material)
```

Explorar interativamente:

```bash
uv run kedro ipython
# >>> df = catalog.load("master_table")
# >>> df["split"].value_counts()
```

### O que você deve ver

```
Completed 18 out of 18 tasks
Pipeline execution completed successfully
```

E os arquivos:

```
data/02_intermediate/cleaned_telco-churn-modelling.csv
data/03_primary/split_telco-churn-modelling.csv          (train 4491 / test 922 / validate 925)
data/04_feature/encoded_telco-churn-modelling.csv
data/05_model_input/master_table_telco-churn-modelling.csv
data/06_models/{modelling,production}_{encoders,scalers}.pkl
data/06_models/{baseline,optimized,production}_model.pkl
data/07_model_output/inference_predictions.json          (705 predições)
data/08_reporting/{baseline,optimized}_metrics.json
```

Métricas da execução de referência (split de teste):

| Modelo | accuracy | roc_auc | f1_macro |
|---|---|---|---|
| baseline — LogisticRegression | 0,7744 | 0,8145 | 0,7079 |
| otimizado — CatBoost | 0,7831 | 0,8390 | 0,7137 |

Melhor combinação do grid search: `depth=4`, `iterations=100`,
`learning_rate=0.1` (roc_auc de CV = 0,8451).

---

## Passo 10 — Qualidade: Ruff, pytest e logging

```bash
uv run ruff check src/ tests/     # lint  (F, W, E, I, UP, PL, T201)
uv run ruff format src/ tests/    # formatação
uv run pytest                     # testes + cobertura
```

Testes escritos:

- [`tests/test_run.py`](./churn/tests/test_run.py) — o projeto carrega, as 5
  pipelines estão registradas e o `__default__` tem 18 nós (pega erro de nome
  de dataset/parâmetro antes de rodar tudo);
- [`tests/pipelines/data_engineering/test_nodes.py`](./churn/tests/pipelines/data_engineering/test_nodes.py)
  — nós testados com DataFrames sintéticos, incluindo um teste que prova que o
  encoder **não** aprende categorias que só existem no split de teste (o
  anti-data-leakage) e um que prova o determinismo do split.

Logging: cada `nodes.py` usa `logger = logging.getLogger(__name__)`, nunca
`print()` (a regra `T201` do Ruff reprova `print`). A configuração fica em
`conf/logging.yml`.

---

## Conferência com o projeto do professor

O material traz a build estática do `kedro viz` do professor
(`code/churn-viz/`). Comparando o JSON daquela build com a build deste
projeto, batem:

- as 5 pipelines (`__default__`, `data_engineering`, `modelling`, `refit`, `inference`);
- os 47 elementos do grafo (18 nós de tarefa, 20 datasets, 9 blocos de parâmetros);
- as 61 arestas do DAG;
- os valores de **todos** os parâmetros;
- o tipo e o nome de arquivo de **todos** os datasets do catálogo.

Duas diferenças que não afetam a execução: os `filepath` do professor são
absolutos da máquina dele (`/datascience/data/40_courses/...`) e aqui são
relativos ao projeto; e o nome do pacote/projeto é o mesmo (`churn`).

---

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `DatasetNotFoundError` | nome no `Node(...)` diferente do `catalog.yml` | conferir a grafia nos dois lados |
| `MissingConfigException` | chave ausente em `parameters*.yml` | conferir o nome depois de `params:` |
| `DatasetError` ao salvar pickle/json | biblioteca do dataset não instalada | `uv add "kedro-datasets[pickle-pickledataset]"` |
| `CatBoostError: Can't create train working dir` | pasta `data/tmp/` inexistente | `mkdir -p data/tmp` (o `.gitkeep` já vai no repo) |
| `ValueError: Split proportions must sum to 1.0` | proporções do `split` erradas | somar 1.0 em `parameters_data_engineering.yml` |
| VSCode não acha o `kedro` | interpretador errado | selecionar `churn/.venv/bin/python` |

---

## Próximos passos (aulas 3 e 4 e o exercício de avaliação)

- **Aula 3 — FastAPI:** expor `POST /inference` injetando os dados recebidos
  em `raw_inference_data`/`raw_inference_dataframe` como `MemoryDataset` e
  rodando a pipeline `inference` com `SequentialRunner`; `POST /train` roda
  `data_engineering + modelling + refit`.
- **Aula 4 — Docker:** `Dockerfile` com `uv sync --frozen` em duas camadas
  (deps e depois código) e `docker compose up --build`.
- **Exercício avaliado (trios, entrega 23/09):** repetir esta estrutura para o
  notebook `diabetes-prediction.ipynb` com os `diabetes-dataset-*.csv`. A
  estrutura de pastas, o `pipeline_registry`, o padrão fit/transform e os nós
  genéricos (`train_model`, `evaluate_model`, `_load_class`) são reaproveitáveis
  quase sem mudança: muda `columns` no `parameters.yml`, os `filepath` no
  `catalog.yml` e o `class_path` do modelo.
