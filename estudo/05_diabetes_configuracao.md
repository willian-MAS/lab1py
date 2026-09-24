# Bloco 05 — Diabetes: a configuração

Arquivos para abrir junto: tudo em `diabetes/conf/base/`.

A regra do Kedro: **lógica em `src/`, configuração em `conf/`**. Neste projeto, nenhuma lista de colunas, ponto de corte ou hiperparâmetro aparece no Python.

---

## 5.1 `catalog.yml` — onde cada dado mora

Cada entrada tem um nome, um tipo e um caminho:

```yaml
raw_diabetes_data:
  type: pandas.CSVDataset
  filepath: data/01_raw/diabetes-dataset-modelling.csv

cleaned_diabetes_data:
  type: pandas.CSVDataset
  filepath: data/02_intermediate/cleaned_diabetes-modelling.csv
  save_args:
    index: false
```

- `type`: a classe que sabe ler e gravar aquele formato.
- `filepath`: relativo à raiz do projeto. Funciona em Windows, Linux e dentro do Docker.
- `save_args: {index: false}`: repassado ao `df.to_csv(...)`. Sem isso, o pandas grava o índice como uma coluna extra sem nome.

São 32 entradas, organizadas por pipeline:

| Tipo | Quantos | Exemplos |
|---|---:|---|
| `pandas.CSVDataset` | 18 | tabelas de cada etapa |
| `pickle.PickleDataset` | 11 | imputadores, limites, encoders, scalers, modelos |
| `json.JSONDataset` | 3 | métricas e predições |

**Um dataset está de fora de propósito:** `raw_inference_dataframe`. Sem entrada no catálogo, o Kedro usa um `MemoryDataset`. É por essa porta que a API injeta os dados que chegam por HTTP (bloco 09).

## 5.2 `parameters.yml` — compartilhado por todas as pipelines

### `columns`: os grupos de colunas

```yaml
columns:
  target: Outcome
  numerical: [Pregnancies, Glucose, BloodPressure, SkinThickness,
              Insulin, BMI, DiabetesPedigreeFunction, Age]
  categorical: []                   # o bruto não tem categóricas
  zero_as_missing: [Glucose, BloodPressure, SkinThickness, Insulin, BMI]
  engineered_numerical: [NEW_GLUCOSE_INSULIN, NEW_GLUCOSE_PREGNANCIES]
  engineered_categorical: [NEW_AGE_CAT, NEW_BMI_CAT, NEW_GLUCOSE_CAT,
                           NEW_GLUCOSE_LEVEL, NEW_AGE_BMI_CAT,
                           NEW_AGE_GLUCOSE_CAT, NEW_INSULIN_SCORE]
```

Os nós nunca escrevem nome de coluna. Eles recebem **nomes de grupos** (`numerical`, `engineered_categorical`) e a função `columns_from_groups` (em `src/diabetes/common.py`) traduz o grupo para a lista de colunas.

Repare que `Pregnancies` está em `numerical` mas **não** em `zero_as_missing`: zero gestações é um valor legítimo.

### `feature_engineering`: as regras clínicas

Quatro famílias de regra:

```yaml
feature_engineering:
  binned:                     # faixas de uma coluna (pd.cut)
    NEW_AGE_CAT:
      source: Age
      bins: [0, 50, 200]
      labels: [mature, senior]
      right: false
    NEW_BMI_CAT:
      source: BMI
      bins: [0, 18.5, 24.9, 29.9, 200]
      labels: [Underweight, Healthy, Overweight, Obese]
    NEW_GLUCOSE_CAT:
      source: Glucose
      bins: [0, 140, 200, 500]
      labels: [Normal, Prediabetes, Diabetes]
    NEW_GLUCOSE_LEVEL:
      source: Glucose
      bins: [0, 70, 100, 126, 500]
      labels: [low, normal, hidden, high]
      right: false
  ranged:                     # dentro ou fora de uma faixa de referência
    NEW_INSULIN_SCORE: {source: Insulin, low: 16, high: 166,
                        inside: Normal, outside: Abnormal}
  combined:                   # junta duas categóricas já criadas
    NEW_AGE_BMI_CAT: [NEW_BMI_CAT, NEW_AGE_CAT]
    NEW_AGE_GLUCOSE_CAT: [NEW_GLUCOSE_LEVEL, NEW_AGE_CAT]
  products:                   # multiplica duas numéricas
    NEW_GLUCOSE_INSULIN: [Glucose, Insulin]
    NEW_GLUCOSE_PREGNANCIES: [Glucose, Pregnancies]
```

#### O que é `right: false`

O `pd.cut` cria intervalos. Por padrão, eles são fechados à **direita**: com `bins: [0, 50, 200]`, os intervalos são `(0, 50]` e `(50, 200]`. Então 50 anos cairia em "mature".

O notebook diz `Age >= 50` → senior. Com `right: false`, os intervalos viram `[0, 50)` e `[50, 200)`, e 50 cai em "senior". O mesmo raciocínio vale para as faixas de glicose (`bins: [0, 70, 100, 126, 500]` com `right: false` reproduz "<70 low, 70–99 normal, 100–125 hidden, >125 high", já que a glicose é inteira).

## 5.3 `parameters_data_engineering.yml`

```yaml
split:
  train: 0.7
  test: 0.15
  validate: 0.15
  random_state: 42

modelling_imputers:
  columns: [numerical]
  strategy: median
  split_to_fit: [train]

modelling_outliers:
  columns: [numerical]
  q_low: 0.05
  q_high: 0.95
  iqr_factor: 1.5
  split_to_fit: [train]

modelling_encoders:
  columns: [engineered_categorical]
  split_to_fit: [train]

modelling_scalers:
  columns: [numerical, engineered_numerical]
  class_path: sklearn.preprocessing.RobustScaler
  split_to_fit: [train]
```

O padrão se repete em todos os blocos:

- `columns`: **quais grupos** tratar;
- `split_to_fit`: **em quais linhas aprender**. Aqui, sempre `[train]`.

O `class_path` do scaler é carregado dinamicamente. Para trocar de `RobustScaler` para `StandardScaler`, basta mudar essa linha.

## 5.4 `parameters_modelling.yml`

```yaml
_feature_groups: &feature_groups
  - numerical
  - engineered_numerical
  - engineered_categorical

modelling_baseline:
  class_path: sklearn.linear_model.LogisticRegression
  target_column: Outcome
  feature_groups: *feature_groups
  init_args: {random_state: 46, max_iter: 1000}
  train_splits: [train]
  eval_splits: [train, test, validate]

modelling_optimization:
  class_path: sklearn.ensemble.RandomForestClassifier
  target_column: Outcome
  feature_groups: *feature_groups
  init_args: {random_state: 46}
  cv: 5
  scoring: roc_auc
  param_grid:
    n_estimators: [100, 200, 300]
    max_depth: [null, 5, 10]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 3]
    class_weight: [null, balanced]
  train_splits: [train]
  eval_splits: [train, test, validate]
```

Dois recursos de YAML aparecem aqui:

- **`&feature_groups` e `*feature_groups`**: o `&` dá um nome a um trecho, e o `*` reusa o trecho. É um "copiar e colar" do próprio YAML, para não repetir a lista.
- **Chave começando com `_`**: o Kedro trata como auxiliar de template e não cria um parâmetro `params:_feature_groups`.

`null` em YAML vira `None` no Python. `max_depth: null` = árvores sem limite de profundidade.

## 5.5 `parameters_refit.yml`

**Os mesmos blocos**, com uma única diferença que importa:

```yaml
refit_imputers:
  columns: [numerical]
  strategy: median
  split_to_fit: [train, test, validate]    ← aqui está a diferença
```

E o `refit_model`, que só diz em quais splits retreinar.

Esse contraste é o ponto central do projeto:

| Arquivo | `split_to_fit` | Para quê |
|---|---|---|
| `parameters_data_engineering.yml` | `[train]` | avaliação honesta |
| `parameters_refit.yml` | `[train, test, validate]` | produção, com o máximo de dados |

**A mesma função Python** (`fit_imputers`), dois comportamentos, zero `if` no código.

## 5.6 `parameters_inference.yml`

Vazio. A inferência não aprende nada. Ela usa `columns`, `feature_engineering` e os artefatos de produção.

## 5.7 `conf/local/` e `conf/logging.yml`

- `conf/local/`: sobrescreve `base` na sua máquina. Fica fora do git. É onde iriam senhas (`credentials.yml`), se houvesse um banco de dados.
- `conf/logging.yml`: define como os logs aparecem (cores no terminal e um arquivo `info.log` rotativo). É por isso que o `kedro run` mostra aquelas linhas `INFO` coloridas.
