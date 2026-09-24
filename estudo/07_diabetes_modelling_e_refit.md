# Bloco 07 — Diabetes: modelagem e refit

Arquivos para abrir junto:

- `diabetes/src/diabetes/pipelines/modelling/nodes.py` e `pipeline.py`
- `diabetes/src/diabetes/pipelines/refit/nodes.py` e `pipeline.py`
- `diabetes/src/diabetes/common.py` (a função `load_class`)

---

## 7.1 Os 4 nós da modelagem

| Nó | Função | Entrada → Saída |
|---|---|---|
| `train_baseline_model` | `train_model` | `master_table` → `baseline_model` |
| `evaluate_baseline_model` | `evaluate_model` | `baseline_model` + `master_table` → `baseline_metrics` |
| `optimize_hyperparameters` | `optimize_hyperparameters` | `master_table` → `optimized_model` |
| `evaluate_optimized_model` | `evaluate_model` | `optimized_model` + `master_table` → `optimized_metrics` |

A mesma função `evaluate_model` aparece duas vezes, com nomes de nó diferentes.

## 7.2 O modelo vem do YAML: `load_class`

```python
def load_class(class_path: str) -> type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
```

Com `class_path = "sklearn.ensemble.RandomForestClassifier"`:

1. `rsplit(".", 1)` separa no **último** ponto: `"sklearn.ensemble"` e `"RandomForestClassifier"`.
2. `importlib.import_module("sklearn.ensemble")` é o mesmo que `import sklearn.ensemble`, só que a partir de uma string.
3. `getattr(modulo, "RandomForestClassifier")` pega a classe de dentro do módulo.

Resultado: o código Python não sabe qual modelo vai treinar. Trocar para XGBoost é mudar uma linha de YAML (e instalar o pacote).

## 7.3 Nó `train_model` (o baseline)

```python
def train_model(master_table, columns, params):
    target = params["target_column"]                                   # "Outcome"
    feature_cols = columns_from_groups(columns, params["feature_groups"])   # 17 colunas
    train_df = master_table[master_table["split"].isin(params["train_splits"])]

    estimator_class = load_class(params["class_path"])
    estimator = estimator_class(**params.get("init_args", {}))         # LogisticRegression(random_state=46, max_iter=1000)
    estimator.fit(train_df[feature_cols], train_df[target])

    return {"estimator": estimator, "target_column": target,
            "feature_columns": feature_cols, "eval_splits": params.get("eval_splits", [])}
```

Dois pontos:

- **`**params.get("init_args", {})`**: o `**` "desempacota" o dicionário em argumentos nomeados. `{"random_state": 46, "max_iter": 1000}` vira `LogisticRegression(random_state=46, max_iter=1000)`.
- **Devolve um dicionário, não só o modelo.** Esse "artefato de modelo" carrega junto tudo o que a avaliação e a inferência precisam: quais colunas usar, qual é o alvo. Assim, `evaluate_model` e `predict` não precisam receber essa informação de novo.

**Por que LogisticRegression como baseline?** É simples, rápido e interpretável. Serve de régua: se o modelo sofisticado não for claramente melhor, não vale a complexidade.

## 7.4 Nó `optimize_hyperparameters`: o GridSearchCV

### Hiperparâmetro × parâmetro

- **Parâmetro**: o que o modelo aprende com os dados (os coeficientes da regressão, os cortes das árvores).
- **Hiperparâmetro**: o que **você** escolhe antes de treinar (quantas árvores, qual profundidade máxima).

### A grade

```yaml
param_grid:
  n_estimators: [100, 200, 300]          # quantas árvores na floresta
  max_depth: [null, 5, 10]               # profundidade máxima de cada árvore
  min_samples_split: [2, 5, 10]          # mínimo de linhas para dividir um nó da árvore
  min_samples_leaf: [1, 3]               # mínimo de linhas numa folha
  class_weight: [null, balanced]         # compensar o desbalanceamento?
```

3 × 3 × 3 × 2 × 2 = **108 combinações**.

### Validação cruzada (cv=5)

Para cada combinação, o GridSearchCV:

1. divide as 452 linhas de treino em 5 partes (*folds*);
2. treina em 4 partes e mede na 5ª;
3. repete 5 vezes, trocando a parte de medição;
4. tira a média das 5 notas.

```
fold:     1      2      3      4      5
rodada 1: [med] [trn] [trn] [trn] [trn]
rodada 2: [trn] [med] [trn] [trn] [trn]
...
rodada 5: [trn] [trn] [trn] [trn] [med]
```

108 combinações × 5 folds = **540 treinos**. É por isso que esse é o nó mais demorado (cerca de 1 minuto no seu computador). O `n_jobs=-1` usa todos os núcleos do processador em paralelo.

No fim, o GridSearchCV retreina a combinação vencedora com as 452 linhas inteiras (é o `best_estimator_`).

**Por que isso é melhor do que testar cada combinação no `test`?** Porque a escolha seria "sob medida" para aquelas 109 linhas. A validação cruzada escolhe usando só o treino, e o `test` e o `validate` continuam intocados.

### O vencedor

```
n_estimators=200, max_depth=10, min_samples_split=10,
min_samples_leaf=1, class_weight=balanced
roc_auc médio na validação cruzada: 0,8691
```

### `scoring: roc_auc`, e por que não acurácia

Com 65% de negativos, um modelo que responde "não tem diabetes" para todo mundo acerta 65%. A acurácia engana em base desbalanceada. A ROC AUC mede outra coisa (explicada abaixo) e não cai nessa armadilha.

### `class_weight: balanced`

Diz ao modelo: "errar um positivo custa mais do que errar um negativo", na proporção inversa da frequência. Em triagem médica faz sentido: mandar para exame uma pessoa saudável (falso alarme) custa menos do que mandar para casa uma pessoa doente (falso negativo). O grid testou com e sem, e escolheu `balanced`.

## 7.5 Nó `evaluate_model`: as métricas

Para cada split listado em `eval_splits`, calcula 5 métricas. A forma mais fácil de entender é pela **matriz de confusão**. Esta é a do RandomForest no `validate` (91 pacientes):

|  | Previu "não" | Previu "sim" |
|---|---:|---:|
| **Não tem diabetes (60)** | 52 ✅ verdadeiro negativo | 8 ❌ falso positivo |
| **Tem diabetes (31)** | 7 ❌ falso negativo | 24 ✅ verdadeiro positivo |

| Métrica | Fórmula | Conta | Valor | Em português |
|---|---|---|---:|---|
| **Accuracy** | acertos / total | (52+24) / 91 | 0,835 | de todas, quantas acertou |
| **Recall** | VP / (VP+FN) | 24 / 31 | 0,774 | das doentes, quantas encontrou |
| **Precision** | VP / (VP+FP) | 24 / 32 | 0,750 | de quem ele disse "doente", quantas eram |
| **F1** | média harmônica de precision e recall | | 0,762 | equilíbrio entre as duas |
| **ROC AUC** | | | 0,890 | ver abaixo |

**ROC AUC em uma frase:** se você sortear uma paciente doente e uma saudável, é a probabilidade de o modelo dar uma nota maior para a doente. 0,5 = moeda; 1,0 = perfeito. Ela usa as **probabilidades** (`predict_proba`), não o "sim/não", então não depende do limiar de 50%.

## 7.6 Os resultados completos

| Modelo | Split | Accuracy | Recall | Precision | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| LogisticRegression | train | 0,792 | 0,626 | 0,729 | 0,674 | 0,871 |
| LogisticRegression | test | 0,642 | 0,429 | 0,545 | 0,480 | 0,723 |
| LogisticRegression | validate | 0,824 | 0,710 | 0,759 | 0,733 | 0,888 |
| RandomForest | train | 0,923 | 0,974 | 0,830 | 0,896 | 0,991 |
| RandomForest | test | 0,624 | 0,595 | 0,510 | 0,550 | 0,727 |
| RandomForest | validate | **0,835** | **0,774** | 0,750 | **0,762** | **0,890** |

### Como ler essa tabela (e se preparar para perguntas)

**Train muito acima do resto no RandomForest (0,991 × 0,890):** é overfitting, o modelo "decorou" parte do treino. É normal em floresta aleatória com árvores profundas. O que importa é o desempenho fora do treino.

**O `test` está bem pior que o `validate` para os dois modelos:** são só 109 e 91 linhas. Com amostras pequenas, a variação entre um sorteio e outro é grande, e o `test` pegou pacientes mais difíceis. Como os **dois** modelos caem juntos no `test`, é efeito da amostra, não de um modelo específico.

**No `test`, o RandomForest tem acurácia menor que o baseline (0,624 × 0,642). Então ele é pior?** Não necessariamente. Ele tem recall bem maior (0,595 × 0,429) e AUC um pouco maior. O `class_weight=balanced` troca precisão por recall: acha mais doentes ao custo de mais falsos alarmes, e isso pode derrubar a acurácia. No `validate` ele ganha em todas as métricas.

**Quais features pesaram mais** (importância no RandomForest):

| Feature | Importância |
|---|---:|
| `NEW_GLUCOSE_INSULIN` | 0,171 |
| `Glucose` | 0,142 |
| `Age` | 0,132 |
| `BMI` | 0,104 |
| `DiabetesPedigreeFunction` | 0,066 |
| `NEW_GLUCOSE_CAT` | 0,058 |

A feature derivada `NEW_GLUCOSE_INSULIN` ficou em primeiro: a engenharia de features do notebook se pagou.

## 7.7 O refit

### Os 8 nós

| Nó | Função (reusada da data_engineering) | Saída |
|---|---|---|
| `refit_imputers` | `fit_imputers` | `production_imputers` |
| `refit_impute_missing_values` | `impute_missing_values` | `refit_imputed_data` |
| `refit_outlier_thresholds` | `fit_outlier_thresholds` | `production_outlier_thresholds` |
| `refit_clip_outliers` | `clip_outliers` | `refit_clipped_data` |
| `refit_add_engineered_features` | `add_engineered_features` | `refit_featured_data` |
| `refit_encoders` | `fit_encoders` | `production_encoders` |
| `refit_scalers` | `fit_scalers` | `production_scalers` |
| `refit_model` | `refit_model` (a única nova) | `production_model` |

Sete dos oito nós **importam funções da `data_engineering`**:

```python
from diabetes.pipelines.data_engineering.nodes import (
    add_engineered_features, clip_outliers, fit_encoders, fit_imputers,
    fit_outlier_thresholds, fit_scalers, impute_missing_values,
)
```

O que muda é o parâmetro: `params:refit_imputers` tem `split_to_fit: [train, test, validate]`.

Por que o refit reimputa, retrunca e recria as features em vez de só reajustar o modelo? Porque os limites de outlier de produção precisam ser calculados sobre os dados **já imputados com as medianas de produção**. Cada artefato depende do anterior, então a cadeia inteira é refeita.

Efeito prático: a mediana de `Insulin` passa de 119 (só treino) para 123,5 (todos os dados).

### O nó `refit_model`

```python
source_estimator = optimized_model["estimator"]
estimator = type(source_estimator)(**source_estimator.get_params())
estimator.fit(x_train, y_train)          # agora com as 652 linhas
```

- `type(source_estimator)` → a classe (`RandomForestClassifier`).
- `.get_params()` → os hiperparâmetros vencedores (`n_estimators=200, max_depth=10, ...`).
- Cria um modelo **novo e vazio** com esses hiperparâmetros e treina com tudo.

O modelo de produção não tem métrica própria (ele viu todos os dados). A estimativa de desempenho que vale é a do `validate` na modelagem: 0,835 de acurácia e 0,890 de AUC.

## Para praticar

```powershell
uv run kedro run --pipeline=modelling
uv run kedro ipython
```

```python
m = catalog.load("optimized_model")
m["best_params"], m["cv_best_score"]
catalog.load("optimized_metrics")["validate"]["roc_auc"]
```

Experimente mudar o `class_path` do otimizado para `sklearn.ensemble.GradientBoostingClassifier`, apagar o `param_grid` atual e colocar `n_estimators: [50, 100]` e `learning_rate: [0.05, 0.1]`. Rode `--pipeline=modelling` e compare.
