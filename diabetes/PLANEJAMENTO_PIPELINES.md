# Planejamento dos pipelines Kedro

> Item 1 do exercicio: *"Planeje os pipelines kedro para esse notebook"*.
> Este documento mostra a leitura do notebook `diabetes-prediction.ipynb`, os
> problemas que impedem coloca-lo em producao como esta, e o desenho dos
> quatro pipelines que resolvem cada um deles.

## 1. O problema

Prever se uma paciente tem diabetes a partir de oito medidas de exame
(dataset Pima Indians, mulheres de 21 anos ou mais). Alvo: `Outcome`
(1 = positivo). Base de modelagem com 652 linhas; base de inferencia com 116.

## 2. O que o notebook faz

| Secao | O que acontece |
|---|---|
| 4 a 12 | Analise exploratoria, correlacoes, graficos |
| 13 | Baseline com 9 modelos, antes da engenharia de features |
| 15 | Faltantes: zeros impossiveis em `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI` viram nulos e sao preenchidos pela **mediana** |
| 16 | Outliers: limites por quantis (0.05 / 0.95) com 1,5 x IQR, valores fora sao truncados |
| 17 | Feature extraction: 8 colunas derivadas (`NEW_AGE_CAT`, `NEW_BMI`, `NEW_GLUCOSE`, `NEW_AGE_BMI_NOM`, `NEW_AGE_GLUCOSE_NOM`, `NEW_INSULIN_SCORE`, `NEW_GLUCOSE * INSULIN`, `NEW_GLUCOSE * PREGNANCIES`) |
| 18 | Encoding: label encoding nas binarias, one-hot nas demais |
| 19 | Padronizacao com `RobustScaler` |
| 20 | 9 modelos com `GridSearchCV` (cv=5) e comparacao de metricas |

## 3. Cinco problemas para colocar isso em producao

1. **Data leakage.** Mediana de imputacao, limites de outlier, encoders e
   scaler sao calculados sobre o dataset **inteiro**, e so depois vem o
   `train_test_split`. O modelo e avaliado com informacao que, na vida real,
   ele nao teria. E o problema que a disciplina ataca no slide 21 da aula 1.
2. **Nada e reaproveitavel na inferencia.** As transformacoes existem como
   celulas soltas, nao como artefatos salvos. Nao ha como aplicar em um
   paciente novo exatamente o mesmo tratamento do treino.
3. **Configuracao espalhada pelo codigo.** Pontos de corte clinicos (IMC 18,5 /
   24,9 / 29,9; glicemia 140 / 200) e hiperparametros estao dentro das celulas.
4. **Caminhos e ordem de execucao.** `pd.read_csv` fixo no codigo e resultado
   dependente da ordem em que as celulas foram rodadas.
5. **Sem interface.** Nao ha como outro sistema pedir uma predicao.

## 4. A abordagem analitica

Quatro pipelines, na linha do que a disciplina usou no caso de churn:

```
                    ┌──────────────────── DATA ENGINEERING ────────────────────┐
 diabetes-           clean_data ─→ add_split_column ─→ fit_imputers ─┐
 modelling.csv                                                       ├→ impute
                     ┌───────────────────────────────────────────────┘
                     └→ fit_outlier_thresholds ─→ clip ─→ add_engineered_features
                          ─→ fit_encoders ─→ encode ─→ fit_scalers ─→ scale
                                                                   ⇓
                                                             master_table
                    ┌──────── MODELLING ────────┐      ┌──────── REFIT ────────┐
                    │ baseline (LogisticReg.)   │      │ reajusta imputers,    │
                    │ otimizado (RandomForest   │      │ thresholds, encoders, │
                    │   + GridSearchCV cv=5)    │      │ scalers e o modelo    │
                    │ avaliacao dos dois        │      │ campeao em TODOS os   │
                    ⇓                           │      │ splits                │
              baseline_metrics                  │      ⇓                       │
              optimized_metrics                 │   production_* (5 artefatos) │
                    └───────────────────────────┘      └───────────────────────┘
                    ┌──────────────────────── INFERENCE ───────────────────────┐
 diabetes-           to_dataframe ─→ clean ─→ impute ─→ clip ─→ features
 inference.csv        ─→ encode ─→ scale ─→ predict ⇒ inference_predictions
 ou JSON da API       (mesmas funcoes, agora com os artefatos de producao)
                    └──────────────────────────────────────────────────────────┘
```

O criterio para separar as pipelines: **o que aprende** fica na
data_engineering e na modelling; **o que vai para producao** e refeito no
refit; **o que so aplica** fica na inference.

## 5. Como cada etapa do notebook virou no Kedro

| Notebook | No Kedro | Onde mora a configuracao |
|---|---|---|
| zeros impossiveis -> NaN | `clean_data` | `columns.zero_as_missing` |
| `train_test_split` | `add_split_column` | `split` (0.7 / 0.15 / 0.15, semente 42) |
| `df[col].median()` | `fit_imputers` + `impute_missing_values` | `modelling_imputers.split_to_fit` |
| `outlier_thresholds` / `replace_with_thresholds` | `fit_outlier_thresholds` + `clip_outliers` | `modelling_outliers.q_low/q_high/iqr_factor` |
| blocos `df.loc[...] = "mature"`, `pd.cut`, `Glucose * Insulin` | `add_engineered_features` | `feature_engineering` (binned / ranged / combined / products) |
| `label_encoder` / `one_hot_encoder` | `fit_encoders` + `transform_encoders` | `modelling_encoders` |
| `RobustScaler()` | `fit_scalers` + `transform_scalers` | `modelling_scalers.class_path` |
| `rf_model.fit(...)` | `train_model` | `modelling_baseline.class_path` |
| `GridSearchCV(rf_model, parameters, cv=5)` | `optimize_hyperparameters` | `modelling_optimization.param_grid` |
| `accuracy/recall/precision/f1/auc` | `evaluate_model` | `eval_splits` |
| (nao existe no notebook) | `refit_model` | `refit_model.train_splits` |
| (nao existe no notebook) | `predict` | artefatos de producao |

## 6. Decisoes de projeto

**Fit e transform separados.** Cada `fit_*` recebe `split_to_fit` e so olha
para as linhas daqueles splits. Na modelagem, `[train]`; no refit,
`[train, test, validate]`. Mesma funcao, comportamento diferente, zero `if`
no codigo. E isso que elimina o vazamento do notebook.

**Papel de cada split.** `train` ajusta, `test` compara os modelos,
`validate` so e tocado no fim, como estimativa honesta. Por isso o baseline e
o modelo otimizado treinam no mesmo split: comparacao justa.

**Imputacao e outliers viram artefato.** No notebook sao operacoes passageiras;
aqui a mediana de cada coluna e os limites de corte sao salvos em
`06_models/` e reaplicados identicamente na inferencia. Sem isso, um paciente
novo receberia um pre-processamento diferente do que o modelo aprendeu.

**Modelo definido por YAML.** `class_path: sklearn.ensemble.RandomForestClassifier`.
Trocar por XGBoost ou LightGBM (que o notebook tambem testa) e mudar uma linha
de configuracao, sem tocar em Python.

**Label encoding nas derivadas, e nao one-hot.** O modelo campeao e baseado em
arvores, que lidam bem com codigo inteiro, e assim o numero de colunas nao
explode a cada categoria nova. O padrao fit/transform fica igual ao da aula.

**Categoria desconhecida vira `-1`.** Em producao a API recebe qualquer coisa;
uma faixa nunca vista no treino nao pode derrubar o servico com excecao.

**`class_weight: balanced` entrou no grid.** A base e desbalanceada (65%
negativos) e, em triagem clinica, deixar passar uma paciente doente custa mais
caro do que um falso alarme. O grid escolhe, mas a opcao existe.

**Correcao de um bug do notebook.** Na celula que monta `NEW_AGE_BMI_NOM`, as
duas ultimas condicoes usam `df["BMI"] > 18.5`, o que sobrescreve todas as
faixas anteriores e marca quase todo mundo como "obese". Aqui a combinacao e
feita concatenando as faixas ja calculadas, entao o resultado sai correto.

## 7. Os artefatos de producao

O refit gera cinco arquivos em `data/06_models/`, e a inferencia depende
exatamente deles:

| Artefato | O que guarda |
|---|---|
| `production_imputers.pkl` | mediana de cada coluna |
| `production_outlier_thresholds.pkl` | limite inferior e superior de cada coluna |
| `production_encoders.pkl` | um `LabelEncoder` por categorica derivada |
| `production_scalers.pkl` | um `RobustScaler` por coluna numerica |
| `production_model.pkl` | estimador + coluna alvo + colunas de feature |

## 8. Por que isso permite a API e o container

`raw_inference_dataframe` nao esta no catalogo de proposito: sem entrada, o
Kedro usa um `MemoryDataset`. A API aproveita isso e substitui as entradas e
saidas da inferencia por datasets em memoria, rodando **a mesma pipeline** do
modo batch para escorar o JSON que chegou por HTTP. O container, por sua vez,
so precisa empacotar `conf/`, `src/` e os artefatos -- a logica nao muda entre
rodar no notebook do aluno, no servidor ou no Docker.
