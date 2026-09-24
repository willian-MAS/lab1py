# Bloco 06 — Diabetes: engenharia de dados, nó por nó

Arquivos para abrir junto:

- `diabetes/src/diabetes/pipelines/data_engineering/nodes.py` (as funções)
- `diabetes/src/diabetes/pipelines/data_engineering/pipeline.py` (o encaixe)

Para não ficar abstrato, vamos seguir **uma paciente real**: a primeira linha do `diabetes-dataset-modelling.csv`. Ela tem diabetes (`Outcome = 1`) e foi sorteada para o split de `test`.

---

## A paciente no começo

```
Pregnancies=3  Glucose=162  BloodPressure=52  SkinThickness=38
Insulin=0      BMI=37.2     DiabetesPedigreeFunction=0.652  Age=24   Outcome=1
```

Repare no `Insulin=0`. Guarde isso.

## Visão geral dos 11 nós

| # | Nó (`name=`) | Função | Entrada → Saída |
|---|---|---|---|
| 1 | `clean_data` | `clean_data` | `raw_diabetes_data` → `cleaned_diabetes_data` |
| 2 | `add_split_column` | `add_split_column` | `cleaned` → `split_diabetes_data` |
| 3 | `fit_imputers` | `fit_imputers` | `split` → `modelling_imputers` 🧠 |
| 4 | `impute_missing_values` | `impute_missing_values` | `split` + imputers → `imputed_diabetes_data` |
| 5 | `fit_outlier_thresholds` | `fit_outlier_thresholds` | `imputed` → `modelling_outlier_thresholds` 🧠 |
| 6 | `clip_outliers` | `clip_outliers` | `imputed` + limites → `clipped_diabetes_data` |
| 7 | `add_engineered_features` | `add_engineered_features` | `clipped` → `featured_diabetes_data` |
| 8 | `fit_encoders` | `fit_encoders` | `featured` → `modelling_encoders` 🧠 |
| 9 | `encode_categorical_features` | `transform_encoders` | `featured` + encoders → `encoded_diabetes_data` |
| 10 | `fit_scalers` | `fit_scalers` | `encoded` → `modelling_scalers` 🧠 |
| 11 | `scale_numerical_features` | `transform_scalers` | `encoded` + scalers → `master_table` |

🧠 = nó que **aprende** algo (um "fit"). Todos os quatro aprendem **só com o split de treino**.

Repare que o nome do nó (`name=`) nem sempre é o nome da função: o nó 9 se chama `encode_categorical_features`, mas chama a função `transform_encoders`. A mesma função é reusada em outros nós com outros nomes (na inferência ela vira `encode_inference_data`).

---

## Nó 1 — `clean_data`

```python
def clean_data(raw_diabetes_data, columns):
    df_raw = raw_diabetes_data.copy()

    # 1. seleciona só as colunas declaradas nos grupos target/numerical/categorical
    selected = [col for group in ("target", "numerical", "categorical")
                for col in as_list(columns.get(group))
                if col in df_raw.columns]
    df_flt = df_raw[selected].copy()

    # 2. garante tipo numérico (texto estranho vira NaN)
    for col in as_list(columns.get("numerical")):
        if col in df_flt.columns:
            df_flt[col] = pd.to_numeric(df_flt[col], errors="coerce")

    # 3. zeros impossíveis viram NaN
    for col in as_list(columns.get("zero_as_missing")):
        if col in df_flt.columns:
            df_flt[col] = df_flt[col].replace(0, np.nan)

    return df_flt
```

Três detalhes:

- **`.copy()`**: a função nunca altera o DataFrame que recebeu. Isso é o que torna o nó "puro".
- **`if col in df_raw.columns`**: se uma coluna não existe, é pulada em silêncio. Isso permite usar a **mesma** função na inferência, onde o `Outcome` pode não vir.
- **`errors="coerce"`**: se chegar um texto como `"abc"` numa coluna numérica, vira `NaN` em vez de quebrar.

**A paciente depois:** `Insulin: 0 → NaN`. O resto igual.

Na base inteira, 546 valores viram `NaN` (5 + 27 + 192 + 314 + 8).

---

## Nó 2 — `add_split_column`

```python
def add_split_column(df_in, split):
    total = split["train"] + split["test"] + split["validate"]
    if not np.isclose(total, 1.0):
        raise ValueError(f"As proporcoes do split devem somar 1.0, e somam {total}")

    rng = np.random.default_rng(split["random_state"])
    labels = rng.choice(a=["train", "test", "validate"], size=len(df_in),
                        p=[split["train"], split["test"], split["validate"]])
    return df_in.assign(split=labels)
```

- Cada linha "tira na sorte" o seu grupo, com probabilidades 70/15/15.
- **`default_rng(42)`**: gerador de números aleatórios com semente fixa. Toda execução sorteia exatamente os mesmos grupos. É o determinismo que o notebook não tinha.
- **`np.isclose`** em vez de `==`: `0.7 + 0.15 + 0.15` em ponto flutuante dá `0.9999999999999999`, não `1.0`.
- A validação com `raise ValueError` protege contra erro de digitação no YAML.

Resultado: **452 train, 109 test, 91 validate**. Como é sorteio, não dá exatamente 70/15/15, e tudo bem.

**A paciente:** `split = "test"`.

---

## Nós 3 e 4 — imputação (o primeiro fit/transform)

### Nó 3 — `fit_imputers` (aprende)

```python
def fit_imputers(df_in, columns, params):
    mask = df_in["split"].isin(params["split_to_fit"])     # só linhas de treino
    imputers = {}
    for col in columns_from_groups(columns, params["columns"]):
        serie = df_in.loc[mask, col]
        imputers[col] = float(serie.median())              # ou mean(), conforme o YAML
    return imputers
```

A linha que importa é a do `mask`: `df_in["split"].isin(["train"])` cria uma coluna de verdadeiro/falso, e `df_in.loc[mask, col]` pega só as linhas de treino. A mediana nunca "vê" as linhas de teste e validação.

O resultado é um dicionário simples, salvo em `data/06_models/modelling_imputers.pkl`:

| Coluna | Mediana (treino) |
|---|---:|
| Pregnancies | 3 |
| Glucose | 117 |
| BloodPressure | 72 |
| SkinThickness | 28 |
| Insulin | 119 |
| BMI | 31,95 |
| DiabetesPedigreeFunction | 0,351 |
| Age | 29 |

A mediana ignora os `NaN` automaticamente. Então a mediana de `Insulin` é calculada só com quem **fez** o exame.

**Por que mediana e não média?** Entre quem fez o exame, a insulina vai até 846, mas a maioria fica entre 100 e 200. A média (155) é puxada para cima por esses extremos; a mediana (123,5 na base toda, 119 no treino) não.

### Nó 4 — `impute_missing_values` (aplica)

```python
def impute_missing_values(df_in, imputers):
    df_out = df_in.copy()
    for col, value in imputers.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(value)
    return df_out
```

Não aprende nada, só preenche. Aplica em **todas** as linhas (treino, teste e validação), usando o valor aprendido no treino.

**A paciente:** `Insulin: NaN → 119`.

### O padrão fit/transform, em uma frase

> O **fit** aprende um número (ou um objeto) olhando só para o treino. O **transform** aplica esse número a qualquer dado.

É esse padrão que se repete nos nós 5–6, 8–9 e 10–11. Entendeu um, entendeu os quatro.

---

## Nós 5 e 6 — outliers

### Nó 5 — `fit_outlier_thresholds` (aprende)

```python
quartile_low  = serie.quantile(0.05)
quartile_high = serie.quantile(0.95)
iqr = quartile_high - quartile_low
thresholds[col] = {"low": quartile_low - 1.5 * iqr,
                   "up":  quartile_high + 1.5 * iqr}
```

É a regra do notebook (`outlier_thresholds`): o intervalo é medido entre os quantis 5% e 95%, e o limite fica a 1,5 intervalo de distância.

### Nó 6 — `clip_outliers` (aplica)

```python
df_out[col] = df_out[col].clip(lower=limits["low"], upper=limits["up"])
```

`clip` "achata" quem passa do limite: um valor acima de `up` vira `up`.

### Uma observação crítica (boa para mostrar que você entendeu)

A regra clássica de outlier usa os quartis **25% e 75%**. O notebook usa **5% e 95%**, que já são extremos, e ainda soma 1,5 × esse intervalo. Os limites ficam enormes:

| Coluna | Limite inferior | Limite superior |
|---|---:|---:|
| Glucose | −70,7 | 331,1 |
| Insulin | −321,7 | 666,9 |
| BMI | −10,7 | 76,9 |

Na prática, só **3 valores** em toda a base de modelagem foram truncados. Mantivemos a regra para ser fiel ao notebook, mas, se o professor perguntar, a resposta honesta é: "com esses quantis, o tratamento de outlier quase não age; com 25/75 ele seria bem mais agressivo, e isso é uma mudança de uma linha no YAML".

**A paciente:** nenhum valor fora dos limites. Nada muda.

---

## Nó 7 — `add_engineered_features`

Cria 9 colunas novas, lendo as regras do YAML (bloco 05).

```python
# binned: faixas
df_out[name] = pd.cut(df_out[spec["source"]], bins=spec["bins"],
                      labels=spec["labels"], right=spec.get("right", True))

# ranged: dentro ou fora da faixa
inside = df_out[source].between(spec["low"], spec["high"])
df_out[name] = np.where(inside, spec["inside"], spec["outside"])

# combined: concatena duas categóricas com "_"
df_out[name] = df_out[a].astype(str) + "_" + df_out[b].astype(str)

# products: multiplica duas numéricas
df_out[name] = df_out[a] * df_out[b]
```

**A paciente** (Glucose 162, BMI 37,2, Age 24, Insulin 119 já imputada, Pregnancies 3):

| Feature | Regra | Valor |
|---|---|---|
| `NEW_AGE_CAT` | 24 está em [0, 50) | `mature` |
| `NEW_BMI_CAT` | 37,2 > 29,9 | `Obese` |
| `NEW_GLUCOSE_CAT` | 140 < 162 ≤ 200 | `Prediabetes` |
| `NEW_GLUCOSE_LEVEL` | 162 ≥ 126 | `high` |
| `NEW_INSULIN_SCORE` | 16 ≤ 119 ≤ 166 | `Normal` |
| `NEW_AGE_BMI_CAT` | `Obese` + `mature` | `Obese_mature` |
| `NEW_AGE_GLUCOSE_CAT` | `high` + `mature` | `high_mature` |
| `NEW_GLUCOSE_INSULIN` | 162 × 119 | `19278` |
| `NEW_GLUCOSE_PREGNANCIES` | 162 × 3 | `486` |

### Três pontos para discutir

**1. A ordem importa.** As features vêm **depois** da imputação. Se viessem antes, `NEW_GLUCOSE_INSULIN` seria `162 × 0 = 0` para quase metade da base.

**2. Efeito colateral da imputação.** 314 pacientes tinham insulina zero, receberam 119, e 119 está dentro de [16, 166]. Então todas ganham `NEW_INSULIN_SCORE = Normal`. A feature fica "contaminada" pela imputação. Não é um erro, mas é uma limitação que vale saber explicar.

**3. Um bug do notebook corrigido.** No notebook, a célula que cria `NEW_AGE_BMI_NOM` termina assim:

```python
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"
```

Deveria ser `BMI >= 30`. Como essas linhas vêm por último, elas **sobrescrevem** as anteriores, e quase todo mundo vira "obese". Aqui a combinação é feita concatenando as faixas já calculadas (`NEW_BMI_CAT + NEW_AGE_CAT`), então o resultado sai certo.

---

## Nós 8 e 9 — encoding

Modelos de machine learning só entendem números. `"Obese"` precisa virar um número.

### Nó 8 — `fit_encoders`

Para cada uma das 7 categóricas, um `LabelEncoder` aprende a lista de categorias existentes **no treino**, em ordem alfabética:

| Coluna | Categorias aprendidas → código |
|---|---|
| `NEW_AGE_CAT` | mature → 0, senior → 1 |
| `NEW_BMI_CAT` | Healthy → 0, Obese → 1, Overweight → 2, Underweight → 3 |
| `NEW_GLUCOSE_CAT` | Normal → 0, Prediabetes → 1 |
| `NEW_INSULIN_SCORE` | Abnormal → 0, Normal → 1 |

Repare: `NEW_GLUCOSE_CAT` não tem "Diabetes". A maior glicose da base é 199, então ninguém cai na faixa acima de 200.

### Nó 9 — `transform_encoders`

```python
mapping = {label: code for code, label in enumerate(encoder.classes_)}
codes = df_out[col].astype(str).map(mapping)
df_out[col] = codes.fillna(UNKNOWN_CATEGORY_CODE).astype(int)    # -1
```

Em vez de chamar `encoder.transform(...)`, que **quebra** diante de uma categoria nunca vista, montamos um dicionário e usamos `.map()`. O que não está no dicionário vira `-1`.

Por que isso importa: em produção, a API pode receber uma paciente com glicose 250. Ela vira `NEW_GLUCOSE_CAT = "Diabetes"`, categoria que o encoder nunca viu. Com `transform` puro, a API cairia; assim, vira `-1` e segue.

**A paciente:** `NEW_AGE_CAT: mature → 0`, `NEW_BMI_CAT: Obese → 1`, `NEW_INSULIN_SCORE: Normal → 1`, ...

**Por que label encoding e não one-hot?** O modelo campeão é baseado em árvores, que lidam bem com códigos inteiros. One-hot criaria uma coluna por categoria (só `NEW_AGE_GLUCOSE_CAT` tem 8), e o número de colunas explodiria sem ganho para árvores.

---

## Nós 10 e 11 — escalonamento

Colunas em escalas diferentes (glicose na casa de 100; `DiabetesPedigreeFunction` na casa de 0,5) atrapalham modelos como a regressão logística.

### Nó 10 — `fit_scalers`

Um `RobustScaler` por coluna numérica (as 8 originais + os 2 produtos). O `RobustScaler` aprende:

- o **centro**: a mediana;
- a **escala**: o intervalo interquartil (quartil 75% − quartil 25%).

E transforma com `(valor − mediana) / IQR`. É o "robust" do nome: mediana e IQR não se deixam puxar por outliers, diferente da média e do desvio padrão do `StandardScaler`.

| Coluna | Centro (mediana treino) | Escala (IQR treino) |
|---|---:|---:|
| Glucose | 117 | 40 |
| BMI | 31,95 | 9,075 |

### Nó 11 — `transform_scalers`

**A paciente:**

- `Glucose: (162 − 117) / 40 = 1,125`
- `BMI: (37,2 − 31,95) / 9,075 = 0,579`
- `Insulin: (119 − 119) / IQR = 0,0`: ela recebeu exatamente a mediana, então fica no centro.

As categóricas encodadas **não** são escalonadas: continuam 0, 1, 2...

---

## A paciente no fim: uma linha da `master_table`

```
Outcome=1  split=test
Pregnancies=0.0  Glucose=1.125  BloodPressure=-1.25  SkinThickness=1.25
Insulin=0.0  BMI=0.579  DiabetesPedigreeFunction=0.801  Age=-0.312
NEW_AGE_CAT=0  NEW_BMI_CAT=1  NEW_GLUCOSE_CAT=1  NEW_GLUCOSE_LEVEL=1
NEW_INSULIN_SCORE=1  NEW_AGE_BMI_CAT=2  NEW_AGE_GLUCOSE_CAT=2
NEW_GLUCOSE_INSULIN=0.626  NEW_GLUCOSE_PREGNANCIES=0.264
```

19 colunas: 17 features + alvo + split. É isso que o modelo recebe.

(Spoiler do bloco 07: o RandomForest dá a ela 82,8% de probabilidade de diabetes. Ela tem.)

---

## Como o `pipeline.py` liga tudo

```python
Node(func=fit_imputers,
     inputs=["split_diabetes_data", "params:columns", "params:modelling_imputers"],
     outputs="modelling_imputers",
     name="fit_imputers"),
Node(func=impute_missing_values,
     inputs=["split_diabetes_data", "modelling_imputers"],
     outputs="imputed_diabetes_data",
     name="impute_missing_values"),
```

A saída `modelling_imputers` do primeiro é entrada do segundo. É só isso que o Kedro precisa para saber a ordem. `inputs` é uma lista na **mesma ordem** dos argumentos da função: `split_diabetes_data` vira `df_in`, `params:columns` vira `columns`, `params:modelling_imputers` vira `params`.

## Para praticar

```powershell
uv run kedro run --pipeline=data_engineering
uv run kedro ipython
```

E no `ipython`:

```python
df = catalog.load("master_table")
df["split"].value_counts()
catalog.load("modelling_imputers")          # o dicionário de medianas
catalog.load("modelling_encoders")["NEW_BMI_CAT"].classes_
```
