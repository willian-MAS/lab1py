# Bloco 04 — Diabetes: o problema e o planejamento

Arquivos para abrir junto: `diabetes/notebooks/diabetes-prediction.ipynb` e `diabetes/PLANEJAMENTO_PIPELINES.md`.

---

## 4.1 O problema de negócio

Prever se uma paciente tem diabetes a partir de 8 medidas de exame. Os dados vêm do estudo Pima Indians (mulheres de 21 anos ou mais, Arizona).

| Coluna | Significado |
|---|---|
| `Pregnancies` | número de gestações |
| `Glucose` | glicose no plasma 2h após teste oral de tolerância |
| `BloodPressure` | pressão diastólica (mm Hg) |
| `SkinThickness` | espessura da dobra cutânea do tríceps (mm) |
| `Insulin` | insulina 2h após o teste (mu U/ml) |
| `BMI` | índice de massa corporal |
| `DiabetesPedigreeFunction` | índice de histórico familiar |
| `Age` | idade |
| `Outcome` | **alvo**: 1 = diabetes, 0 = não |

Dois arquivos:

- `diabetes-dataset-modelling.csv`: **652 linhas**, para treinar e avaliar. 424 negativos e 228 positivos (65% × 35%, desbalanceado).
- `diabetes-dataset-inference.csv`: **116 linhas**, simulando "pacientes novos". Esse arquivo também traz o `Outcome` verdadeiro, então dá para conferir se as predições acertaram. O modelo nunca vê essa coluna.

## 4.2 O detalhe que muda tudo: zeros impossíveis

Contando zeros no arquivo de modelagem:

| Coluna | Zeros | Faz sentido? |
|---|---:|---|
| `Pregnancies` | 92 | sim, a paciente pode nunca ter engravidado |
| `Glucose` | 5 | não, glicose zero é incompatível com a vida |
| `BloodPressure` | 27 | não |
| `SkinThickness` | 192 | não |
| `Insulin` | **314** | não, é quase metade da base |
| `BMI` | 8 | não |

Esses zeros são **dados faltantes disfarçados**: o exame não foi feito, e alguém preencheu com 0. Se o modelo aprender com eles, vai achar que "insulina 0" é um valor real. Por isso o primeiro nó transforma esses zeros em `NaN` (nulo), e um nó adiante preenche com a mediana.

## 4.3 O que o notebook faz

| Seção | O que acontece |
|---|---|
| 4 a 12 | análise exploratória, gráficos, correlações |
| 13 | 9 modelos "crus", antes de qualquer tratamento |
| 15 | zeros viram nulo e são preenchidos com a mediana |
| 16 | outliers truncados (quantis 0,05 / 0,95 com 1,5 × IQR) |
| 17 | 8 variáveis novas (faixas de idade, IMC e glicose, combinações, produtos) |
| 18 | label encoding e one-hot |
| 19 | RobustScaler |
| 20 | 9 modelos com GridSearchCV e comparação |

## 4.4 Por que ele não serve para produção

### Problema 1 — Data leakage (vazamento)

Olhe a ordem do notebook: **primeiro** calcula a mediana, os limites de outlier, os encoders e o scaler usando as 652 linhas; **depois** faz o `train_test_split`.

Um exemplo concreto do que isso causa: a mediana de `Insulin` calculada só no treino é **119**; com a base inteira, é **123,5**. No notebook, o modelo é avaliado no teste usando uma mediana que já "viu" o teste. É como estudar para a prova com o gabarito: a nota fica boa, mas não mede o que vai acontecer com um paciente de verdade.

A regra certa (slide 21 da aula 1):

```
ERRADO:  fit(tudo)  →  separar  →  avaliar
CERTO:   separar  →  fit(só treino)  →  transform(tudo)  →  avaliar
```

### Problema 2 — Nada é reaproveitável

A mediana, os limites e os encoders existem só como variáveis soltas numa sessão do Jupyter. Quando chegar uma paciente nova amanhã, não há como aplicar **exatamente** o mesmo tratamento que o modelo viu no treino.

### Problema 3 — Configuração espalhada

Pontos de corte clínicos (IMC 18,5 / 24,9 / 29,9; glicose 140 / 200) e hiperparâmetros estão dentro das células. Mudar um exige caçar no código.

### Problema 4 — Resultado depende da ordem das células

Rode as células fora de ordem e o resultado muda.

### Problema 5 — Sem interface

Outro sistema não consegue pedir uma predição.

## 4.5 A abordagem analítica: 4 pipelines

O critério para separar:

- o que **aprende** a partir dos dados fica na `data_engineering` e na `modelling`;
- o que **vai para produção** é refeito no `refit`;
- o que só **aplica** fica na `inference`.

```
┌────────────────────────── DATA ENGINEERING (11 nós) ─────────────────────────┐
│ raw → clean → split → fit_imputers → impute → fit_outliers → clip           │
│     → features → fit_encoders → encode → fit_scalers → scale → master_table │
└──────────────────────────────────────────────────────────────────────────────┘
                 │                                          │
┌──── MODELLING (4 nós) ────┐              ┌──────── REFIT (8 nós) ────────────┐
│ baseline: LogisticReg.    │              │ refaz imputers, outliers,         │
│ otimizado: RandomForest   │─ campeão ──→ │ encoders, scalers e o modelo      │
│   + GridSearchCV          │              │ usando TODOS os splits            │
│ avaliação dos dois        │              │ ⇒ 5 artefatos production_*        │
└───────────────────────────┘              └───────────────────────────────────┘
                                                         │
┌──────────────────────────── INFERENCE (8 nós) ─────────┴─────────────────────┐
│ dado novo → clean → impute → clip → features → encode → scale → predict     │
│ (as MESMAS funções da data_engineering, alimentadas com os artefatos        │
│  de produção; nada é ajustado aqui)                                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

Total: 31 nós.

## 4.6 Os três splits e o papel de cada um

O nó `add_split_column` sorteia cada linha para um grupo (semente 42, então o sorteio é sempre o mesmo):

| Split | Linhas | Papel |
|---|---:|---|
| `train` | 452 | ajusta tudo: medianas, limites, encoders, scalers, modelos |
| `test` | 109 | compara os modelos entre si |
| `validate` | 91 | holdout final: a estimativa honesta de desempenho |

Por que três e não dois? Se você usa o teste para escolher o modelo, o teste passa a fazer parte da escolha, e a nota dele fica otimista. O `validate` fica guardado para o fim.

O baseline e o RandomForest treinam no **mesmo** split (`train`), para a comparação ser justa.

## 4.7 Por que existe o refit

Na modelagem tudo foi ajustado com 452 linhas, para a avaliação ser honesta. Depois de saber que a abordagem funciona, não faz sentido jogar fora 200 linhas: o modelo de produção é retreinado com as 652, usando **a mesma classe e os mesmos hiperparâmetros vencedores**. O mesmo vale para medianas, limites, encoders e scalers.

Custo: o modelo de produção não tem uma métrica "limpa" própria (ele viu todos os dados). A estimativa de desempenho que vale é a do `validate` na modelagem.
