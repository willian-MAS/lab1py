# Bloco 02 — Kedro: os fundamentos

## 2.1 O problema que o Kedro resolve

Um notebook é ótimo para explorar e péssimo para produção. O slide 20 da aula 1 lista por quê:

| Problema do notebook | Consequência |
|---|---|
| a ordem das células importa | quem roda de novo, em outra ordem, tem outro resultado |
| `pd.read_csv("../data/x.csv")` no meio do código | trocar a origem do dado exige editar código |
| hiperparâmetros espalhados pelas células | ninguém sabe qual configuração gerou qual modelo |
| limpeza, features e modelo misturados | impossível reaproveitar só uma parte |
| `fit` no dataset inteiro antes de separar treino e teste | **data leakage**: a avaliação fica otimista |
| só roda dentro do Jupyter | outro sistema não consegue pedir uma predição |

O Kedro é um framework (criado pela QuantumBlack, hoje na Linux Foundation) que obriga o código a seguir uma estrutura que evita esses problemas. Ele **não** é um agendador (como o Airflow) nem uma plataforma de MLOps. Ele organiza o código.

## 2.2 As peças

### Nó (node)

Uma função Python **pura**, embrulhada com nome de entrada e de saída.

"Pura" quer dizer: recebe dados pelos argumentos, devolve dados pelo `return`, e não faz mais nada. Não lê arquivo, não grava arquivo, não altera a entrada.

```python
def clean_data(raw_diabetes_data: pd.DataFrame, columns: dict) -> pd.DataFrame:
    df = raw_diabetes_data.copy()       # nunca altera a entrada
    ...
    return df                            # devolve o resultado
```

E o embrulho, que diz de onde vem e para onde vai cada coisa:

```python
Node(
    func=clean_data,
    inputs=["raw_diabetes_data", "params:columns"],
    outputs="cleaned_diabetes_data",
    name="clean_data",
)
```

`raw_diabetes_data` e `cleaned_diabetes_data` não são variáveis Python: são **nomes de datasets do catálogo**. O Kedro carrega o primeiro, chama a função e salva o resultado no segundo.

### Pipeline

Uma lista de nós. A **ordem de execução não é declarada**: o Kedro deduz pelos nomes. Se um nó produz `cleaned_diabetes_data` e outro consome `cleaned_diabetes_data`, o segundo roda depois do primeiro.

### DAG

*Directed Acyclic Graph*, grafo direcionado sem ciclos. É o desenho que o `kedro viz` mostra: caixas (nós e datasets) ligadas por setas (quem alimenta quem). "Acíclico" quer dizer que nenhuma seta volta para trás, então sempre existe uma ordem válida de execução.

Uma consequência: nós sem dependência entre si podem rodar em paralelo (`ParallelRunner`).

### Data Catalog (`conf/base/catalog.yml`)

O registro de **onde** cada dado mora e **em que formato**:

```yaml
raw_diabetes_data:
  type: pandas.CSVDataset
  filepath: data/01_raw/diabetes-dataset-modelling.csv

production_model:
  type: pickle.PickleDataset
  filepath: data/06_models/production_model.pkl
```

É o contrato entre o código e o armazenamento. O nó pede `raw_diabetes_data`; se amanhã esse dado estiver num bucket da AWS em Parquet, muda-se o YAML e **nenhuma linha de Python**.

Tipos que usamos:

| Tipo | Guarda | Exemplo |
|---|---|---|
| `pandas.CSVDataset` | DataFrame em CSV | tabelas |
| `pickle.PickleDataset` | qualquer objeto Python | modelos, encoders, scalers |
| `json.JSONDataset` | dict ou lista | métricas, predições |
| `MemoryDataset` | fica só na memória | usado quando um dataset não está no catálogo |

**Regra importante:** se um nome de dataset não aparece no catálogo, o Kedro cria um `MemoryDataset` automaticamente. O dado passa de um nó para outro na memória e some no fim. Isso vai ser central na API (bloco 09).

### Parâmetros (`conf/base/parameters*.yml`)

Toda configuração: listas de colunas, proporções do split, hiperparâmetros. No nó, entram com o prefixo `params:`:

```python
inputs=["cleaned_diabetes_data", "params:split"]
```

Isso injeta o bloco `split:` do YAML como um dicionário no argumento da função.

O Kedro junta **todos** os arquivos `parameters*.yml` num lugar só. Separamos em um arquivo por pipeline só para organizar.

### Camadas de dados (`data/01_raw` ... `data/08_reporting`)

Uma convenção de "maturidade" do dado:

| Camada | Conteúdo | No diabetes |
|---|---|---|
| `01_raw` | dado original, **imutável** | os 2 CSV do exercício |
| `02_intermediate` | limpo e tipado | `cleaned_*` |
| `03_primary` | entidades do negócio | `split_*` (com a coluna de split) |
| `04_feature` | features | imputado, truncado, com features, encodado |
| `05_model_input` | pronto para o modelo | `master_table`, `scaled_inference_data` |
| `06_models` | artefatos treinados | modelos, encoders, scalers, imputadores |
| `07_model_output` | saída do modelo | `inference_predictions.json` |
| `08_reporting` | relatórios | `baseline_metrics.json`, `optimized_metrics.json` |

Cada dataset intermediário salvo em disco é um **checkpoint**. Se o pipeline quebra no nó 20, dá para recomeçar dali (`--from-nodes`) sem refazer os 19 anteriores.

## 2.3 A anatomia do projeto

```
diabetes/
├── conf/
│   ├── base/          ← configuração versionada (vale para todo mundo)
│   ├── local/         ← configuração da sua máquina e senhas (fora do git)
│   └── logging.yml    ← como os logs aparecem
├── data/              ← as camadas
├── notebooks/         ← exploração
├── src/diabetes/
│   ├── pipelines/
│   │   └── <nome>/
│   │       ├── nodes.py      ← as funções
│   │       └── pipeline.py   ← o encaixe (Node + inputs/outputs)
│   ├── pipeline_registry.py  ← registra as pipelines
│   └── settings.py           ← configurações do Kedro
├── tests/
└── pyproject.toml
```

### `pipeline_registry.py`

```python
def register_pipelines():
    pipelines = find_pipelines(raise_errors=True)
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
```

`find_pipelines()` procura toda pasta dentro de `pipelines/` que tenha um `create_pipeline()`. O `__default__` é a soma de todas: é o que roda quando você digita só `kedro run`.

### `conf/base` × `conf/local`

`local` sobrescreve `base`. Serve para configuração que muda de máquina para máquina e para credenciais (`credentials.yml`), que **nunca** vão para o git. É o fator "configuração no ambiente, não no código" do 12-Factor App (slide 8 da aula 2).

## 2.4 Os comandos

| Comando | O que faz |
|---|---|
| `kedro new --name x --tools=...` | cria o scaffold do projeto |
| `kedro pipeline create nome` | cria a pasta de uma pipeline nova |
| `kedro run` | roda o `__default__` (tudo) |
| `kedro run --pipeline=modelling` | roda uma pipeline |
| `kedro run --nodes=clean_data` | roda um nó |
| `kedro run --from-nodes=refit_model` | de um nó em diante |
| `kedro run --to-nodes=fit_encoders` | até um nó |
| `kedro viz run` | sobe o site do DAG em `http://127.0.0.1:4141` |
| `kedro viz build` | gera o site do DAG como arquivos estáticos (foi assim que o professor gerou o `churn-viz` do material) |
| `kedro ipython` | abre um Python já com `catalog` carregado: `catalog.load("master_table")` |
| `kedro info` | mostra a versão e os plugins |

Sempre com `uv run` na frente, para usar o Kedro do `.venv`.

## 2.5 O Kedro e os 12 fatores (aula 2)

| Fator | Como o Kedro atende |
|---|---|
| Base de código | estrutura padrão, compatível com git |
| Dependências | `pyproject.toml` + `uv.lock` |
| Configuração | `conf/base` × `conf/local` |
| Serviços de apoio | Data Catalog: a fonte do dado é um recurso trocável |
| Build, release, run | `kedro package` / Docker / `kedro run` |
| Processos sem estado | nós são funções puras |
| Port binding | a API (FastAPI) expõe as pipelines numa porta |
| Descartabilidade | checkpoints: dá para retomar de onde quebrou |
| Paridade dev/prod | mesmo código, só muda a configuração |
| Logs | `logging.yml` e `logger.info(...)` em vez de `print` |
