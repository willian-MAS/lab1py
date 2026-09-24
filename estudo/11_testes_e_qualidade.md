# Bloco 11 — Testes e qualidade

Arquivos para abrir junto: a pasta `diabetes/tests/`.

---

## 11.1 Por que testar

Um teste é um pedaço de código que chama o seu código e confere o resultado. Serve para duas coisas:

1. **provar que funciona hoje**;
2. **avisar quando alguém quebrar amanhã**: você muda um nó, roda os testes, e se algo que funcionava parou, você sabe na hora.

## 11.2 Como rodar

```powershell
uv run pytest
```

Saída esperada: `36 passed`, mais uma tabela de **cobertura** (quanto do código foi executado pelos testes; aqui ~80%).

## 11.3 A anatomia de um teste

```python
def test_mediana_vem_apenas_do_split_de_treino():
    # PREPARA: um DataFrame pequeno, feito à mão
    df_in = pd.DataFrame({
        "split":   ["train", "train", "train", "test"],
        "Glucose": [100.0,   200.0,   300.0,   9999.0],
        ...
    })
    params = {"columns": ["numerical"], "strategy": "median", "split_to_fit": ["train"]}

    # EXECUTA
    imputers = fit_imputers(df_in, COLUMNS, params)

    # CONFERE
    assert imputers["Glucose"] == 200.0     # mediana de 100, 200, 300; o 9999 do teste ficou de fora
```

- O nome começa com `test_`, e é assim que o pytest acha.
- `assert condição`: se a condição for falsa, o teste falha e mostra o porquê.
- Não precisa de Kedro, de catálogo nem de arquivo. É a vantagem de os nós serem **funções puras**: dá para testar cada uma isolada.

## 11.4 Os arquivos de teste

| Arquivo | O que testa |
|---|---|
| `tests/test_run.py` | o projeto carrega, as 5 pipelines existem, cada uma tem o número certo de nós, e a inferência só consome artefatos `production_*` |
| `tests/pipelines/data_engineering/test_nodes.py` | cada nó da engenharia de dados, com DataFrames feitos à mão |
| `tests/pipelines/inference/test_nodes.py` | `to_dataframe` e `predict` |
| `tests/test_api.py` | todos os endpoints da API |

Os testes que protegem as ideias centrais do projeto:

| Teste | Protege |
|---|---|
| `test_mediana_vem_apenas_do_split_de_treino` | não há vazamento na imputação |
| `test_encoder_nao_aprende_categoria_exclusiva_do_teste` | não há vazamento no encoding |
| `test_categoria_desconhecida_nao_derruba_a_inferencia` | categoria nova vira -1, não exceção |
| `test_e_reprodutivel` | o split sorteia igual toda vez |
| `test_inferencia_so_consome_artefatos_de_producao` | a inferência nunca usa um artefato de modelagem por engano |
| `test_campo_ausente_e_recusado_com_422` | a API recusa paciente incompleto com o código certo |

## 11.5 Testando a API sem subir servidor: `TestClient`

```python
from fastapi.testclient import TestClient
from diabetes.api import app

with TestClient(app) as client:
    resposta = client.post("/inference", json={"instances": [...]})
    assert resposta.status_code == 200
```

O `TestClient` "finge" ser um cliente HTTP e fala direto com o `app`, na memória. Não precisa de uvicorn nem de porta.

### Fixtures

```python
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client
```

Uma **fixture** prepara algo que vários testes usam. O teste declara `client` como argumento, e o pytest entrega. `scope="module"` = cria uma vez e reusa em todos os testes do arquivo.

### `pytest.skip`

Alguns testes da API precisam de modelos treinados. Se alguém rodar os testes num clone novo, sem `kedro run`, esses testes são **pulados** (não falham), com a mensagem "rode `kedro run` antes".

### `monkeypatch`: trocando uma peça por um dublê

O teste do `POST /train` não pode rodar o pipeline de verdade (demoraria e sobrescreveria os dados). Então ele troca a função que roda o pipeline por uma falsa:

```python
@pytest.fixture(autouse=True)
def _executor_falso(self, monkeypatch):
    def _executor(run_id, pipeline_names):
        api._set_run(run_id, status="completed", finished_at="2026-01-01T00:00:00Z")
    monkeypatch.setattr(api, "_run_pipelines_background", _executor)
```

Assim testamos a **mecânica** do endpoint (gera `run_id`, registra, consulta, dá 404 para id inexistente) sem o custo do treino.

## 11.6 Ruff: lint e formatação

```powershell
uv run ruff check src tests      # lint: aponta problemas
uv run ruff format src tests     # formatação: padroniza o estilo
```

- **Lint** acha coisas como import que não é usado, variável que não existe, `print` esquecido (a regra `T201` proíbe `print`: use `logger`).
- **Formatação** padroniza espaços, quebras de linha e aspas. Ninguém discute estilo em revisão de código.

As regras estão no `pyproject.toml`, em `[tool.ruff.lint]`. Nos testes desligamos a regra `PLR2004` ("número mágico"), porque em teste comparar com `200` ou `404` direto é mais legível.

Com o `.vscode/settings.json` do projeto, o VSCode formata sozinho a cada Ctrl+S.

## 11.7 Logging em vez de `print`

```python
logger = logging.getLogger(__name__)
logger.info("Imputados %d valores nulos", filled)
```

Diferente de `print`, o log tem nível (`INFO`, `WARNING`, `ERROR`), horário e origem, e pode ir para arquivo. É o que aparece colorido no `kedro run`. A configuração fica em `conf/logging.yml`.
