# Bloco 09 — API com FastAPI

Arquivo para abrir junto: `diabetes/src/diabetes/api.py`.

---

## 9.1 O que é uma API (sem jargão)

Até agora, para ter uma predição, alguém precisava abrir um terminal e rodar `kedro run`. Uma **API** permite que **outro programa** peça uma predição pela rede: um site, um app de celular, o sistema do hospital.

A conversa segue o protocolo **HTTP** (o mesmo do navegador):

```
CLIENTE (navegador, curl, outro sistema)            SERVIDOR (nossa API)
         │                                                   │
         │  POST /inference                                  │
         │  {"instances": [{"Glucose": 148, ...}]}           │
         │ ───────────────────────────────────────────────▶ │
         │                                                   │  roda a pipeline
         │  200 OK                                           │
         │  {"predictions": [{"prediction": 1, ...}]}        │
         │ ◀─────────────────────────────────────────────── │
```

| Conceito | O que é | Exemplo |
|---|---|---|
| **rota** (endpoint) | o "endereço" de uma função da API | `/inference`, `/health` |
| **método** | o tipo de pedido | `GET` = consultar; `POST` = enviar dados / disparar ação |
| **corpo** (body) | os dados enviados, em JSON | `{"instances": [...]}` |
| **status** | um código de 3 dígitos dizendo como foi | `200` ok, `404` não achei, `422` pedido mal feito, `500` erro no servidor |
| **porta** | o "número da porta" onde o servidor escuta | `8000` |

`localhost` quer dizer "este mesmo computador".

## 9.2 As três ferramentas

| Ferramenta | Papel |
|---|---|
| **FastAPI** | o framework: você escreve funções Python e marca quais rotas elas atendem |
| **Pydantic** | define o formato dos dados de entrada e saída e **valida** automaticamente |
| **uvicorn** | o servidor que fica escutando a porta e entrega as requisições ao FastAPI |

```powershell
uv run uvicorn diabetes.api:app --host 0.0.0.0 --port 8000
```

`diabetes.api:app` = "no módulo `diabetes/api.py`, use o objeto chamado `app`".

## 9.3 Como uma rota é escrita

```python
app = FastAPI(title="Diabetes ML API", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ...
    return HealthResponse(status="ok", ...)
```

- `@app.get("/health")` é um **decorador**: diz ao FastAPI que um `GET` em `/health` deve chamar a função logo abaixo.
- `response_model=HealthResponse` diz o formato da resposta. O FastAPI converte para JSON e documenta.

## 9.4 Pydantic: o contrato e a validação

```python
class PatientRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    Pregnancies: float | None
    Glucose: float | None
    BloodPressure: float | None
    SkinThickness: float | None
    Insulin: float | None
    BMI: float | None
    DiabetesPedigreeFunction: float | None
    Age: float | None


class InferenceRequest(BaseModel):
    instances: list[PatientRecord]
```

Isso declara: o corpo do `POST /inference` precisa ter uma chave `instances` com uma lista de pacientes, e cada paciente precisa ter as 8 medidas.

O Pydantic **valida antes** da sua função rodar:

| O cliente manda | O que acontece |
|---|---|
| as 8 medidas | ✅ segue |
| `"Insulin": null` | ✅ aceita; vira `NaN` e é imputado pela mediana, como um zero impossível |
| sem `"Age"` | ❌ `422`, com a lista do que falta, e a função nem roda |
| `"Glucose": "abc"` | ❌ `422` |
| um campo a mais (`"Outcome": 1`) | ✅ aceito (`extra="allow"`) e ignorado pelo modelo |

Antes dessa validação, um paciente incompleto chegava até o fim da pipeline e estourava como `500` ("erro do servidor"). O correto é `422` ("o seu pedido está errado"), e foi o que corrigimos.

## 9.5 O Swagger (`/docs`)

O FastAPI lê as rotas e os modelos Pydantic e gera **sozinho** uma página de documentação interativa em `http://localhost:8000/docs`. Foi ali que você clicou em **Try it out** → **Execute**. O exemplo de paciente já vem preenchido porque ele foi declarado no `json_schema_extra` do modelo.

## 9.6 Inicializando o Kedro dentro da API

```python
_bootstrap_lock = threading.Lock()
_bootstrapped = False

def _ensure_bootstrap():
    global _bootstrapped
    if _bootstrapped:
        return
    with _bootstrap_lock:
        if not _bootstrapped:
            bootstrap_project(PROJECT_PATH)
            configure_project("diabetes")
            _bootstrapped = True
```

Para usar o Kedro de dentro de um programa (em vez de pelo terminal), é preciso "apresentar" o projeto uma vez: onde ele está e qual o pacote. Várias requisições podem chegar ao mesmo tempo (várias *threads*), então:

- o `Lock` garante que só uma thread faz a inicialização;
- o duplo `if` (antes e depois do lock) evita pegar o lock toda vez depois que já está inicializado.

O padrão se chama *double-checked locking* e é o mesmo do repositório do professor (slide 32 da aula 1).

A função `lifespan` roda `_ensure_bootstrap()` quando o servidor sobe.

## 9.7 Os endpoints, um por um

### `GET /health` — o serviço está de pé?

Responde `status: ok`, as pipelines registradas e se os 5 artefatos de produção existem (`catalog.exists(...)`). É o que o Docker consulta no `HEALTHCHECK`.

### `GET /datasets` e `GET /datasets/{name}` — expor um dataset (extra do enunciado)

```python
@app.get("/datasets/{name}")
def read_dataset(name: str, limit: int = Query(100, ge=1, le=10_000), offset: int = Query(0, ge=0)):
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        catalog = _get_catalog(session)
        data = catalog.load(name)          # o catálogo sabe onde e como ler
    ...
```

- `{name}` na rota vira um argumento: `/datasets/master_table` → `name = "master_table"`.
- `limit` e `offset` vêm da URL (`?limit=5&offset=10`): paginação, para não mandar 652 linhas de uma vez. `Query(..., ge=1, le=10_000)` valida o intervalo.
- `catalog.load(name)`: a API não sabe se aquilo é CSV, JSON ou pickle. É a abstração do catálogo chegando ao HTTP.

Respostas possíveis:

| Situação | Status |
|---|---|
| tabela (DataFrame) | `200` com `rows`, `columns` e `records` |
| lista ou dicionário (métricas, predições) | `200` |
| nome inexistente | `404` |
| existe mas ainda não foi gerado | `404` com a dica "rode POST /train" |
| objeto que não vira JSON (modelo em pickle) | `415` |

### `POST /train` e `GET /train/{run_id}` — treinar pela API

Treinar leva cerca de 1 minuto. Uma requisição HTTP não deve ficar esperando tanto. Solução: disparar em **segundo plano** e devolver um "número de protocolo".

```python
def _start_background_run(pipeline_label, pipeline_names):
    run_id = str(uuid.uuid4())                            # protocolo único
    _set_run(run_id, status="pending", pipeline=pipeline_label)
    threading.Thread(target=_run_pipelines_background,
                     args=(run_id, pipeline_names), daemon=True).start()
    return RunResponse(run_id=run_id, status="pending")   # responde na hora
```

A thread roda `data_engineering`, `modelling` e `refit` e vai atualizando o registro: `pending` → `running` → `completed` (ou `failed`, com o erro). O cliente consulta com `GET /train/{run_id}`.

Limitação: o registro (`runs`) fica na memória. Se o servidor reiniciar, os protocolos se perdem. Para produção de verdade, iria para um banco de dados.

### `POST /batch-inference` — inferência em lote

Mesmo mecanismo, rodando só a pipeline `inference` sobre o CSV do catálogo. O resultado vai para `inference_predictions.json` (consultável por `GET /datasets/inference_predictions`).

### `POST /inference` — o truque central

```python
ONLINE_OVERRIDES = ["raw_inference_data", "cleaned_inference_data", "imputed_inference_data",
                    "clipped_inference_data", "featured_inference_data",
                    "encoded_inference_data", "scaled_inference_data", "inference_predictions"]

with KedroSession.create(project_path=PROJECT_PATH) as session:
    catalog = _get_catalog(session)

    for dataset_name in ONLINE_OVERRIDES:
        catalog[dataset_name] = MemoryDataset()           # 1. troca arquivos por memória

    catalog["raw_inference_data"] = MemoryDataset(        # 2. injeta o JSON recebido
        data=pd.DataFrame([r.model_dump() for r in request.instances]))

    SequentialRunner().run(kedro_pipelines["inference"], catalog)   # 3. roda a MESMA pipeline
    predictions = catalog.load("inference_predictions")             # 4. lê o resultado da memória
```

1. No catálogo, `raw_inference_data` aponta para um CSV e os intermediários para arquivos. Aqui eles são trocados por `MemoryDataset`, que existe só durante essa requisição. Nada é lido nem gravado em disco, e duas requisições simultâneas não se atropelam nos arquivos.
2. O `MemoryDataset` de entrada já nasce com os pacientes da requisição.
3. Roda **exatamente a mesma pipeline `inference`** do modo batch: mesma limpeza, imputação, features, encoding e escala, com os mesmos artefatos de produção.
4. Lê as predições da memória e devolve.

Os artefatos de produção (`production_*`) **não** são trocados: continuam sendo lidos dos arquivos em `data/06_models/`.

É o slide 33 da aula 1: "Same inference pipeline for batch (CSV) and online (HTTP)".

## 9.8 Testando

**No navegador:** `http://localhost:8000/docs`.

**No PowerShell:** cuidado, no PowerShell 5 o `curl` é um apelido de outro comando. Use `curl.exe` ou o comando nativo:

```powershell
Invoke-RestMethod http://localhost:8000/health

$corpo = '{"instances":[{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}]}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/inference -ContentType "application/json" -Body $corpo
```

Resposta esperada para essa paciente: `prediction = 1`, probabilidade ≈ 0,89.

Um teste que mostra a validação:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/inference -ContentType "application/json" -Body '{"instances":[{"Glucose":148}]}'
```

Deve dar erro `422`, listando os 7 campos que faltam.
