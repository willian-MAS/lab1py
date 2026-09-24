# Bloco 10 — Docker

Arquivos para abrir junto: `diabetes/Dockerfile`, `diabetes/docker-compose.yml` e `diabetes/.dockerignore`.

---

## 10.1 O problema que o Docker resolve

Você viveu isso: Python da Microsoft Store, OneDrive bloqueando hardlink, DNS, Windows × Linux. Cada máquina tem as suas particularidades, e "na minha funciona" não é garantia de nada.

O Docker empacota **o sistema operacional inteiro** (um Linux mínimo), o Python, as bibliotecas e o seu código numa caixa. Essa caixa roda igual no seu Windows, no Mac do professor e num servidor na nuvem.

## 10.2 Os cinco conceitos

| Conceito | Analogia | O que é |
|---|---|---|
| **Dockerfile** | a receita | arquivo de texto com os passos para montar o ambiente |
| **Imagem** | o bolo congelado | o resultado de executar a receita; fica guardada, imutável |
| **Container** | uma fatia servida | uma imagem **rodando**; você pode ter vários containers da mesma imagem |
| **docker-compose.yml** | o garçom | diz como rodar o container: portas, pastas compartilhadas, variáveis |
| **Volume** | uma janela | uma pasta do seu computador que aparece dentro do container |

E o **Docker Desktop** é o programa que roda tudo isso no Windows. Por baixo, ele mantém uma pequena máquina Linux (via WSL2), porque os containers são Linux.

### Container × máquina virtual

Uma máquina virtual simula um computador inteiro, com o próprio kernel: pesada, minutos para ligar. Um container compartilha o kernel do hospedeiro e isola só o resto: leve, sobe em segundos.

## 10.3 O Dockerfile, linha por linha

```dockerfile
FROM python:3.11-slim
```
Ponto de partida: uma imagem oficial com Debian Linux mínimo (`slim`) e Python 3.11 instalado. Vem do Docker Hub.

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
```
Copia os programas `uv` e `uvx` de **outra** imagem (a oficial do uv) para dentro da nossa. É mais rápido e limpo do que instalar o uv com pip.

```dockerfile
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
```
Variáveis de ambiente:
- `UV_COMPILE_BYTECODE=1`: pré-compila os `.py` em `.pyc`, e o container sobe mais rápido;
- `UV_LINK_MODE=copy`: o mesmo ajuste de hardlink que você fez no Windows;
- `PYTHONUNBUFFERED=1`: os logs aparecem na hora, sem ficar presos num buffer.

```dockerfile
WORKDIR /app
```
Cria e entra na pasta `/app` dentro do container. Todo o resto acontece ali.

```dockerfile
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
```
**A parte mais inteligente do arquivo.** Copia só os arquivos de dependências e instala as bibliotecas:
- `--frozen`: usa o `uv.lock` exatamente como está, sem resolver de novo;
- `--no-dev`: sem pytest e ruff, que não são necessários para servir a API;
- `--no-install-project`: ainda não instala o nosso código (ele nem foi copiado).

```dockerfile
COPY conf/ conf/
COPY src/ src/
COPY data/01_raw/ data/01_raw/
```
Agora sim, a configuração, o código e os dados brutos.

```dockerfile
RUN mkdir -p data/02_intermediate data/03_primary ... data/tmp
```
Cria as outras camadas vazias, para o Kedro ter onde gravar.

```dockerfile
RUN uv sync --frozen --no-dev
```
Instala o nosso pacote (`diabetes`). As bibliotecas já estão lá, então é rápido.

```dockerfile
EXPOSE 8000
```
Documenta que o container escuta a porta 8000. (Quem abre a porta de verdade é o compose.)

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1
```
A cada 30 segundos, o Docker chama o nosso `/health`. Se falhar 3 vezes seguidas, marca o container como `unhealthy`. O Docker Desktop mostra esse status.

```dockerfile
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "diabetes.api:app", "--host", "0.0.0.0", "--port", "8000"]
```
O comando que roda quando o container liga: sobe a API.
- `--no-dev`: sem isso, o `uv run` tentaria instalar pytest e ruff toda vez que o container subisse.
- `--host 0.0.0.0`: **importante**. `127.0.0.1` quer dizer "só aceito conexões de dentro deste container". `0.0.0.0` quer dizer "aceito de qualquer lugar", inclusive do seu Windows, lá fora.

### Por que separar dependências e código (o cache de camadas)

Cada instrução do Dockerfile gera uma **camada**, e o Docker guarda cada camada em cache. No próximo build, ele reaproveita todas as camadas até a primeira que mudou.

```
FROM python:3.11-slim                 ← cache
COPY uv ...                           ← cache
COPY pyproject.toml uv.lock ...       ← cache (você não mexeu nas dependências)
RUN uv sync ... --no-install-project  ← cache (a parte lenta: ~150 pacotes!)
COPY src/ src/                        ← MUDOU (você editou um nó)
RUN uv sync ...                       ← refaz (rápido)
```

Se o `COPY src/` viesse antes do `uv sync`, qualquer vírgula mudada no código faria o Docker reinstalar as ~150 bibliotecas. É o "two-phase install for layer caching" do slide 34.

## 10.4 O `.dockerignore`

Lista o que **não** entra na imagem:

- `.venv/`: o venv do Windows não serve no Linux do container, e a imagem cria o dela;
- `.git/`, caches, notebooks;
- `data/02_intermediate` a `data/08_reporting`: são gerados pelo pipeline (ou chegam pelo volume);
- `conf/local/`: credenciais nunca vão para uma imagem.

Imagem menor, build mais rápido, e nada sensível dentro.

## 10.5 O `docker-compose.yml`, linha por linha

```yaml
services:
  api:                              # o nome do serviço
    build: .                        # monte a imagem com o Dockerfile desta pasta
    image: diabetes-api:latest      # e chame a imagem assim
    container_name: diabetes-api    # nome do container (aparece no Docker Desktop)
    ports:
      - "8000:8000"                 # porta DO SEU PC : porta DO CONTAINER
    volumes:
      - ./conf:/app/conf:ro         # pasta do PC : pasta do container : somente leitura
      - ./data:/app/data            # leitura e escrita
    environment:
      - DO_NOT_TRACK=1              # desliga a telemetria do Kedro
    restart: unless-stopped         # se cair, o Docker religa sozinho
```

### `ports: "8000:8000"`

O container é uma caixa fechada. Esta linha abre um "túnel": o que chegar na porta 8000 do seu Windows é repassado para a porta 8000 do container. Poderia ser `"9000:8000"`, e aí você acessaria `localhost:9000`.

### `volumes`

- `./conf:/app/conf:ro`: o container enxerga a sua pasta `conf`. Se você mudar um parâmetro, não precisa rebuildar a imagem. `:ro` = *read-only*: o container não pode alterar a sua configuração.
- `./data:/app/data`: o container enxerga a sua pasta `data`. Foi por isso que o `/health` respondeu `production_artifacts_ready: true` logo de cara: os modelos que o seu `kedro run` gerou estavam em `C:\PADS_kedro\data\06_models`, e o container os viu. E, se você rodar `POST /train` pela API do container, os modelos novos aparecem na sua pasta.

Sem volume, tudo o que o container grava some quando ele é apagado.

## 10.6 O que aconteceu quando você rodou `docker compose up --build`

Aquele monte de texto rolando, em ordem:

1. **O compose leu o `docker-compose.yml`** e viu que o serviço `api` tem `build: .`.
2. **Enviou a pasta do projeto para o Docker** (o *build context*), menos o que está no `.dockerignore`.
3. **`FROM python:3.11-slim`**: baixou a imagem base do Docker Hub (algumas dezenas de MB), só na primeira vez.
4. **`COPY --from=ghcr.io/astral-sh/uv`**: baixou a imagem do uv e copiou os programas.
5. **`RUN uv sync ... --no-install-project`**: baixou e instalou ~150 pacotes (Kedro, pandas, scikit-learn, FastAPI...) **dentro** da imagem. É a etapa que mais demora.
6. **`COPY conf/ src/ data/01_raw/`** e **`RUN uv sync`**: colocou o nosso código e o instalou.
7. **Salvou a imagem** como `diabetes-api:latest`. Ela aparece em **Images** no Docker Desktop.
8. **Criou o container** `diabetes-api` a partir da imagem, abriu a porta 8000 e montou as pastas `conf` e `data`. Ele aparece em **Containers**.
9. **Executou o `CMD`**: subiu o uvicorn. Você viu `Uvicorn running on http://0.0.0.0:8000`.
10. **O terminal "parou"**, porque o servidor ficou rodando e mostrando os logs de cada requisição.
11. Quando você abriu `localhost:8000/docs`, o pedido saiu do navegador, entrou pela porta 8000 do Windows, atravessou o túnel até o container, chegou ao uvicorn → FastAPI → Kedro → modelo, e a resposta fez o caminho de volta.

Na segunda vez que você rodar `docker compose up --build`, os passos 3 a 5 vêm do cache e o build leva segundos, a menos que as dependências tenham mudado.

**Importante:** se o código mudar (por exemplo, depois de um `git pull`), rode com `--build` de novo. Sem `--build`, o compose reusa a imagem antiga, com o código antigo.

## 10.7 Comandos do dia a dia

| Comando | O que faz |
|---|---|
| `docker compose up --build` | monta a imagem (se preciso) e sobe o container, preso ao terminal |
| `docker compose up -d` | sobe em segundo plano (*detached*), liberando o terminal |
| `docker compose logs -f` | mostra os logs do container (Ctrl+C sai dos logs, o container continua) |
| `docker compose ps` | lista os containers do projeto e o status (inclusive `healthy`) |
| `docker compose down` | para e **apaga** o container (a imagem e a pasta `data/` continuam) |
| `docker compose exec api bash` | abre um terminal **dentro** do container, para explorar (`exit` para sair) |
| `docker images` | lista as imagens baixadas/construídas |
| `docker system df` | quanto espaço o Docker está ocupando |

Para parar o que está preso ao terminal: **Ctrl+C**, e depois `docker compose down`.

## 10.8 Rodar sem compose (para entender o que ele faz por você)

O compose é só um jeito organizado de escrever este comando:

```powershell
docker build -t diabetes-api:latest .
docker run --name diabetes-api -p 8000:8000 -v ${PWD}/conf:/app/conf:ro -v ${PWD}/data:/app/data diabetes-api:latest
```

Sem os `-v` (volumes), o container usaria as cópias de `conf/` e `data/01_raw/` que estão dentro da imagem. Aí não haveria modelos treinados: `/health` diria `production_artifacts_ready: false`, e seria preciso chamar `POST /train` primeiro.
