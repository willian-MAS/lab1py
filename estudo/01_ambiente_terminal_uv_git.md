# Bloco 01 — Ambiente: terminal, uv, venv e git

Antes de qualquer Kedro, existe uma camada de ferramentas que você usou o tempo todo. Este bloco explica cada uma e cada erro que apareceu no caminho.

---

## 1.1 O terminal (PowerShell)

O terminal é um programa onde você dá ordens ao computador escrevendo, em vez de clicando. No Windows, o terminal padrão é o **PowerShell**.

Três ideias que resolvem 90% das dúvidas:

**Você sempre está "dentro" de uma pasta.** O prompt mostra qual:

```
PS C:\PADS_kedro>
```

Quase todo comando age sobre essa pasta. Por isso `uv run kedro run` só funciona dentro da pasta do projeto: o Kedro procura o `pyproject.toml` ali.

**`cd` troca de pasta.**

```powershell
cd C:\PADS_kedro      # caminho absoluto: começa no disco
cd data\01_raw        # caminho relativo: parte de onde você está
cd ..                 # sobe uma pasta
```

**Cada comando é uma linha.** Se a colagem quebrar a linha no meio, o PowerShell executa a primeira metade sozinha e tenta rodar a segunda como se fosse outro comando. Foi o que aconteceu no seu `Copy-Item`: a primeira metade copiou o arquivo para a pasta atual (porque não tinha destino), e a segunda metade deu "não é reconhecido como nome de cmdlet".

### Comandos de servidor "travam" o terminal, e é de propósito

`uv run kedro viz run`, `uv run uvicorn ...` e `docker compose up` sobem **servidores**. Um servidor fica rodando, esperando requisições, até você mandar parar. Enquanto isso o terminal fica ocupado. Não é travamento.

- Para parar: **Ctrl+C**.
- Para rodar outra coisa ao mesmo tempo: abra outra aba no **+** do terminal.

### Equivalências PowerShell × Linux/Mac

Os slides do professor usam comandos de Linux. No PowerShell:

| Linux / Mac | PowerShell |
|---|---|
| `ls` | `dir` (o `ls` também funciona) |
| `cp origem destino` | `Copy-Item origem -Destination destino` |
| `mv` | `Move-Item` |
| `rm -rf pasta` | `Remove-Item pasta -Recurse -Force` |
| `export VAR=valor` | `$env:VAR="valor"` |
| `\` no fim da linha (continua na próxima) | `` ` `` (crase), ou escreva tudo numa linha |
| `echo "x" > arquivo` | `Set-Content -Path arquivo -Value "x" -Encoding ascii` |

O último item importa: `echo "3.11" > .python-version` no PowerShell grava o arquivo em UTF-16, e o uv não consegue ler. Por isso usamos `Set-Content ... -Encoding ascii`.

---

## 1.2 Ambiente virtual (venv)

**Problema:** o projeto A precisa do pandas 2; o projeto B precisa do pandas 3. Se os dois usam o mesmo Python da máquina, um quebra o outro.

**Solução:** cada projeto ganha a sua pasta `.venv`, com uma cópia isolada do Python e **apenas** os pacotes daquele projeto.

```
C:\PADS_kedro\
├── .venv\            ← Python + pacotes só deste projeto
│   ├── Lib\site-packages\kedro\ ...
│   └── Scripts\python.exe
├── src\
└── pyproject.toml
```

Regras práticas:

- O `.venv` **nunca** vai para o git (está no `.gitignore`), porque tem centenas de MB e é específico da máquina.
- Não se edita o `.venv` à mão. Se quebrar: apague e rode `uv sync`.
- O `.venv` guarda caminhos absolutos. Se você renomear a pasta do projeto, ele quebra. Foi por isso que sugeri renomear `DeployKedros` antes de criar o venv.

---

## 1.3 uv: o gerenciador de pacotes

O `uv` substitui `pip` + `virtualenv` + `poetry`. Ele cria o venv, instala pacotes e grava as versões exatas.

### Os dois arquivos que ele gerencia

| Arquivo | Analogia | Conteúdo |
|---|---|---|
| `pyproject.toml` | a lista de compras | "preciso de pandas>=2.2, kedro~=1.6" — você escreve (ou o `uv add` escreve) |
| `uv.lock` | a nota fiscal | "instalei pandas 3.0.5, numpy 2.3.1, ... e mais 180 pacotes" — o uv escreve |

É o `uv.lock` que garante que na sua máquina, na da Nicole, na do professor e dentro do Docker instalem **exatamente** as mesmas versões. É a reprodutibilidade do slide 12 da aula 1.

### Os comandos

| Comando | O que faz |
|---|---|
| `uvx kedro new ...` | roda o Kedro num ambiente **temporário**, sem instalar nada. Usado só para criar o projeto, quando ainda não existe venv. |
| `uv add pacote` | adiciona o pacote ao `pyproject.toml`, resolve as versões, atualiza o `uv.lock` e instala no `.venv` |
| `uv add --dev pacote` | mesmo, mas como dependência só de desenvolvimento (pytest, ruff) — não vai para o Docker |
| `uv sync` | deixa o `.venv` exatamente igual ao `uv.lock`. É o comando de quem clona o projeto. |
| `uv run comando` | roda o comando **usando o `.venv` do projeto**, sem precisar ativar nada |
| `uv export ... -o requirements.txt` | gera um `requirements.txt` a partir do lock, para quem usa pip |

### `uvx` × `uv run`

- `uvx kedro ...` → Kedro de fora do projeto, temporário.
- `uv run kedro ...` → Kedro de dentro do `.venv` do projeto.

Depois que o projeto existe, é sempre `uv run`.

---

## 1.4 Os erros de ambiente que você encontrou (e o motivo de cada um)

### Erro 1 — `dns error ... Este host não é conhecido (os error 11001)`

O computador não conseguiu traduzir `pypi.org` em um endereço IP. Sem isso, o uv não baixa nada. Causa típica: VPN, rede corporativa, DNS instável. Não tinha nada a ver com o comando.

### Erro 2 — `kedro : O termo 'kedro' não é reconhecido`

O Kedro não estava instalado em nenhum lugar que o PowerShell conhecesse. Por isso usamos `uvx kedro new` (baixa e roda na hora).

### Erro 3 — `Rename-Item: O processo não pode acessar o arquivo porque está sendo usado`

O VSCode estava com a pasta aberta. O Windows não deixa renomear uma pasta que outro programa está segurando. Solução: fechar o VSCode e renomear pelo PowerShell do Windows.

### Erro 4 — `failed to hardlink ... A operação de nuvem não pode ser executada (os error 396)`

Para economizar disco, o uv não copia os pacotes para o `.venv`: ele cria **hardlinks** (um "atalho de baixo nível" apontando para o cache dele). O OneDrive e o Python da Microsoft Store não se dão bem com hardlinks. Solução:

```powershell
$env:UV_LINK_MODE = "copy"                                             # vale para este terminal
[Environment]::SetEnvironmentVariable("UV_LINK_MODE","copy","User")   # grava para os próximos
```

Isso manda o uv copiar em vez de linkar.

### Erro 5 — `churn:dev and churn[dev] are incompatible`

O scaffold do Kedro cria uma lista de dependências de desenvolvimento no formato antigo (`[project.optional-dependencies] dev`, com `pytest-mock<2.0`). O `uv add --dev` cria uma lista nova no formato moderno (`[dependency-groups] dev`, com `pytest-mock>=3.14`). O uv tenta satisfazer as duas, e elas pedem versões incompatíveis. Solução: apagar a lista antiga do `pyproject.toml`.

### Erro 6 — `Can't create train working dir: data/tmp/catboost_info`

Esse foi no projeto de churn. O CatBoost grava arquivos de log durante o treino. O `GridSearchCV` roda com `n_jobs=-1` (vários processos em paralelo), e os processos tentaram criar a mesma pasta ao mesmo tempo. Solução: `allow_writing_files: false` no YAML, para o CatBoost não gravar nada.

---

## 1.5 Git e GitHub

### Os conceitos

| Termo | O que é |
|---|---|
| **repositório** | uma pasta cujo histórico o git acompanha |
| **commit** | uma "foto" do projeto num momento, com uma mensagem dizendo o que mudou |
| **branch** | uma linha paralela de trabalho. A principal se chama `main`. |
| **remote / origin** | a cópia do repositório que está no GitHub |
| **clone** | baixar um repositório do GitHub pela primeira vez |
| **push** | enviar seus commits para o GitHub |
| **fetch** | baixar as novidades do GitHub, sem mexer nos seus arquivos |
| **`.gitignore`** | lista do que o git deve ignorar (`.venv`, dados gerados, credenciais) |

### Os comandos que você rodou

```powershell
git clone https://github.com/willian-MAS/PADS_kedro.git   # baixa o repositório
git fetch origin                                          # busca novidades do GitHub
git reset --hard origin/main                              # iguala sua cópia à do GitHub
```

Por que o `reset --hard` foi necessário: eu reescrevi o commit no GitHub (para tirar o coautor e o nome do Alfredo). Reescrever = substituir o commit por outro, com outro identificador (`75f1196` virou `2c48c7e`). A sua cópia local ainda tinha o antigo. O `fetch` trouxe o novo, e o `reset --hard origin/main` fez a sua pasta ficar igual ao GitHub.

O `reset --hard` só mexe em arquivos que o git acompanha. Os arquivos gerados pelo `kedro run` (em `data/`) estão no `.gitignore`, então não foram tocados.

### Por que o repositório tinha que nascer vazio

Se você marca "Add README" ao criar, o GitHub faz um primeiro commit. Aí existem duas histórias diferentes (a do GitHub e a local), e o primeiro push é recusado. Vazio, o push simplesmente vira a primeira história.

### Por que os dados brutos estão no git, mas o resto de `data/` não

O `.gitignore` do Kedro ignora `data/**` inteiro, porque dados geralmente são grandes ou sensíveis. Abrimos uma exceção só para `data/01_raw/*.csv`, porque o enunciado pede o repositório com os datasets. Todo o resto (`02_intermediate` a `08_reporting`) é **regenerado** pelo `kedro run`, então não precisa ser versionado.
