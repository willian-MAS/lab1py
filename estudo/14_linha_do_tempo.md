# Bloco 14 — Linha do tempo

Tudo o que foi feito, em ordem.

---

## Aulas 1 e 2 (setembro)

- O professor apresentou o Kedro e construiu ao vivo o projeto de churn (Telco): 4 pipelines, catálogo, parâmetros. A aula não foi gravada.

## Reconstrução do projeto de churn

- No material da disciplina, a pasta `code/churn-viz/` era a build estática do `kedro viz` do professor. Os arquivos em `api/nodes/` guardavam o código de cada nó, o tipo e o caminho de cada dataset e os valores de todos os parâmetros.
- A partir disso, o projeto `churn/` foi reconstruído com `kedro new` e `kedro pipeline create`, e comparado com a build do professor: **5 pipelines, 47 elementos e 61 arestas idênticos**.
- O guia `03_aulas_1_2_passo_a_passo_churn.md` documenta cada passo.

## Reprodução no seu Windows

1. `uvx kedro new` falhou por **DNS** (rede). Resolvida a rede, funcionou.
2. Havia restos de tentativas anteriores (`churn_old`, um `.venv` e um `src/` soltos na raiz). Limpamos.
3. A pasta `DeployKedros` foi renomeada para `DeployKedro`. O primeiro rename falhou porque o **VSCode estava com a pasta aberta**.
4. `uv add` falhou com **hardlink / operação de nuvem**. Solução: `UV_LINK_MODE=copy`.
5. `uv add --dev` falhou por **conflito entre a lista dev antiga do scaffold e a nova**. Solução: apagar `[project.optional-dependencies]`.
6. `Copy-Item` com a **linha quebrada** copiou os CSV para o lugar errado. Movidos para `data/01_raw`.
7. `kedro run`: **18 de 18 nós**, com métricas idênticas às da reconstrução (reprodutibilidade entre máquinas).
8. 7 treinos do CatBoost falharam por **disputa pela pasta de log** no GridSearchCV paralelo. Solução: `allow_writing_files: false`.

## O exercício de diabetes

- Leitura do notebook (195 células) e dos dados: 652 + 116 linhas, zeros impossíveis em 5 colunas.
- Leitura do repositório do professor (`datadiversbr/insper-productive-data-science`), que trouxe o padrão de `api.py`, `Dockerfile` e `docker-compose.yml`.
- Desenho das 4 pipelines com fit/transform separados (31 nós). Modelo campeão: RandomForest; baseline: LogisticRegression.
- Correção de um bug do notebook (`NEW_AGE_BMI_NOM`).
- API FastAPI com 8 rotas e testes automáticos. O Docker foi escrito seguindo o padrão do professor.

## Publicação e testes finais

1. Repositório público `PADS_kedro` criado vazio e o projeto enviado para a `main`.
2. No seu Windows: `uv sync`, `kedro run` (**31 de 31 em 125,8 s**), `kedro viz`.
3. O commit foi reescrito para ter só você como autor e para tirar o terceiro integrante. Isso exigiu `git fetch` + `git reset --hard origin/main` na sua máquina.
4. O `CMD` do Dockerfile ganhou `--no-dev`, para não instalar ferramentas de desenvolvimento ao subir.
5. `docker compose up --build` no seu Windows: API no ar, `POST /inference` respondeu **200** no Swagger.
6. Versão pública com documentação enxuta.
7. Correção das faixas de idade e glicose (`right: false`): 50 anos passou a ser "senior", como no notebook.
8. Correção na API: paciente incompleto agora recebe **422**, e não 500. Valor `null` é imputado.
9. Este repositório de estudo, privado.

## Entrega

- Prazo: 25/09.
- Link do repositório público + `.zip` (botão **Code → Download ZIP**).
- Para: ThanuciS@insper.edu.br, com cópia para donaldn@insper.edu.br.
