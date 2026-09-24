# Como estudar este material

Este repositório é privado e serve só para estudo. Ele junta três coisas:

| Pasta | O que é |
|---|---|
| `estudo/` | Os guias, em blocos, do básico ao avançado. Comece por aqui. |
| `churn/` | O projeto das aulas 1 e 2 (Telco Churn), igual ao que o professor mostrou. |
| `diabetes/` | O projeto do exercício de avaliação, na versão com comentários detalhados no código. |

## Ordem sugerida

Cada bloco se apoia no anterior. Se tiver pouco tempo, os blocos marcados com ⭐ são os que o professor mais provavelmente vai perguntar.

| # | Bloco | Tempo | O que você vai entender |
|---|---|---|---|
| 01 | [Ambiente: terminal, uv, venv e git](./01_ambiente_terminal_uv_git.md) | 30 min | O que cada comando que você rodou fez, e por que os erros aconteceram |
| 02 ⭐ | [Kedro: os fundamentos](./02_kedro_fundamentos.md) | 40 min | Nó, pipeline, DAG, catálogo, parâmetros, camadas de dados |
| 03 | [Aulas 1 e 2: o projeto de churn](./03_aulas_1_2_passo_a_passo_churn.md) | 30 min | O projeto de referência que o diabetes imita |
| 04 ⭐ | [Diabetes: o problema e o planejamento](./04_diabetes_problema_e_planejamento.md) | 30 min | Por que o notebook não serve para produção e como as 4 pipelines resolvem isso |
| 05 | [Diabetes: a configuração](./05_diabetes_configuracao.md) | 30 min | `catalog.yml` e os `parameters*.yml`, chave por chave |
| 06 ⭐ | [Diabetes: engenharia de dados](./06_diabetes_data_engineering.md) | 60 min | Os 11 nós, um por um, com exemplos numéricos |
| 07 ⭐ | [Diabetes: modelagem e refit](./07_diabetes_modelling_e_refit.md) | 45 min | Baseline, GridSearchCV, métricas e por que o refit existe |
| 08 | [Diabetes: inferência](./08_diabetes_inference.md) | 20 min | Como o mesmo código serve para dado novo |
| 09 ⭐ | [API com FastAPI](./09_api_fastapi.md) | 45 min | O que é uma API, cada endpoint, e o truque do MemoryDataset |
| 10 ⭐ | [Docker](./10_docker.md) | 45 min | Imagem, container, Dockerfile linha a linha e o que aconteceu no seu `docker compose up` |
| 11 | [Testes e qualidade](./11_testes_e_qualidade.md) | 20 min | pytest, TestClient, Ruff |
| 12 ⭐ | [Perguntas prováveis do professor](./12_perguntas_provaveis.md) | 30 min | Treino para defender o trabalho |
| 13 | [Glossário](./13_glossario.md) | consulta | Todos os termos técnicos em uma frase |
| 14 | [Linha do tempo](./14_linha_do_tempo.md) | 10 min | Tudo o que foi feito, em ordem, do zero à entrega |

## Como ler os guias

Abra o VSCode com o projeto ao lado (`code C:\PADS_kedro_estudo\diabetes`) e, a cada bloco, abra o arquivo citado. Ler o guia sem olhar o código rende metade.

Quando o guia mostrar um comando, rode. Quando mostrar um trecho de código, procure no arquivo real e veja o que vem antes e depois.

## Rodando os projetos deste repositório

Cada projeto tem o próprio ambiente:

```powershell
cd C:\PADS_kedro_estudo\diabetes
uv sync
uv run kedro run

cd C:\PADS_kedro_estudo\churn
uv sync
uv run kedro run
```
