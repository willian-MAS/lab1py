# Bloco 13 — Glossário

Uma frase por termo. Em ordem alfabética.

| Termo | Significado |
|---|---|
| **422** | status HTTP "pedido mal formado": faltou campo ou veio tipo errado |
| **404** | status HTTP "não encontrado" |
| **415** | status HTTP "formato não suportado"; usamos para datasets que não viram JSON |
| **500** | status HTTP "erro no servidor": algo quebrou do nosso lado |
| **API** | interface para um programa pedir coisas a outro pela rede |
| **artefato** | objeto treinado e salvo em disco para reuso (modelo, encoder, scaler, imputador) |
| **baseline** | modelo simples usado como régua de comparação |
| **batch** | processar muitos registros de uma vez, a partir de um arquivo |
| **branch** | linha paralela de trabalho no git; a principal é `main` |
| **build (Docker)** | executar o Dockerfile para gerar uma imagem |
| **cache de camadas** | o Docker reaproveita as etapas do build que não mudaram |
| **catálogo (Data Catalog)** | YAML que diz onde cada dataset mora e em que formato |
| **checkpoint** | dataset intermediário salvo, que permite retomar o pipeline dali |
| **class_path** | caminho completo de uma classe, em texto, para carregá-la pelo YAML |
| **class_weight** | peso de cada classe no treino; `balanced` compensa desbalanceamento |
| **CLI** | interface de linha de comando (terminal) |
| **clone** | baixar um repositório do GitHub pela primeira vez |
| **commit** | uma "foto" do projeto com mensagem, no git |
| **container** | uma imagem Docker em execução, isolada |
| **cobertura (coverage)** | quanto do código foi executado pelos testes |
| **cross-validation (cv)** | dividir o treino em partes e alternar qual parte avalia, para uma nota mais estável |
| **DAG** | grafo direcionado sem ciclos: o desenho de quem alimenta quem no pipeline |
| **data leakage** | informação de avaliação vazando para o treino; deixa a métrica otimista |
| **dataset** | no Kedro, qualquer dado com nome no catálogo (tabela, modelo, JSON) |
| **decorador** | o `@app.get(...)` acima de uma função; liga a rota à função |
| **dependência** | biblioteca de terceiros que o projeto usa |
| **Docker Desktop** | o programa que roda Docker no Windows/Mac |
| **docker-compose.yml** | arquivo que descreve como rodar o(s) container(s): portas, volumes |
| **Dockerfile** | a receita para montar a imagem |
| **encoder / encoding** | transformar categorias em números |
| **endpoint (rota)** | um endereço da API, como `/inference` |
| **F1** | média harmônica entre precision e recall |
| **FastAPI** | framework Python para escrever APIs |
| **feature** | variável de entrada do modelo |
| **feature engineering** | criar variáveis novas a partir das existentes |
| **fit** | aprender a partir dos dados (mediana, categorias, coeficientes...) |
| **fixture** | no pytest, algo preparado uma vez e entregue aos testes |
| **`.gitignore`** | lista do que o git não deve versionar |
| **GridSearchCV** | testa todas as combinações de uma grade de hiperparâmetros com cross-validation |
| **hardlink** | "atalho de baixo nível" de arquivo; o uv usa para economizar disco |
| **healthcheck** | checagem periódica de que o serviço está vivo |
| **hiperparâmetro** | configuração do modelo escolhida antes do treino (n_estimators, max_depth) |
| **holdout** | parte dos dados guardada para a avaliação final (`validate`) |
| **HTTP** | o protocolo de pedido e resposta da web |
| **imagem (Docker)** | o ambiente empacotado e congelado, pronto para virar container |
| **imputação** | preencher valores faltantes |
| **inferência** | usar o modelo treinado para prever dados novos |
| **IQR** | intervalo interquartil: distância entre dois quantis |
| **JSON** | formato de texto para dados estruturados (`{"chave": valor}`) |
| **Kedro** | framework que organiza código de ciência de dados em pipelines de nós |
| **kedro viz** | ferramenta que desenha o DAG no navegador |
| **label encoding** | cada categoria vira um número inteiro |
| **lint** | análise automática que aponta problemas no código |
| **localhost** | "este mesmo computador" |
| **lockfile (`uv.lock`)** | versões exatas de todos os pacotes instalados |
| **MemoryDataset** | dataset que só existe na memória durante a execução |
| **monkeypatch** | trocar temporariamente uma função por outra num teste |
| **nó (node)** | função pura embrulhada com nomes de entrada e saída |
| **one-hot** | cada categoria vira uma coluna de 0/1 |
| **outlier** | valor muito fora do padrão |
| **overfitting** | o modelo decora o treino e vai pior em dados novos |
| **parâmetros (params)** | configuração em YAML, injetada nos nós com `params:` |
| **pickle** | formato do Python para salvar qualquer objeto em disco |
| **pipeline** | conjunto de nós que forma um DAG |
| **porta** | número onde um servidor escuta conexões (8000, 4141) |
| **precision** | dos que o modelo disse "sim", quantos eram |
| **`predict_proba`** | probabilidade de cada classe, em vez do "sim/não" |
| **Pydantic** | biblioteca que define e valida formatos de dados |
| **`pyproject.toml`** | arquivo de configuração do projeto Python e suas dependências |
| **pytest** | ferramenta de testes do Python |
| **push** | enviar commits para o GitHub |
| **recall** | dos que eram "sim", quantos o modelo encontrou |
| **refit** | retreinar com todos os dados, mantendo método e hiperparâmetros |
| **repositório** | pasta com histórico controlado pelo git |
| **RobustScaler** | escala com mediana e IQR, resistente a outliers |
| **ROC AUC** | chance de o modelo dar nota maior a um positivo do que a um negativo |
| **Ruff** | ferramenta de lint e formatação para Python |
| **run_id** | identificador de uma execução em segundo plano na API |
| **scaffold** | esqueleto de projeto gerado automaticamente |
| **scaler** | objeto que coloca números numa escala comum |
| **seed / random_state** | semente que torna o sorteio reprodutível |
| **servidor** | programa que fica rodando, esperando pedidos |
| **split** | divisão da base em train, test e validate |
| **Swagger** | a página `/docs`, gerada automaticamente pelo FastAPI |
| **`split_to_fit`** | parâmetro que diz em quais splits um `fit_*` aprende |
| **TestClient** | cliente HTTP falso para testar a API sem subir servidor |
| **thread** | linha de execução paralela dentro do mesmo programa |
| **transform** | aplicar o que foi aprendido no fit |
| **uv / uvx** | gerenciador de pacotes e ambientes; `uvx` roda ferramentas temporárias |
| **uvicorn** | servidor que roda aplicações FastAPI |
| **venv (`.venv`)** | ambiente Python isolado de um projeto |
| **volume** | pasta do computador compartilhada com o container |
| **WSL2** | Linux dentro do Windows; o Docker Desktop usa por baixo |
| **YAML** | formato de configuração legível (`chave: valor`, indentação por espaços) |
