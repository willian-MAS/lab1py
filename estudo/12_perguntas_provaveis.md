# Bloco 12 — Perguntas prováveis do professor

Tente responder cada uma em voz alta **antes** de ler a resposta.

---

## Sobre o desenho

**1. Por que quatro pipelines, e não uma só?**

> Porque cada uma tem um papel e um momento diferente. A `data_engineering` e a `modelling` aprendem e avaliam, usando só o treino. O `refit` refaz os artefatos com todos os dados para produção. A `inference` só aplica, e pode rodar sozinha, várias vezes, sem retreinar. Separadas, dá para rodar cada uma quando precisa (`--pipeline=inference`) e a API pode disparar só a parte que interessa.

**2. Qual o maior problema do notebook original para produção?**

> Data leakage. Ele calcula a mediana de imputação, os limites de outlier, os encoders e o scaler com a base inteira antes do `train_test_split`, então a avaliação usa informação do teste. Além disso, nada disso fica salvo para ser aplicado num paciente novo.

**3. Como vocês evitaram o vazamento?**

> Separando fit de transform. Cada função `fit_*` recebe `split_to_fit` e só olha para as linhas daqueles splits. Na modelagem é `[train]`. A função `transform_*` só aplica o que foi aprendido, em qualquer linha. Tem um teste que prova isso: `test_mediana_vem_apenas_do_split_de_treino`.

**4. Para que serve o refit, se o modelo já foi treinado?**

> Na modelagem treinamos com 452 linhas para ter uma avaliação honesta. Depois de validada a abordagem, o modelo de produção é retreinado com as 652, com a mesma classe e os mesmos hiperparâmetros vencedores. Mais dados, mesmo método. O custo é que o modelo de produção não tem métrica própria; a estimativa que vale é a do `validate` na modelagem.

**5. Por que três splits?**

> Train para ajustar, test para comparar os modelos entre si, validate como holdout final. Se usássemos o mesmo conjunto para escolher e para reportar, o número reportado seria otimista.

**6. O que acontece se chegar uma categoria nova em produção?**

> Vira -1. O `transform_encoders` usa um dicionário com `.map()` em vez do `LabelEncoder.transform`, que lançaria exceção. Tem teste para isso.

## Sobre o Kedro

**7. O que é o Data Catalog e por que usar?**

> É o registro de onde cada dado está e em que formato. O código dos nós nunca lê nem grava arquivo; ele pede um dataset pelo nome. Se o dado mudar de CSV local para Parquet num bucket, muda o YAML e nenhuma linha de Python.

**8. Como o Kedro sabe a ordem de execução?**

> Pelos nomes de entrada e saída dos nós. Se um nó produz `master_table` e outro consome `master_table`, o segundo roda depois. Isso forma um DAG, que é o que o `kedro viz` desenha.

**9. O que é o `raw_inference_dataframe` e por que ele não está no catálogo?**

> É a saída do `to_dataframe`. Sem entrada no catálogo, o Kedro usa um `MemoryDataset`. Isso deixa a inferência pronta para receber dados de fora, e a API aproveita: troca a entrada e os intermediários por `MemoryDataset` e roda a mesma pipeline com o JSON da requisição.

**10. Como trocar o RandomForest por XGBoost?**

> Instalar o `xgboost` e mudar o `class_path` em `parameters_modelling.yml` para `xgboost.XGBClassifier`, ajustando o `param_grid`. O código carrega a classe com `importlib`, então não muda nada em Python.

## Sobre o modelo

**11. Por que ROC AUC e não acurácia para escolher o modelo?**

> A base tem 65% de negativos. Um modelo que diz "não" para todo mundo acerta 65%. A ROC AUC mede se o modelo ordena bem as pacientes (doente com nota maior que saudável) e não é enganada pelo desbalanceamento.

**12. Por que `class_weight: balanced`?**

> Para compensar o desbalanceamento e porque, em triagem, deixar passar uma paciente doente custa mais do que um falso alarme. Ele aumenta o recall. O grid testou com e sem, e escolheu com.

**13. No split de teste, o RandomForest tem acurácia menor que o baseline. Por que ficaram com ele?**

> No teste ele tem recall bem maior (0,60 × 0,43) e AUC um pouco maior, e no validate ganha em todas as métricas. O `class_weight=balanced` troca precisão por recall, o que pode derrubar a acurácia. E o teste tem só 109 linhas: os dois modelos caem juntos nele, o que indica que é a amostra, não o modelo.

**14. O RandomForest tem 0,99 de AUC no treino e 0,89 no validate. É overfitting?**

> Sim, parcialmente. Floresta aleatória com árvores profundas decora parte do treino. O que importa é o desempenho fora dele, que é bom (0,89 no validate, 0,82 na base de inferência separada).

**15. O tratamento de outliers faz diferença?**

> Quase nenhuma, e é bom saber por quê: o notebook usa quantis 5% e 95% com 1,5 × IQR, o que gera limites enormes (glicose entre −70 e 331). Só 3 valores foram truncados. Mantivemos para ser fiel ao notebook; com os quartis clássicos (25/75), seria uma mudança de uma linha no YAML.

**16. Qual feature mais pesou?**

> `NEW_GLUCOSE_INSULIN`, o produto glicose × insulina, com 17% da importância, seguida de `Glucose`, `Age` e `BMI`. A engenharia de features do notebook se pagou.

**17. Vocês encontraram algum problema no notebook?**

> Na célula que monta `NEW_AGE_BMI_NOM`, as duas últimas condições usam `BMI > 18.5` em vez de `BMI >= 30`, e como vêm por último sobrescrevem as outras: quase todo mundo vira "obese". Fizemos a combinação concatenando as faixas já calculadas.

## Sobre a API e o Docker

**18. Como a API faz a inferência online?**

> Ela abre uma sessão Kedro, troca no catálogo a entrada e os intermediários da inferência por `MemoryDataset`, coloca os registros recebidos no de entrada e roda a pipeline `inference` com o `SequentialRunner`. É exatamente a mesma pipeline do modo batch, com os mesmos artefatos de produção.

**19. Por que o `/train` devolve um `run_id` em vez da resposta?**

> O treino leva cerca de um minuto. A API dispara numa thread em segundo plano e responde na hora com um identificador; o cliente consulta o status em `GET /train/{run_id}`. Limitação: o registro fica em memória, então se perde se o servidor reiniciar.

**20. Como o dataset é exposto pela API?**

> `GET /datasets` lista o catálogo e `GET /datasets/{nome}` carrega o dataset pelo catálogo e devolve em JSON, com paginação por `limit` e `offset`. Modelos em pickle, que não viram JSON, respondem 415.

**21. Por que o Dockerfile instala as dependências antes de copiar o código?**

> Por causa do cache de camadas. Se o código mudar, o Docker reaproveita a camada das ~150 bibliotecas e só refaz a instalação do nosso pacote. Com a ordem invertida, qualquer mudança no código reinstalaria tudo.

**22. Para que servem os volumes no docker-compose?**

> `./conf` entra somente leitura, para mudar parâmetros sem rebuildar a imagem. `./data` entra com escrita, para os modelos gerados dentro do container aparecerem na máquina e para o container usar os modelos já treinados.

**23. Por que `--host 0.0.0.0`?**

> Com `127.0.0.1`, o uvicorn só aceitaria conexões de dentro do próprio container. `0.0.0.0` aceita conexões de fora, e é assim que o navegador do Windows chega nele pela porta mapeada.

## Sobre o ambiente

**24. Para que serve o `uv.lock`?**

> Guarda a versão exata de cada pacote instalado, inclusive das dependências das dependências. Com ele, `uv sync` recria o mesmo ambiente em qualquer máquina e no Docker (`--frozen`).

**25. O que é `split_to_fit` e por que ele é o centro do projeto?**

> É o parâmetro que diz em quais linhas cada `fit_*` aprende. A mesma função serve à modelagem, com `[train]`, e à produção, com todos os splits. Uma linha de YAML separa avaliação honesta de modelo de produção, sem nenhum `if` no código.
