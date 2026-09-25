# catalogo_match — casamento neuro-simbólico CATMAT ↔ e-Fisco

Casamento automático de itens entre dois catálogos de compras públicas: o
**e-Fisco/CADMAT** (Estado de Pernambuco, texto livre) e o **CATMAT** (Governo
Federal, rótulo estruturado `PDM, ATRIBUTO: VALOR, ...`). Dado um item do
e-Fisco, o sistema devolve os itens do CATMAT mais prováveis de serem o *mesmo
material*, com a métrica-alvo sendo o **Recall@3**: em quantas consultas o item
correto está entre os três primeiros.

O sistema é **neuro-simbólico**: um bi-encoder multilíngue (E5) faz a
similaridade semântica, e três camadas simbólicas — **rede semântica** (grafo
léxico de domínio), **ontologia/taxonomia** (OWL + SWRL e alinhamento das
classes dos dois catálogos) e **GraphRAG** (grafo de conhecimento de entidades
com comunidades) — corrigem o que o embedding não vê: abreviação, unidade
convertida, classe do material, atributo raro.

Nenhum módulo conhece o domínio. O que o sistema sabe sobre agulhas, gauge ou
cláusula do art. 31 da Lei 8.078/90 é **dado**, em `config/`, não código.

| Dataset | O que é | Pares | Perfil | Resultados |
|---|---|---|---|---|
| `mmh` | Material Médico-Hospitalar, ground truth 2026-04-08 | 1 012 consultas × 1 054 itens | `mmh` (curado) | [RESULTADOS.md](RESULTADOS.md) |
| `mmh_opme` | recorte OPME do MMH | 434 × 464 | `mmh` | [RESULTADOS.md](RESULTADOS.md) |
| `bigdata_profs` | pares ouro multi-domínio (mineração + profissionais de saúde), 2026-09-16 | 1 375 × 1 297 | `compras_publicas` (herda `mmh`) | [docs/RESULTADOS-BIGDATA-PROFS.md](docs/RESULTADOS-BIGDATA-PROFS.md) |

---

## 1. Arquitetura

### 1.1 Três camadas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DADO                    config/datasets/<nome>.yaml   onde está o corpus     │
│                          config/perfis/<nome>.yaml     o que se sabe do domínio│
│                          dados/<dataset>/*.csv          ground truth           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ALGORITMO (catalogo_match/)                                                 │
│                                                                              │
│   config ──► preprocessamento ──► caracteristicas ──► rede_semantica         │
│      │              │                    │                 │                 │
│      └──────► graphrag ──► taxonomia ──► ontologia_owl ──► pipeline          │
│                                                            avaliacao_modular │
│                                                            diagnostico       │
├─────────────────────────────────────────────────────────────────────────────┤
│  SAÍDA                   resultados/<dataset>/   métricas, matriz, grafos     │
│                          cache/<dataset>/        embeddings, textos, LLM      │
│                          graphrag_workspace/<dataset>/  índice MS GraphRAG    │
└─────────────────────────────────────────────────────────────────────────────┘
```

Tudo é **por dataset**: dois corpora nunca compartilham cache, saída ou índice.
Um índice de outro corpus reaproveitado em silêncio daria resultado errado com
cara de certo.

### 1.2 O conhecimento como dado: dataset × perfil

| Arquivo | Declara | Muda quando |
|---|---|---|
| `config/datasets/<nome>.yaml` | CSVs, separador, encoding, mapeamento de colunas, **qual perfil usa** | o **corpus** é outro |
| `config/perfis/<nome>.yaml` | boilerplate jurídico, stopwords, abreviações, variantes ortográficas, sinônimos, hierarquia de PDMs, **tabela de unidades** (com fatores de conversão e grafias), famílias, esquema de atributos, características definidoras, regras regex de extração, papéis dos prompts | o **domínio** é outro |

Os perfis formam uma cadeia de herança, resolvida por merge recursivo em que o
filho vence chave a chave:

```
base.yaml            português, SI, formato do rótulo CATMAT — nada de domínio
  └── mmh.yaml       material hospitalar: gauge, french, polegada, léxico curado
        └── compras_publicas.yaml   (herda: mmh) + massa/dose, concentração, UI, W, V
```

O que o perfil não traz, o pipeline **induz** do corpus (esquema de atributos,
famílias, hierarquia, stopwords, boilerplate, unidades) e registra como
`induzido` na proveniência do run — e o que não dá para induzir com honestidade
(fator de conversão, cláusula legal esparsa) fica como lacuna declarada. Ver
[docs/NOVO-DATASET.md](docs/NOVO-DATASET.md).

### 1.3 Dois modos de execução

**(A) Pipeline completo** — `python -m catalogo_match.pipeline` — uma
configuração fixa das fases [0]–[9] da Figura 3 do documento técnico, com
blocking, LLM e reasoner:

```
[0] fontes ─► [1] rede semântica ─► [2] extração de atributos (LLM, 8 threads; regex sem chave) + SHACL
 ─► [3] blocking (PDM, subsunção, família, classe, léxico + top-50 pela FUSÃO E5+TF-IDF)
 ─► [4] ontologia OWL + Pellet (SWRL) ─► [5] matcher: estágio 1 = fusão, estágio 2 = cross-encoder
 ─► [5b] sinais simbólicos (medida tipada, GraphRAG-IDF, taxonomia LOO) somados ao score neural
 ─► [6] Microsoft GraphRAG (índice + local search) adjudica a zona cinzenta, da menor margem para a maior
 ─► [8] confiança ─► [*] avaliação com ablação por camada ─► [7] grafo unificado ─► [8b] explicabilidade ─► [9] análise
```

O que a grade modular (B) mediu como melhor foi levado para dentro do pipeline,
e os padrões do pipeline são os medidos como melhores:

- a **fusão linear** E5 + TF-IDF de palavra + TF-IDF de caractere é a
  similaridade do blocking semântico, do corte dos blocos grandes e do estágio 1
  do matcher (sobre o texto normalizado, não sobre a expansão da rede);
- a fase **[5b]** soma ao score neural, com peso 0,05, quatro sinais na mesma
  escala: medida tipada, cobertura de entidades do KG por IDF, taxonomia (LOO) e
  a **dedução ontológica** da fase [4] (`equivalenteA` = 1, `broadMatch` = 0,5) —
  que antes entrava como fatia convexa de 0,35 do score final e, nessa escala,
  puxava pares errados ao topo;
- o **cross-encoder** fica desligado (`--cross` religa): −1,7 pp e 33 min de CPU;
- a fase **[6]** adjudica **em lista**: a LLM vê os 5 melhores candidatos das
  consultas de menor margem e pontua cada um; o julgamento par a par (`--adjudicacao par`)
  rebaixava corretos com especificação incompleta. O índice Microsoft GraphRAG só
  entra com `--contexto-ms`;
- a **confiança** (Alta/Média/Baixa) é dada pela **margem** entre o 1º e o 2º
  candidato, validada contra o gabarito (P@1 por faixa sai na avaliação).

A avaliação reporta a **ablação por camada** (cosseno E5 → fusão → cross →
sinais → fase [6] → score final) sobre o mesmo top-K, para que cada fase
responda pelo que acrescentou. Flags: `--sem-llm`, `--sem-rede`, `--cross`,
`--sem-sinais`, `--adjudicacao {lista,par,nenhuma}`, `--k-lista`,
`--max-adjudicacoes`, `--contexto-ms`, `--peso-rel`, `--peso-ontologia`,
`--fator-veto`, `--neural-texto rede`, `--k-rerank`, `--saida`.

**(B) Grade modular** — `python -m catalogo_match.avaliacao_modular` — cada
etapa vira um módulo trocável e **todas as combinações** são medidas lado a
lado, sem blocking (cada consulta contra o catálogo inteiro, teto 100%). É o
instrumento com que se decide o que funciona:

```
 PRÉ-PROCESSADOR            PROCESSADOR                     PÓS-PROCESSADOR
 (reescreve o texto)        (ranqueia o catálogo inteiro)   (reordena o top-K)

 nada            ┐          tfidf | fuzzy | e5 | e5_cross    ┌ nada
 basico          │          ─────────────────────────────    │ unidades | pdm
 rede_semantica  ├─► Textos ─► fusão de rankings (RRF) ──► top-K ─┤ graphrag | graphrag_idf
 graphrag        │          e5_tfidf | e5_kg | e5_tfidf_kg  │ medidas | taxonomia
 graphrag_leve   ┘          e5_taxonomia | e5_tfidf_taxonomia└ combinações somadas
                                                                     │
                                          avaliar ─► MRR, R@1, R@3, R@5, R@10
                                          matriz pré × proc · ranking · YAML · heatmap
```

Os pós-processadores só reordenam o top-K (20 candidatos). O que o processador
deixou na 30ª posição está perdido para eles — por isso a fusão de rankings
existe no eixo do **processador**: ela junta fontes com erros diferentes
*antes* do corte e ataca o teto.

### 1.4 Onde cada camada simbólica entra

**Rede semântica** (`rede_semantica.py`, fase [1]) — grafo léxico com cinco
tipos de nó (PDM, Conceito, Termo, ValorDeAtributo, Unidade) e nove relações
(`abreviacaoDe`, `varianteOrtograficaDe`, `sinonimoDe`, `hiperonimoDe`,
`mapeiaParaPDM`, `temUnidade`, `convertePara`, ...). Construído em seis passos:
(a) semente estrutural dos rótulos CATMAT, (b) mineração do histórico de
vínculos por PMI, (c) semente lexical curada do perfil, (d) indução de sinônimos
por fastText treinado no corpus, (e) fila de curadoria, (f) unidades. Serviço de
consulta: normalização token → forma canônica, expansão por *spreading
activation*, ancoragem em PDM. Exportado em GraphML e SKOS/RDF; servido por
FastAPI (`servico_rede_semantica.py`). Na grade é o pré-processador
`rede_semantica`.

**Ontologia OWL** (`ontologia_owl.py`, fase [4]) — TBox com um `owl:Class` por
PDM e a hierarquia do perfil, `owl:DatatypeProperty` por atributo, regra SWRL da
§2.2 (mesmo PDM + características definidoras iguais ⇒ `equivalenteA`) executada
pelo Pellet em lotes por PDM; `broadMatch` por subsunção; veto por conflito de
definidora, marcado como verificação Python e não dedução.

**Taxonomia** (`taxonomia.py`) — os dois catálogos *já são* ontologias de
classificação (e-Fisco: grupo > classe; CATMAT: grupo > classe > PDM). O módulo
alinha as duas e produz a compatibilidade entre a classe da consulta e a do
candidato, por dois caminhos: **histórico**, P(classe CATMAT | classe e-Fisco)
estimada nos vínculos conhecidos em *leave-one-out* (a consulta avaliada nunca
contribui para a própria estimativa), e **rótulo**, similaridade E5 entre os
nomes das classes, sem vínculo nenhum — que serve de controle. Ambos caem para o
nível de grupo quando a classe é desconhecida. Na grade: pós-processador
`taxonomia` e processador `e5_taxonomia` (o sinal aplicado ao catálogo inteiro,
antes do corte).

**GraphRAG** (`graphrag.py`) — em dois sabores. O **oficial** (Microsoft
GraphRAG v3) indexa um `.txt` por item, extrai entidades e comunidades com LLM e
serve `local_search` para adjudicar pares da zona cinzenta na fase [6]; exige
`OPENAI_API_KEY`. O **offline** constrói, sem API, um KG bipartido item ↔
entidade (PDM, valores de atributo, tokens) com comunidades Louvain sobre a
projeção consulta ↔ consulta, e entra na grade em **três eixos**:

| Eixo | Módulo | O que faz |
|---|---|---|
| pré | `graphrag`, `graphrag_leve` | reescreve a consulta com entidades dos vizinhos por busca local + rótulo da comunidade |
| proc | `e5_kg`, `e5_tfidf_kg` | a busca local vira um ranking completo (soma do IDF das entidades compartilhadas) e é fundida ao E5 por RRF |
| pós | `graphrag`, `graphrag_idf` | cobertura das entidades da consulta pelo candidato — peso fixo, ou ponderada pelo IDF da entidade no catálogo |

**Medida tipada** (`caracteristicas.py`) — valor + unidade extraídos do texto
livre e convertidos à base da categoria (`3 1/2"` → 88,9 mm; `500 MG` → 0,5 g;
`210X86X162 CM` → três comprimentos), comparados com tolerância. É o que o
bi-encoder é estruturalmente cego para. Tudo vem da tabela `unidades` do perfil;
unidade sem `aliases` não é extraída.

### 1.5 Fluxo de uma execução da grade

1. `config.ativar(dataset)` carrega o `DatasetSpec` e o `PerfilDominio` (com a
   cadeia de herança).
2. `carregar_corpus` lê o CSV, separa **consultas** (e-Fisco únicos),
   **catálogo** (CATMAT únicos) e **gabarito**, e completa o perfil por indução.
3. Cada pré-processador gera `Textos(queries, catalogo)`, cacheado em
   `cache/<dataset>/embeddings/pre_*.json` (a assinatura inclui o fonte dos
   módulos e o YAML do perfil — mudar uma stopword invalida).
4. Cada processador ranqueia o catálogo inteiro e retém o top-K. Embeddings E5
   ficam em `e5_*.npz`, indexados pelo texto.
5. Cada pós-processador soma ou mistura o seu sinal ao score do top-K e reordena.
6. `avaliar` compara com o gabarito; `exportar` grava
   `avaliacao_modular.yaml` (com a proveniência do perfil e o que cada sinal viu
   no corpus), `matriz_pre_x_proc.csv`, `ranking_combinacoes.csv` e
   `matriz_modular.png`.
7. `diagnostico` responde *onde* o R@3 se perde: posição do correto por faixa,
   por origem, por grupo, por presença de medida; e quem entrou/saiu do top-3
   entre duas combinações.

---

## 2. Como executar

```bash
pip install -r requirements.txt      # Python 3.13; Java no PATH para o Pellet

# Grade modular (sem LLM, sem blocking) — o instrumento de medição
python -m catalogo_match.avaliacao_modular --dataset bigdata_profs --metrica recall_at_3
python -m catalogo_match.avaliacao_modular --dataset bigdata_profs \
    --pre nada,basico --proc e5,e5_tfidf,e5_tfidf_taxonomia \
    --pos nada,medidas,medidas_taxonomia,medidas_graphrag_idf_taxonomia --metrica recall_at_3
python -m catalogo_match.avaliacao_modular --listar          # módulos disponíveis
python -m catalogo_match.avaliacao_modular --dataset mmh     # reproduz RESULTADOS.md
python -m catalogo_match.avaliacao_modular --dataset bigdata_profs --perfil base   # sem curadoria

# Onde o R@3 se perde, e o que mudou entre duas combinações
python -m catalogo_match.diagnostico --dataset bigdata_profs --pre nada --proc e5 --pos nada \
    --contra nada,e5_tfidf,medidas_taxonomia

# O que o extrator de medidas vê no corpus (e o que falta declarar no perfil)
python -m catalogo_match.caracteristicas --dataset bigdata_profs
python -m catalogo_match.caracteristicas --texto 'DIPIRONA 500 MG/ML 10 ML' --perfil compras_publicas

# Pipeline completo das fases [0]-[9]
python -m catalogo_match.pipeline --dataset bigdata_profs --smoke     # testa a chave (.env)
python -m catalogo_match.pipeline --dataset bigdata_profs             # padrão: LLM na [2] e [6] (lista), ~5 min com cache
python -m catalogo_match.pipeline --dataset bigdata_profs --sem-llm   # sem chave: regex, votação na [6]
python -m catalogo_match.pipeline --dataset bigdata_profs --adjudicacao nenhuma --cross --saida teste   # ablações isoladas

# Rede semântica isolada e serviço HTTP
python -m catalogo_match.rede_semantica --dataset bigdata_profs
uvicorn catalogo_match.servico_rede_semantica:app --reload
```

A primeira execução da grade codifica o corpus com o E5 em CPU (~10 min por
pré-processador neste corpus); as seguintes reaproveitam
`cache/<dataset>/embeddings/` e levam segundos.

**Sem `OPENAI_API_KEY`** tudo que está descrito neste README roda, exceto: a
extração LLM da fase [2] (cai para as regex do perfil), a adjudicação e a
explicabilidade das fases [6] e [8b], e a variante `--graphrag-ms` da expansão.
A chave vai num arquivo `.env` na raiz (`OPENAI_API_KEY=...`), que o `.gitignore`
já exclui. A grade modular do `bigdata_profs` foi medida **sem LLM**; o pipeline
completo, **com** (ver a seção correspondente em
[docs/RESULTADOS-BIGDATA-PROFS.md](docs/RESULTADOS-BIGDATA-PROFS.md)).

---

## 3. Resultados

Os números completos, com a leitura do que cada sinal fez, estão em:

- **[RESULTADOS.md](RESULTADOS.md)** — dataset `mmh`: comparativo regex × LLM,
  blocking por embedding, a grade modular, a separação de características e a
  verificação de que a refatoração multi-dataset não alterou nenhum número.
- **[docs/RESULTADOS-BIGDATA-PROFS.md](docs/RESULTADOS-BIGDATA-PROFS.md)** —
  dataset `bigdata_profs`: o que foi testado para elevar o Recall@3 com GraphRAG,
  taxonomia e rede semântica, o que funcionou, o que não funcionou e por quê.

Resumo do que cada camada rendeu no `bigdata_profs` (1 375 consultas, sem LLM,
sem blocking; `pre = basico`):

| cadeia | MRR | R@1 | R@3 |
|---|---|---|---|
| `e5` (similaridade semântica pura) | 0,669 | 56,2% | 74,3% |
| `e5 + medidas_graphrag` (a melhor combinação do MMH, sem ajuste) | 0,693 | 58,5% | 77,7% |
| `e5 + medidas_graphrag_idf` (GraphRAG ponderado por IDF) | 0,704 | 60,2% | 78,7% |
| `e5_tfidf_char_lin` (fusão linear E5 + TF-IDF palavra + caractere) | 0,743 | 64,4% | 81,9% |
| `e5_tfidf_char_lin + medidas_graphrag_idf` | 0,758 | 66,1% | 83,6% |
| **`e5_tfidf_char_lin + combinado_calibrado`** (+ taxonomia como desempate) | **0,763** | **66,8%** | **84,2%** |

O mesmo par processador + pós-processador, sem ajuste, leva o MMH de R@3 67,5%
para **77,0%**. O que não funcionou (RRF, GraphRAG como pré-processador, KG
como score, sinais com peso alto) está documentado com números no mesmo
documento.

**Pipeline completo** (fases [0]–[9], com blocking; recall do blocking 96,8%),
depois de levar a fusão e os sinais para dentro dele:

| execução | LLM | cross-encoder | fase [6] | MRR | R@1 | R@3 | tempo |
|---|---|---|---|---|---|---|---|
| C `--sem-llm --sem-cross` | não | não | votação TF-IDF | 0,751 | 65,0% | 82,5% | 3 min |
| A | gpt-4o-mini na [2], [6], [8b] | sim | adjudicação LLM, contexto TF-IDF | 0,727 | 62,5% | 80,4% | 51 min |
| B `--sem-cross` | gpt-4o-mini na [2], [6], [8b] | não | adjudicação LLM, índice Microsoft GraphRAG | 0,730 | 64,4% | 79,1% | 2 h 59 min |
| D `--sem-cross --max-adjudicacoes 0` (composição antiga) | gpt-4o-mini na [2] | não | votação TF-IDF | 0,743 | 64,1% | 82,3% | 4 min |
| **Padrão final** (etapa 17): fusão, sinais com ontologia, sem cross, adjudicação em lista p=0,1, confiança por margem | gpt-4o-mini na [2] e [6] | não | LLM em lista, 500 consultas | **0,783** | **69,9%** | **85,2%** | 5 min |
| MMH, padrão final | gpt-4o-mini na [2] e [6] | não | LLM em lista, 500 consultas | **0,677** | **56,6%** | **75,2%** | 9 min |
| MMH, `--sem-llm` | não | não | votação TF-IDF | 0,649 | 52,3% | 73,3% | 2 min |
| MMH publicado (RESULTADOS.md) | gpt-4o-mini | sim | índice Microsoft GraphRAG, par a par | 0,553 | 43,9% | 60,7% | 27 min |

A avaliação do pipeline traz a **ablação por camada**: no `bigdata_profs`, o
mesmo top-20 reordenado só pelo E5 dá R@3 75,1%; pela fusão, 81,9%; com os
sinais da [5b] e os atributos da LLM no KG, 84,0%; com a adjudicação em lista,
**85,2%**. O que foi medido e descartado (fica como opção): cross-encoder
(−1,7 pp, `--cross`), adjudicação de pares isolados (−2,2 pp com contexto
TF-IDF, −4,8 pp com o índice Microsoft, `--adjudicacao par --contexto-ms`),
ontologia como fatia convexa do score final (−1,8 pp, `--peso-ontologia 0.35`),
veto (neutro, `--fator-veto 0.25`). O índice Microsoft como **expansor de
consulta** (`--graphrag-ms` na grade, amostra de 200) empata com a normalização
básica: 85,0% contra 84,5%, uma chamada ao modelo por consulta. A faixa de
confiança **Alta** (margem ≥ 0,05) cobre 34% das consultas com 96% de acerto no
top-1; a **Baixa** (44% das consultas) concentra 78% dos erros.

---

## 4. Estrutura do repositório

```
mmh/
├── README.md                     # este arquivo: arquitetura e uso
├── RESULTADOS.md                 # resultados do dataset mmh (e mmh_opme)
├── requirements.txt
├── catalogo_match/               # o pacote: algoritmo, sem conhecimento de domínio
│   ├── config.py                 # DatasetSpec, PerfilDominio, herança e indução do perfil
│   ├── preprocessamento.py       # normalização de texto, atributos do rótulo CATMAT
│   ├── caracteristicas.py        # medida tipada: valor + unidade convertida
│   ├── rede_semantica.py         # fase [1]: grafo léxico, ativação, ancoragem, SKOS
│   ├── ontologia_owl.py          # fase [4]: TBox OWL + SWRL + Pellet
│   ├── taxonomia.py              # alinhamento das taxonomias e-Fisco <-> CATMAT
│   ├── graphrag.py               # MS GraphRAG (fase [6]) + KG offline da grade
│   ├── pipeline.py               # orquestrador das fases [0]-[9]
│   ├── avaliacao_modular.py      # grade pré × proc × pós, fusão de rankings, ranking
│   ├── diagnostico.py            # onde o R@3 se perde; comparação entre combinações
│   └── servico_rede_semantica.py # serviço HTTP do léxico (FastAPI)
├── config/
│   ├── datasets/                 # mmh.yaml, mmh_opme.yaml, bigdata_profs.yaml
│   └── perfis/                   # base.yaml -> mmh.yaml -> compras_publicas.yaml
├── dados/
│   ├── mmh/                      # ground truth MMH (3 CSVs)
│   └── bigdata_profs/            # ground truth multi-domínio
├── resultados/
│   ├── mmh/  mmh_opme/  bigdata_profs/    # uma pasta por dataset: YAML, matriz, ranking, heatmap
│   ├── <dataset>/etapas/NN_*/             # execuções intermediárias congeladas (CSV, YAML, log)
│   ├── <dataset>/varreduras/              # varreduras de parâmetros (catalogo_match.varredura)
│   └── <dataset>/diagnostico/             # posição do correto por consulta; movimentos entre combinações
├── cache/<dataset>/              # embeddings, textos pré-processados, LLM (não versionado)
└── docs/
    ├── apresentacao_pipeline.ipynb   # a história do trabalho, executável sobre os caches (~2 min, sem API)
    ├── NOVO-DATASET.md           # como rodar em outro corpus ou domínio; herança de perfis
    ├── RESULTADOS-BIGDATA-PROFS.md
    └── historico/                # apresentações de 21/09, anteriores à reorganização (não atualizadas)
```

## 5. Estender

- **Outro corpus, mesmo domínio**: um YAML em `config/datasets/`. Zero código.
- **Domínio vizinho**: um perfil com `herda:` e só a diferença.
- **Domínio novo**: `perfil: base` e o pipeline induz o que puder; depois cure
  `lexico.abreviacoes`/`variantes` (é o que o casamento léxico sente),
  `unidades[*].aliases` (é o que liga a medida tipada) e `atributos`.
- **Módulo novo na grade**: uma função e uma linha no registro
  (`PRE_PROCESSADORES`, `PROCESSADORES`, `POS_PROCESSADORES` em
  `avaliacao_modular.py`). Pré: `fn(corpus, ctx) -> Textos`; proc:
  `fn(textos, top_k, ctx) -> (scores, idx)`; pós:
  `fn(corpus, textos, scores, idx, ctx) -> scores`. Fontes novas para a fusão
  entram em `_FONTES` como `fn(textos, ctx) -> matriz n×m`.

Tudo isso está detalhado em [docs/NOVO-DATASET.md](docs/NOVO-DATASET.md).
