# Resultados do Pipeline MMH — CATMAT ↔ e-Fisco

Pipeline neuro-simbólico para casamento automático de itens de Material Médico Hospitalar (MMH) entre os catálogos CATMAT e e-Fisco/CADMAT do Estado de Pernambuco. Implementa as fases [0]–[9] da Figura 3 do documento técnico.

---

## Comparativo: Regex (execução anterior) vs LLM gpt-4o-mini (execução atual)

### Fase [2] — Extração de atributos + Validação SHACL

A fase de percepção lê o texto livre de cada item e extrai atributos no esquema PDM (calibre, volume, material, esterilidade etc.). Com regex, o extrator só captura padrões explícitos e fixos; com LLM, o modelo lê o contexto completo e infere atributos implícitos ou grafados de forma não-padrão.

A validação de esquema verifica se os **atributos definidores** da família de PDM estão presentes (ex.: `calibre` + `comprimento_valor` para AGULHA). Usa shapes SHACL reais (rdflib + pyshacl), com o mesmo vocabulário da ontologia OWL da fase [4].

| Métrica                            | Regex                | LLM                               | Δ      |
| ---------------------------------- | -------------------- | --------------------------------- | ------ |
| Fonte de extração                  | 2 066 regex / 0 LLM  | **2 066 LLM** / 0 regex / 0 erros | —      |
| Consultas com ≥ 1 atributo         | 914 / 1 012 (90%)    | **1 012 / 1 012 (100%)**          | +10 pp |
| Itens do catálogo com ≥ 1 atributo | 1 054 / 1 054 (100%) | 1 054 / 1 054 (100%)              | —      |
| Esquema SHACL completo (consultas) | 777 / 1 012 (77%)    | **847 / 1 012 (84%)**             | +7 pp  |
| Escore médio de completude         | 0,8696               | **0,9101**                        | +4 pp  |

> A cobertura total de atributos nas consultas passou de 90% para 100%: o LLM
> extrai informações de textos com grafia não-padrão, abreviações e termos
> implícitos que o regex não alcançava.

---

### Fase [6] — GraphRAG (adjudicação na zona cinzenta)

Pares com score entre os limiares Baixo e Alto (zona cinzenta) são adjudicados com contexto do grafo + LLM em vez de votação simples.

| Métrica                  | Regex (sem LLM) | LLM     |
| ------------------------ | --------------- | ------- |
| Pares na zona cinzenta   | 314             | 315     |
| Adjudicações via LLM     | 0               | **197** |
| Adjudicações via votação | 0               | 0       |

> Com LLM desligado, a adjudicação da zona cinzenta ficava sem resolução (nenhuma
> chamada era feita). Agora 197 dos 315 casos difíceis recebem análise contextual
> do grafo + modelo.

---

### Fase [8b] — Explicabilidade em linguagem natural

| Métrica             | Regex (sem LLM) | LLM    |
| ------------------- | --------------- | ------ |
| Explicações geradas | 0               | **60** |

> 60 pares de alta confiança recebem justificativa em linguagem natural,
> auditável pelo gestor ou equipe de curadoria.

---

### Avaliação de recuperação (top-1 e recall@k)

Calculado sobre as 1 012 consultas e-Fisco, comparando o casamento produzido pelo pipeline com o gabarito.

| Métrica                       | Valor (LLM) |
| ----------------------------- | ----------- |
| Recall@1                      | 33,0%       |
| Recall@3                      | 45,9%       |
| Recall@5                      | 53,0%       |
| Recall@10                     | 61,5%       |
| MRR (Mean Reciprocal Rank)    | 0,417       |
| Precisão@1                    | 33,0%       |
| Recall do blocking (fase [3]) | 72,0%       |

> O recall do blocking (72%) representa o teto teórico do pipeline: apenas pares
> gerados como candidatos na fase [3] podem ser encontrados depois. Dentro desse
> teto, o pipeline recupera o item correto no top-1 em 33% dos casos e no top-10
> em 61,5%.

---

### Confiança dos casamentos (top-1 por consulta)

| Faixa               | Quantidade | %   |
| ------------------- | ---------- | --- |
| Alta (score ≥ 0,75) | 254        | 25% |
| Média (0,50–0,75)   | 562        | 56% |
| Baixa (< 0,50)      | 196        | 19% |

---

### Tempos de execução

| Fase                                        | Wall time              |
| ------------------------------------------- | ---------------------- |
| [0] Fontes                                  | 0,7 s                  |
| [1] Rede semântica                          | 22,0 s                 |
| **[2] Extração LLM + SHACL**                | **1 840 s (~31 min)**  |
| [3] Blocking                                | 0,8 s                  |
| [4] Ontologia + Pellet                      | 18,0 s                 |
| **[5] Matcher neural (E5 + cross-encoder)** | **915 s (~15 min)**    |
| **[6] GraphRAG (197 chamadas LLM)**         | **185 s**              |
| [7] Grafo unificado                         | 1,0 s                  |
| [8] Confiança                               | 0,2 s                  |
| [8b] Explicabilidade LLM                    | 57 s                   |
| [9] Análise global                          | 0,5 s                  |
| **Total**                                   | **~3 041 s (~50 min)** |

> Sem LLM (execução anterior com regex), a fase [2] levava < 1 s e o total era
> ~1 370 s (~23 min). O custo adicional da LLM (~27 min) está concentrado na
> extração sequencial de 2 066 textos — pode ser reduzido com paralelismo
> assíncrono (`asyncio`) se necessário.

---

## Comparativo de modelos LLM (três execuções)

> **Modelos testados:** `gpt-4o-mini` (baseline LLM), `gpt-4` (depreciado) e `gpt-4o`.

### Fase [2] — Extração de atributos

| Métrica                    | gpt-4o-mini              | gpt-4                          | gpt-4o            |
| -------------------------- | ------------------------ | ------------------------------ | ----------------- |
| Chamadas LLM bem-sucedidas | 2 066 / 2 066            | 0 / 2 066 (**fallback total**) | 2 066 / 2 066     |
| Consultas com ≥ 1 atributo | **1 012 / 1 012 (100%)** | 914 / 1 012 (90%)              | 394 / 1 012 (39%) |
| Esquema SHACL completo     | **847 / 1 012 (84%)**    | 710 / 1 012 (70%)              | 752 / 1 012 (74%) |
| Escore médio de completude | **0,9101**               | 0,8108                         | 0,7619            |

> O `gpt-4o` fez todas as chamadas sem erro, mas extraiu atributos em apenas **39%** das consultas — resultado pior que até o `gpt-4o-mini`. O modelo mais "capaz" não significa necessariamente melhor aderência ao schema PDM esperado pelo pipeline.

### Avaliação de recuperação

| Métrica    | gpt-4o-mini | gpt-4 | gpt-4o |
| ---------- | ----------- | ----- | ------ |
| Recall@1   | **33,0%**   | 33,0% | 32,3%  |
| Recall@3   | **45,9%**   | 45,2% | 44,9%  |
| Recall@5   | **53,0%**   | 52,6% | 52,0%  |
| Recall@10  | **61,5%**   | 61,7% | 60,5%  |
| MRR        | **0,417**   | 0,416 | 0,410  |
| Precisão@1 | **33,0%**   | 33,0% | 32,3%  |

### Confiança dos casamentos (top-1)

| Faixa             | gpt-4o-mini   | gpt-4     | gpt-4o        |
| ----------------- | ------------- | --------- | ------------- |
| Alta (≥ 0,75)     | **254 (25%)** | 165 (16%) | 140 (14%)     |
| Média (0,50–0,75) | 562 (56%)     | 659 (65%) | **785 (78%)** |
| Baixa (< 0,50)    | 196 (19%)     | 188 (19%) | **87 (9%)**   |

### Tempos de execução

| Fase                 | gpt-4o-mini            | gpt-4              | gpt-4o             |
| -------------------- | ---------------------- | ------------------ | ------------------ |
| [2] Extração         | 1 840 s                | 179 s (regex)      | 1 570 s            |
| [6] GraphRAG         | 185 s                  | 191 s              | 12 s               |
| [8b] Explicabilidade | 57 s                   | 123 s              | 14 s               |
| **Total**            | **~3 041 s (~50 min)** | ~1 413 s (~24 min) | ~2 590 s (~43 min) |

### Conclusão

O `gpt-4o-mini` é o **melhor modelo para este pipeline** nas três dimensões avaliadas: cobertura de atributos (100%), métricas de recuperação e casamentos de alta confiança. O `gpt-4o` extraiu menos atributos que o mini (39% vs 100%), possivelmente por responder de forma mais conservadora ao schema PDM. O `gpt-4` está depreciado e caiu inteiramente para regex na fase [2]. O ganho real do LLM está concentrado na extração (fase [2]) — sem ela, o Recall@1 converge para ~33% independente do modelo.

---

## Melhoria: Blocking semântico por embedding E5 (estratégia 6)

A principal limitação das execuções anteriores era o teto do blocking em **72%**: o pipeline só encontrava pares onde o PDM do e-Fisco coincidia com o PDM do CATMAT (estratégias 1–5). Itens com PDMs distintos mas semanticamente equivalentes ficavam fora do bloco.

### Solução implementada

Uma 6ª estratégia de blocking foi adicionada à fase [3]: para cada consulta e-Fisco, o modelo E5 (`multilingual-e5-base`) computa a similaridade de cosseno com **todos os 1 054 itens** do catálogo CATMAT e adiciona os top-50 ao bloco, independente de PDM. Os embeddings são pré-computados e reutilizados pela fase [5] sem re-encode.

### Resultados (gpt-4o-mini + MS GraphRAG + Embedding Blocking)

| Métrica | Antes (sem embed. blocking) | Depois (com embed. blocking) | Δ |
|---|---|---|---|
| Recall do blocking (teto) | 72,0% | **98,2%** | **+26,2 pp** |
| Candidatos gerados | 39 809 | 72 475 | +82% |
| MRR | 0,416 | **0,553** | +13,7 pp |
| Recall@1 | 32,9% | **43,9%** | **+11 pp** |
| Recall@3 | 45,8% | **60,7%** | **+14,9 pp** |
| Recall@5 | 52,5% | **69,6%** | +17,1 pp |
| Recall@10 | 61,4% | **80,4%** | +19 pp |
| Zona cinza GraphRAG | 313 | 150 | −52% |
| Tempo total | 4 684 s | **1 625 s** | 3× mais rápido* |

*O tempo caiu porque o índice GraphRAG já estava em cache (`.indexado_ok`) — a indexação (20–40 min) só ocorre na primeira execução.

### Conclusão

O blocking semântico por embedding foi a melhoria de maior impacto no pipeline: elevou o teto teórico de 72% para **98,2%** e o Recall@3 de 45,8% para **60,7%** — ganho de ~15 pp em uma única modificação. A zona cinza do GraphRAG reduziu à metade (313 → 150), pois mais pares corretos chegam com score alto e dispensam adjudicação.

---

## Grade modular — pré-processador × processador × pós-processador

O pipeline das fases [0]–[9] executa **uma** configuração fixa. A grade modular
(`avaliacao_modular_mmh.py`) transforma cada etapa em módulo trocável e mede
**todas as combinações** lado a lado, produzindo a matriz e o ranking:

```
Pré-proc 1 (Nada) ┐                              ┌ Pós-proc 1 (Nada) ┐
Pré-proc 2        ├─→ Processador 1..N ─→ matriz ─→ Pós-proc 2       ├─→ Rank das
Pré-proc N        ┘      (pré × proc)            └ Pós-proc N        ┘   combinações
```

Sem blocking: cada consulta é pontuada contra o catálogo CATMAT inteiro, então o
teto é 100% e as células são comparáveis entre si. Os pós-processadores só
reordenam o top-K de cada célula.

### Módulos

| Eixo | ID | O que faz |
|---|---|---|
| Pré | `nada` | texto cru, sem transformação |
| Pré | `basico` | boilerplate jurídico + sinônimos + stopwords |
| Pré | `rede_semantica` | grafo léxico: normalização + propagação de ativação (fase [1]) |
| Pré | `graphrag` | **GraphRAG**: 6 entidades vizinhas no KG + rótulo da comunidade |
| Pré | `graphrag_leve` | **GraphRAG** conservador: 3 entidades, sem rótulo de comunidade |
| Proc | `tfidf` | cosseno sobre TF-IDF de 1-2 gramas |
| Proc | `fuzzy` | `token_set_ratio` (rapidfuzz) |
| Proc | `e5` | bi-encoder assimétrico `multilingual-e5-base` |
| Proc | `e5_cross` | E5 + cross-encoder mMARCO reranqueando o top-5 |
| Pós | `nada` | mantém o ranking do processador |
| Pós | `unidades` | equivalência de valores e unidades de medida (18G, 3 ML, 40 MM) |
| Pós | `pdm` | coerência de família PDM entre candidato e consulta |
| Pós | `graphrag` | **GraphRAG**: cobertura das entidades da consulta pelo candidato, no mesmo KG |
| Pós | `unidades_graphrag` | os dois sinais acima com peso igual |

> O GraphRAG aparece em **dois eixos**, sobre o mesmo KG (entidades de
> PDM/atributos/tokens, busca local por sobreposição, comunidades Louvain):
> como **pré-processador**, reescrevendo a consulta como documento virtual
> expandido antes da similaridade; e como **pós-processador**, reordenando o
> top-K pela cobertura das entidades da consulta. O modo offline não custa API;
> `--graphrag-ms` usa o índice oficial da Microsoft na expansão.
>
> KG construído sobre o corpus: **6 316 nós, 27 665 arestas, 4 250 entidades,
> 31 comunidades**. A expansão acrescenta em média **6,32 termos** por consulta
> e alcança **99%** delas (variante conservadora: 2,96 termos).

### Matriz pré × proc (MRR, 1 012 consultas × 1 054 itens, sem pós-processamento)

| pré \ proc | tfidf | fuzzy | e5 | e5_cross |
|---|---|---|---|---|
| `nada` | 0,4193 | 0,3105 | **0,5234** | 0,4385 |
| `basico` | 0,4765 | 0,4451 | **0,5331** | 0,4112 |
| `rede_semantica` | 0,4613 | 0,4091 | 0,5020 | 0,3978 |
| `graphrag` | 0,4231 | 0,3891 | 0,4632 | 0,3730 |
| `graphrag_leve` | 0,4392 | 0,4182 | 0,4935 | 0,3852 |

Visualização em `resultados/matriz_modular.png`.

### Rank das melhores combinações

| # | combinação | MRR | R@1 | R@3 | R@10 |
|---|---|---|---|---|---|
| 1 | `nada` + `e5` + `unidades_graphrag` | **0,5622** | 43,9% | 65,2% | 81,8% |
| 2 | `basico` + `e5` + `unidades` | 0,5547 | 43,3% | 63,0% | 81,7% |
| 3 | `basico` + `e5` + `unidades_graphrag` | 0,5532 | 42,8% | 63,4% | 81,7% |
| 4 | `nada` + `e5` + `unidades` | 0,5501 | 42,5% | 63,5% | 81,8% |
| 5 | `basico` + `e5` + `nada` | 0,5331 | 41,0% | 60,7% | 81,7% |
| 6 | `nada` + `e5` + `graphrag` | 0,5322 | 40,6% | 61,4% | 81,8% |
| 7 | `rede_semantica` + `e5` + `unidades_graphrag` | 0,5272 | 40,8% | 60,8% | 78,5% |

As 100 combinações completas estão em `resultados/avaliacao_modular.yaml` e
`resultados/ranking_combinacoes.csv`.

### Leitura dos resultados

1. **O processador domina o resultado.** O E5 abre ~0,06 MRR sobre o TF-IDF na
   melhor linha e ~0,21 sobre o fuzzy na pior. A escolha do processador pesa
   mais que a de qualquer pré-processador.

2. **Pós-processadores de atributo ajudam; o de família atrapalha.** `unidades`
   ganha +0,02 a +0,04 MRR em todas as 20 células. O `graphrag` (cobertura de
   entidades) ajuda em **15 das 20**, com o ganho concentrado onde o processador
   é fraco (+0,049 em `nada`+`fuzzy`, +0,027 em `basico`+`fuzzy`) e uma perda
   pequena nas células já fortes (−0,010 em `basico`+`e5`). Os dois somados dão
   o **melhor resultado da grade** (0,5622). Já o `pdm` **piora** (−0,03 no
   melhor caso): a família já está embutida no texto, e o bônus acaba premiando
   candidatos genéricos da família certa.

3. **O mesmo grafo vale mais depois do que antes.** Como *pré*-processador, a
   expansão GraphRAG custa −0,07 MRR frente à normalização básica (0,4632 vs
   0,5331 com E5) e a variante conservadora recupera só parte (0,4935). Como
   *pós*-processador, o mesmo KG rende +0,009 sobre o texto cru e, combinado com
   `unidades`, +0,0075 sobre o melhor resultado sem grafo — de 0,5547 para
   **0,5622**, com R@3 subindo de 63,0% para 65,2%.

   A explicação está nos exemplos de expansão: os termos que a busca local traz
   são vocabulário **de família** — `ESTERIL USO UNICO`, `DESCARTAVEL`,
   `DISPOSITIVO P/ ANESTESIA REGIONAL` — comuns a dezenas de candidatos do mesmo
   PDM. Injetados na consulta, aproximam-na da família certa mas diluem o que
   distingue os itens **dentro** dela, que é o que MRR e R@1 medem. O mesmo
   conhecimento aplicado *depois*, sobre 10 candidatos que já são da família
   certa, não sofre dessa diluição: ali a pergunta é qual candidato cobre os
   atributos da consulta, e é exatamente isso que a cobertura de entidades mede.

4. **O cross-encoder mMARCO degrada o ranking** em todas as linhas (−0,08 a
   −0,12 MRR frente ao E5 puro). Ele foi treinado para relevância de passagem em
   busca web, não para equivalência de itens de catálogo com atributos técnicos.

5. **A melhor combinação não usa LLM nem blocking** e chega a MRR 0,5622 —
   acima do 0,553 do pipeline completo com blocking por embedding e LLM em três
   fases. A comparação tem ressalva (lá o ranking ocorre dentro do bloco de
   candidatos), mas indica que boa parte do ganho do pipeline vem do par
   E5 + sinais de atributo, e não das camadas mais caras.

### Como executar

```bash
python avaliacao_modular_mmh.py                     # grade completa (100 combinações)
python avaliacao_modular_mmh.py --listar            # lista os módulos disponíveis
python avaliacao_modular_mmh.py --amostra 200       # subamostra de consultas (rápido)
python avaliacao_modular_mmh.py --pre nada,graphrag --proc e5 --pos nada,unidades
python avaliacao_modular_mmh.py --metrica recall_at_3   # troca a métrica das células
python avaliacao_modular_mmh.py --graphrag-ms       # expansão via índice Microsoft
python avaliacao_modular_mmh.py --sem-cache         # recalcula os pré-processadores
```

Tudo que é caro fica em `cache_embeddings/`: textos dos pré-processadores,
embeddings E5 e scores do cross-encoder. A primeira execução completa leva ~1 h
em CPU de 2 núcleos; as seguintes, ~30 s.

**Reprodutibilidade.** Duas fontes de variação foram fechadas: os desempates da
expansão GraphRAG passaram a ser ordenados por nome de entidade (antes herdavam
a ordem de iteração de `set`, que muda com o hash seed do processo), e os textos
dos pré-processadores são cacheados — necessário porque a construção do grafo
léxico da fase [1] tem passo estocástico. Com o cache limpo, a linha
`rede_semantica` da matriz oscila até ~0,01 MRR entre execuções; as demais são
exatamente reprodutíveis.

A assinatura do cache inclui o fonte de `preprocessamento_mmh.py`,
`rede_semantica_mmh.py` e `graphrag_mmh.py`: mexer em qualquer um deles
invalida os textos cacheados por segurança — inclusive edições inócuas, como um
comentário. É o preço de não usar cache velho depois de uma mudança de lógica.

### Adicionando um módulo

Cada eixo é um registro `{id: Modulo}` no topo do arquivo. Para acrescentar um
pré-processador, escreva `fn(corpus, ctx) -> Textos` e registre em
`PRE_PROCESSADORES`; processadores implementam `fn(textos, top_k, ctx) ->
(scores, idx)` e pós-processadores `fn(corpus, textos, scores, idx, ctx) ->
scores`. Módulos que falharem (dependência ausente) deixam a célula como `n/d`
sem derrubar a grade.

---

## Arquivos gerados

Todos os entregáveis ficam em `resultados/`:

| Arquivo                           | Descrição                                                |
| --------------------------------- | -------------------------------------------------------- |
| `resultado_pipeline_completo.csv` | Pares ranqueados com scores de todas as fases            |
| `stats_pipeline.json`             | Métricas detalhadas por fase + tempos de execução        |
| `resultado_pipeline.yaml`         | Amostras por faixa de confiança (Alta/Média/Baixa)       |
| `grafo_unificado.graphml`         | Grafo com itens, PDMs e arestas `:equivalenteA`          |
| `ontologia_mmh.owl`               | Ontologia OWL gerada pelo pipeline (TBox + ABox parcial) |
| `rede_semantica.graphml`          | Grafo léxico de domínio (5 009 nós, 48 996 arestas)      |
| `rede_semantica_skos.ttl`         | Vista SKOS/RDF do léxico (62 656 triplas)                |
| `fila_curadoria.csv`              | 200 arestas priorizadas para revisão humana              |
| `analise_global.png`              | Visualização das comunidades e anomalias                 |
| `rede_semantica_agulha.png`       | Vizinhança da família AGULHA no grafo léxico             |
| `rede_semantica_geral.png`        | Visão geral da rede semântica (top 80 nós)               |
| `avaliacao_modular.yaml`          | Grade modular: matriz, ranking das 100 combinações, módulos |
| `matriz_pre_x_proc.csv`           | Matriz pré-processador × processador (MRR)               |
| `ranking_combinacoes.csv`         | Combinações ordenadas por métrica, com tempos            |
| `matriz_modular.png`              | Heatmap da matriz pré × proc                             |

---

## Estrutura do repositório

```
mmh/
├── pipeline_completo_mmh.py      # orquestrador principal (fases [0]–[9])
├── avaliacao_modular_mmh.py      # grade pré × proc × pós + ranking de combinações
├── graphrag_mmh.py               # fase [6] (adjudicação) + pré-proc de expansão
├── rede_semantica_mmh.py         # fase [1]: grafo léxico + expansão
├── ontologia_owl_mmh.py          # fase [4]: OWL + reasoner Pellet (SWRL)
├── preprocessamento_mmh.py       # utilitários de normalização de texto
├── servico_rede_semantica.py     # serviço HTTP FastAPI (§3.8)
├── requirements.txt
├── dados/                        # entradas (ground truth)
│   ├── 20260408_ground_truth_mmh_limpa.csv
│   ├── 20260408_ground_truth_mmh_test.csv
│   └── 20260408_ground_truth_mmh_opme_test.csv
├── resultados/                   # saídas geradas pelo pipeline
│   ├── resultado_pipeline_completo.csv
│   ├── stats_pipeline.json
│   ├── resultado_pipeline.yaml
│   ├── grafo_unificado.graphml
│   ├── ontologia_mmh.owl
│   ├── rede_semantica.graphml
│   ├── rede_semantica_skos.ttl
│   ├── fila_curadoria.csv
│   ├── analise_global.png
│   ├── rede_semantica_agulha.png
│   └── rede_semantica_geral.png
└── docs/                         # apresentações e documentação
    ├── apresentacao_pipeline.html
    ├── apresentacao_pre_processamento_mmh.ipynb
    └── resumo_mudancas_reuniao.yaml
```

## Como executar

```bash
# Instalar dependências (Java no PATH para o Pellet)
pip install -r requirements.txt

# Execução completa com LLM (requer OPENAI_API_KEY no .env)
python pipeline_completo_mmh.py

# Execução offline (regex como fallback, sem custo de API)
python pipeline_completo_mmh.py --sem-llm

# Testar conexão com a API antes de rodar
python pipeline_completo_mmh.py --smoke
```
