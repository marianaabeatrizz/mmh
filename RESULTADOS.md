# Resultados do Pipeline — CATMAT ↔ e-Fisco

Pipeline neuro-simbólico para casamento automático de itens entre os catálogos CATMAT e e-Fisco/CADMAT do Estado de Pernambuco. Implementa as fases [0]–[9] da Figura 3 do documento técnico.

Os números deste documento são do dataset **`mmh`** (Material Médico Hospitalar), que segue sendo o dataset padrão. O pipeline é agnóstico de domínio: o conhecimento de domínio vive em `config/perfis/`, o corpus em `config/datasets/`, e nenhuma fase tem vocabulário hospitalar embutido — ver **[docs/NOVO-DATASET.md](docs/NOVO-DATASET.md)**.

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
(`catalogo_match/avaliacao_modular.py`) transforma cada etapa em módulo trocável e mede
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
python -m catalogo_match.avaliacao_modular               # grade completa (100 combinações)
python -m catalogo_match.avaliacao_modular --listar      # lista os módulos disponíveis
python -m catalogo_match.avaliacao_modular --amostra 200 # subamostra de consultas (rápido)
python -m catalogo_match.avaliacao_modular --pre nada,graphrag --proc e5 --pos nada,unidades
python -m catalogo_match.avaliacao_modular --metrica recall_at_3  # troca a métrica
python -m catalogo_match.avaliacao_modular --graphrag-ms # expansão via índice Microsoft
python -m catalogo_match.avaliacao_modular --sem-cache   # recalcula os pré-processadores

# Outro corpus, ou o mesmo corpus sem nenhum conhecimento curado
python -m catalogo_match.avaliacao_modular --dataset mmh_opme
python -m catalogo_match.avaliacao_modular --perfil base
```

Tudo que é caro fica em `cache/<dataset>/embeddings/`: textos dos
pré-processadores, embeddings E5 e scores do cross-encoder. A primeira execução completa leva ~1 h
em CPU de 2 núcleos; as seguintes, ~30 s.

**Reprodutibilidade.** Duas fontes de variação foram fechadas: os desempates da
expansão GraphRAG passaram a ser ordenados por nome de entidade (antes herdavam
a ordem de iteração de `set`, que muda com o hash seed do processo), e os textos
dos pré-processadores são cacheados — necessário porque a construção do grafo
léxico da fase [1] tem passo estocástico. Com o cache limpo, a linha
`rede_semantica` da matriz oscila até ~0,01 MRR entre execuções; as demais são
exatamente reprodutíveis.

A assinatura do cache inclui o fonte de `preprocessamento.py`,
`rede_semantica.py`, `graphrag.py` e `config.py`, **mais o YAML do perfil em
uso**: mexer em qualquer um deles invalida os textos cacheados por segurança —
inclusive edições inócuas, como um comentário. É o preço de não usar cache velho
depois de uma mudança de lógica, e de não comparar dois domínios com os textos de
um só.

### Adicionando um módulo

Cada eixo é um registro `{id: Modulo}` no topo do arquivo. Para acrescentar um
pré-processador, escreva `fn(corpus, ctx) -> Textos` e registre em
`PRE_PROCESSADORES`; processadores implementam `fn(textos, top_k, ctx) ->
(scores, idx)` e pós-processadores `fn(corpus, textos, scores, idx, ctx) ->
scores`. Módulos que falharem (dependência ausente) deixam a célula como `n/d`
sem derrubar a grade.

---

## Multi-dataset: o domínio como dado

Até a refatoração, o conhecimento de domínio estava espalhado por seis módulos:
as 13 regex de boilerplate jurídico no pré-processamento, as abreviações e a
tabela de unidades na rede semântica, as 15 famílias com características
definidoras na ontologia, a lista de famílias do blocking, o esquema de atributos
nos quatro prompts da LLM. Trocar de dataset exigia editar código em seis lugares
— e, na prática, significava não trocar.

Agora são duas configurações e nenhuma linha de código:

| Arquivo | Declara | Muda quando |
|---|---|---|
| `config/datasets/<nome>.yaml` | arquivos, separador, encoding, saídas | o **corpus** é outro |
| `config/perfis/<nome>.yaml` | léxico, famílias, unidades, esquema de atributos, papéis dos prompts | o **domínio** é outro |

A separação tem consequência prática: `mmh` e `mmh_opme` são dois corpora do
mesmo domínio e compartilham o perfil `mmh`. Cada dataset tem saída, cache e
índice GraphRAG próprios (`resultados/<dataset>/`, `cache/<dataset>/`,
`graphrag_workspace/<dataset>/`) — um índice de outro corpus reaproveitado em
silêncio daria resultado errado com cara de certo.

Sem perfil, o pipeline **induz** do corpus o que puder: esquema de atributos,
famílias, definidoras, hierarquia, stopwords, boilerplate e unidades. O que não
dá para induzir com honestidade fica de fora e é registrado como lacuna — fator
de conversão de unidade e cláusula legal esparsa, principalmente. A proveniência
bloco a bloco (`curado`, `induzido`, `vazio`) sai no bloco `PERFIL` do YAML de
resultados. Detalhes e limiares em **[docs/NOVO-DATASET.md](docs/NOVO-DATASET.md)**.

### Quanto vale o conhecimento curado

Mesmo corpus (MMH, 150 consultas), perfil `mmh` contra perfil `base` com tudo
induzido. É a medida direta do que a curadoria humana acrescenta:

| combinação | curado | induzido | Δ |
|---|---|---|---|
| `basico` + `fuzzy` + `unidades` | 0,5155 | 0,4312 | **+0,0843** |
| `basico` + `fuzzy` + `nada` | 0,4643 | 0,3828 | +0,0815 |
| `basico` + `fuzzy` + `pdm` | 0,4609 | 0,4065 | +0,0544 |
| `rede_semantica` + `fuzzy` + `unidades` | 0,4874 | 0,4462 | +0,0412 |
| `basico` + `tfidf` + `pdm` | 0,4158 | 0,4031 | +0,0127 |
| `basico` + `tfidf` + `nada` | 0,4763 | 0,4678 | +0,0085 |
| `basico` + `tfidf` + `unidades` | **0,5189** | **0,5144** | +0,0045 |
| `nada` + `tfidf` + `nada` | 0,3559 | 0,3559 | 0,0000 |

Três leituras:

1. **O casamento léxico é que depende do léxico.** O `fuzzy` perde ~0,08 MRR sem
   o perfil curado: é ele que faz `QUICKLE` e `QUINCKE` serem a mesma coisa, e
   `INOX` e `ACO INOXIDAVEL` também. O TF-IDF quase não percebe (+0,0045),
   porque distribui peso por muitos termos e não depende de um casar exato.
2. **A indução cobre mais do que parece.** A melhor célula sem nenhum
   conhecimento humano (0,5144) fica a 0,0045 da melhor com perfil curado
   (0,5189). Para um domínio novo, começar sem perfil é uma linha de base
   defensável, não um placeholder.
3. **As células `nada` são idênticas nos dois perfis**, o que era o esperado: o
   texto cru não passa pelo perfil. Serve de controle do próprio experimento.

### O MMH não mudou

A refatoração não devia alterar nenhum número publicado aqui, e isso foi
verificado, não presumido:

| O que | Como | Resultado |
|---|---|---|
| Grafo léxico (fase [1]) | hash dos nós e das arestas, indução desligada | idêntico |
| Normalização e ativação | `normalizar_termo` e `expandir_por_ativacao` em 12 termos | idêntico |
| Textos pré-processados | hash de 40 registros dos dois lados | idêntico |
| Extração por regex (fase [2]) | hash da extração sobre 400 textos reais | idêntico |
| Canonicalização do JSON da LLM | 4 casos, inclusive valor inválido | idêntico |
| Shapes SHACL (fase [2]) | isomorfismo de grafo RDF (140 triplas) | isomórfico |
| Famílias do blocking (fase [3]) | lista completa | idêntica |
| Grade modular | 24 combinações, 150 consultas | 16 idênticas, 8 na linha `rede_semantica` |

As 8 diferenças são todas da linha `rede_semantica`, cujo passo de indução por
fastText é estocástico — o mesmo código, rodado duas vezes, varia até 0,012 MRR
nessa amostra, e chega a ~0,02 entre execuções mais distantes. Com a indução
desligada, o grafo sai byte a byte igual, o que localiza a variação no fastText e
não na refatoração. Vale como correção ao que este documento dizia antes: a
oscilação da linha `rede_semantica` é maior que os "~0,01 MRR" registrados na
seção da grade modular, ao menos em subamostra de 150 consultas.

Duas diferenças deliberadas, registradas para quem comparar execuções antigas:

- **O prompt da fase [2] mudou de formatação.** As chaves `volume_valor` e
  `volume_unidade` estavam na mesma linha e agora saem em duas, porque o bloco de
  esquema é gerado a partir do perfil. O conteúdo é o mesmo; o cache de extração é
  indexado pelo texto do item, não pelo prompt, então nada é invalidado.
- **`familias` não é a união automática** com as chaves de
  `atributos_definidores`. Três famílias com definidoras (ESPARADRAPO, FRASCO,
  BOLSA) nunca estiveram na lista do blocking, e uni-las mudaria o recall do
  blocking sem que ninguém tivesse pedido. O perfil deixa a escolha visível em
  vez de herdada.

---

## Separação de características: o que elevou o R@3

O R@3 era o alvo: 65,2% na melhor combinação da grade, com R@10 em 81,8%. A
distância entre os dois diz onde estava o trabalho — em ~17% das consultas o item
correto já estava entre os dez primeiros e não chegava aos três.

### Onde o R@3 se perdia

Diagnóstico das 1 012 consultas com `nada + e5`, olhando as 217 (21,4%) cujo
correto caía entre a 4ª e a 10ª posição:

| achado | % das 217 falhas |
|---|---|
| os três intrusos têm **o mesmo PDM exato** do correto | 67,3% |
| cobertura de atributos **empata** entre correto e intrusos | 43,8% |
| cobertura de atributos **favorece o intruso** | 30,4% |
| cobertura de atributos favorece o correto | 25,8% |
| a consulta contém negação ("sem", "isento de") | 25,3% |

O problema, portanto, não é achar a família: é discriminar **dentro** dela. E o
sinal de cobertura de atributos — que é o que o pós-processador `graphrag` já
media — não separa. Um caso típico:

```
CONSULTA : AGULHA PERIDURAL ... TAMANHO G16 X 3 1/4"
CORRETO  : ... DIÂMETRO AGULHA: 16 G , COMPRIMENTO: CERCA DE 3" - 80 MM
TOP-1    : ... DIMENSÃO: 18 G X 1 1/2"        ← cobertura quase igual, calibre errado
```

O que decide é o **valor numérico por categoria**, e era exatamente o que o
pipeline não comparava: o pós-processador `unidades` casa o par `(valor, unidade)`
como texto, então `3 1/2"` nunca encontrava `90 MM`, e `G16` não era reconhecido
porque a regex exigia o dígito antes do `G`.

### O sinal que funcionou

`catalogo_match/caracteristicas.py` extrai **medida tipada** dos dois lados —
valor mais unidade, convertidos para a unidade base da categoria — e o
pós-processador premia o candidato que satisfaz as medidas enunciadas pela
consulta. Três decisões de implementação, todas vindas de medição:

1. **Conversão antes da comparação.** `3 1/2"` → 88,9 mm encontra `90 MM` dentro
   da tolerância de 8%, que existe porque o catálogo escreve "CERCA DE". Gauge é
   lido nas duas ordens (`18G` e `G16`), e a base de comparação é a unidade
   canônica, não a categoria: 7 FR e 7 G são ambos "calibre" e não são a mesma
   coisa.
2. **Ancorado na consulta.** O denominador é o número de categorias que a
   *consulta* enuncia. Normalizar pelo candidato pune o item mais específico — e
   o mais específico é o certo.
3. **Somado, não misturado.** Ver a próxima subseção.

### Resultado

Grade de 24 combinações, 1 012 consultas × 1 054 itens, sem blocking:

| combinação | MRR | R@1 | R@3 | R@10 |
|---|---|---|---|---|
| **`basico` + `e5` + `medidas_graphrag`** | **0,6035** | **48,3%** | **67,8%** | 82,9% |
| `nada` + `e5` + `medidas_graphrag` | 0,5987 | 47,8% | 67,5% | 82,9% |
| `rede_semantica` + `e5` + `medidas_graphrag` | 0,5855 | 46,3% | 66,7% | 81,4% |
| `nada` + `e5` + `medidas` | 0,5713 | 44,1% | 65,6% | 82,8% |
| `nada` + `e5` + `unidades_graphrag` *(melhor anterior)* | 0,5564 | 42,4% | 64,4% | 82,6% |
| `nada` + `e5` + `nada` | 0,5234 | 39,6% | 60,4% | 81,8% |

Contra a melhor combinação publicada antes (`nada + e5 + unidades_graphrag`, MRR
0,5622 / R@1 43,9% / R@3 65,2%): **R@3 +2,6 pp, R@1 +4,4 pp, MRR +4,1 pp**.
O sinal `medidas` **sozinho** já supera a melhor combinação anterior.

Generalização, medida em dois eixos:

| | R@3 antes | R@3 depois | Δ |
|---|---|---|---|
| `mmh`, perfil curado | 0,6443 | 0,6749 | +3,1 pp |
| `mmh`, **perfil `base`** (domínio induzido) | 0,6294 | 0,6601 | +3,1 pp |
| `mmh_opme` (outro corpus) | 0,7581 | 0,7719 | +1,4 pp |

O ganho não depende da curadoria: com perfil `base`, que só conhece as unidades
do SI, ele é o mesmo — e fica acima do melhor resultado que o perfil curado
alcançava sem o sinal.

### Três variantes reprovadas, e o que elas ensinam

O caminho óbvio era comparar **característica categórica** ("cor: branca",
"tamanho: médio"), inclusive induzindo o léxico valor→chave do próprio catálogo,
que já vem com `COR: BRANCA`. Foi medido nas mesmas 217 falhas, e todas as
variantes perderam para o acaso:

| variante | favorece o correto | favorece o intruso |
|---|---|---|
| cobertura de todos os atributos do candidato | 25,6% | 28,1% |
| só as chaves em que os candidatos divergem | 23,1% | 27,3% |
| idem, com peso por variabilidade na família | 17,4% | 29,8% |
| cobertura ancorada na consulta, com IDF nos candidatos | 18,4% | 24,9% |
| **medida tipada** | **21,2%** | **4,1%** |

A razão é sempre a mesma: a distribuição de características é dominada por
vocabulário de família — `DESCARTAVEL`, `ESTERIL`, `EMBALAGEM INDIVIDUAL` —, que
todo candidato do bloco satisfaz. Pesar por IDF no conjunto de candidatos não
resolve, porque os intrusos que chegaram ao top-3 são lexicalmente parecidos com
a consulta: é por isso que chegaram. O conjunto é adversarial por construção.

Vale a comparação com a leitura já registrada na seção da grade modular, sobre o
GraphRAG como pré-processador diluir a consulta com vocabulário de família. É o
mesmo mecanismo, medido agora de outro ângulo.

Misturar o categórico ao sinal de medida **piora o sinal bom**: a razão
acerto:erro cai de 5,1:1 para 1,4:1. Por isso o módulo entrega só a medida.

**Duas ressalvas honestas.** A primeira: isso foi medido *neste* corpus, onde a
característica categórica é quase toda boilerplate de família. Num catálogo de
alimentos, "branco" e "médio" podem ser justamente o que discrimina — a variante
fica no repositório (`medidas_conflito` e as funções de `caracteristicas.py`) para
ser remedida em corpus de outro tipo, e não como veredito universal. A segunda: o
extrator só vê o que o perfil declara. `50 G` de um ovo não é lido no perfil
`mmh`, porque ali massa não tem `aliases` e a faixa de gauge é 5–34; um domínio de
alimentos precisa declarar massa.

### A forma de combinar importa tanto quanto o sinal

A primeira implementação penalizava conflito de medida, somando ±peso ao score.
Ela **piorou** o R@3 (0,601 contra 0,604 sem pós-processamento), apesar de o
diagnóstico dizer 46 acertos contra 9. O erro estava no diagnóstico, não no
código: ele olhava só as falhas e nunca media o dano nas 401 consultas que já
acertavam no top-1, onde penalizar conflito derruba acerto. Premiar acordo
funciona; punir divergência, neste corpus, não.

A segunda escolha foi a escala. Na mistura convexa dos outros pós-processadores,
peso 0,25 e peso 0,60 dão **exatamente** o mesmo resultado — sinal de que o bônus
já domina a similaridade e o ranking virou "ordena por bônus, desempata por E5".
Somado com peso 0,05, o sinal fica na escala do espalhamento do E5 (~0,1 entre o
1º e o 10º colocado) e refina em vez de substituir:

| forma | peso | R@3 | MRR |
|---|---|---|---|
| convexa | 0,25 e 0,60 (idênticos) | 0,6601 | 0,5747 |
| aditiva | 0,02 | 0,6700 | 0,5944 |
| **aditiva** | **0,05** | **0,6729** | 0,5858 |
| aditiva | 0,20 | 0,6621 | 0,5763 |

Profundidade de reordenação: `top_k` maior eleva o R@10 (0,8399 em 50) e **abaixa**
o R@3 (0,6275 em 50; 0,6018 em 100), porque um conjunto mais largo dá mais chance
a item que casa a medida e erra o resto. O padrão de 20 candidatos ficou.

### Uma pista não explorada

A medida também serve de **índice de recuperação**, e não só de reordenação. Nas
184 consultas cujo correto não entra no top-10, o conjunto dos itens que satisfazem
*todas* as medidas da consulta contém o correto em 34 casos (18,5%), com mediana de
**3 itens**. Injetar esses candidatos elevaria o teto em +3,4 pp de R@10 — o que
exige mexer na recuperação (fase [3]), não no pós-processamento, e não foi feito.

### Como reproduzir

```bash
python -m catalogo_match.avaliacao_modular \
    --pre nada,basico --proc e5 \
    --pos nada,unidades_graphrag,medidas,medidas_graphrag \
    --metrica recall_at_3

# o que o extrator vê no corpus (e o que falta declarar no perfil)
python -m catalogo_match.caracteristicas --dataset mmh
python -m catalogo_match.caracteristicas --texto 'AGULHA G16 X 3 1/4"'
```

> A tabela de unidades do perfil `mmh` ganhou **POL** (polegada, fator 25,4 para
> MM), que faltava e é como o catálogo descreve comprimento de agulha. Isso
> acrescenta um nó `Unidade` ao grafo léxico da fase [1], então a linha
> `rede_semantica` da matriz muda por causa da tabela, e não por mudança de
> lógica.

---

## Arquivos gerados

Todos os entregáveis ficam em `resultados/<dataset>/` — `resultados/mmh/` para os
números deste documento. Uma execução `--model <modelo>` isola a saída um nível
abaixo, em `resultados/<dataset>/<modelo>/`.

| Arquivo                           | Descrição                                                |
| --------------------------------- | -------------------------------------------------------- |
| `resultado_pipeline_completo.csv` | Pares ranqueados com scores de todas as fases            |
| `stats_pipeline.json`             | Métricas detalhadas por fase + tempos de execução        |
| `resultado_pipeline.yaml`         | Amostras por faixa de confiança (Alta/Média/Baixa)       |
| `grafo_unificado.graphml`         | Grafo com itens, PDMs e arestas `:equivalenteA`          |
| `ontologia.owl`                   | Ontologia OWL gerada pelo pipeline (TBox + ABox parcial) |
| `rede_semantica.graphml`          | Grafo léxico de domínio (5 009 nós, 48 996 arestas)      |
| `rede_semantica_skos.ttl`         | Vista SKOS/RDF do léxico (62 656 triplas)                |
| `fila_curadoria.csv`              | 200 arestas priorizadas para revisão humana              |
| `analise_global.png`              | Visualização das comunidades e anomalias                 |
| `rede_semantica_<pdm>.png`        | Vizinhança do PDM em foco (do perfil, ou o mais frequente) |
| `rede_semantica_geral.png`        | Visão geral da rede semântica (top 80 nós)               |
| `avaliacao_modular.yaml`          | Grade modular: matriz, ranking das 100 combinações, módulos |
| `matriz_pre_x_proc.csv`           | Matriz pré-processador × processador (MRR)               |
| `ranking_combinacoes.csv`         | Combinações ordenadas por métrica, com tempos            |
| `matriz_modular.png`              | Heatmap da matriz pré × proc                             |

---

## Estrutura do repositório

```
mmh/
├── catalogo_match/               # o pacote: algoritmo, sem conhecimento de domínio
│   ├── config.py                 # DatasetSpec + PerfilDominio + indução do perfil
│   ├── pipeline.py               # orquestrador principal (fases [0]–[9])
│   ├── avaliacao_modular.py      # grade pré × proc × pós + ranking de combinações
│   ├── graphrag.py               # fase [6] (adjudicação) + pré-proc de expansão
│   ├── rede_semantica.py         # fase [1]: grafo léxico + expansão
│   ├── ontologia_owl.py          # fase [4]: OWL + reasoner Pellet (SWRL)
│   ├── preprocessamento.py       # utilitários de normalização de texto
│   └── servico_rede_semantica.py # serviço HTTP FastAPI (§3.8)
├── config/                       # o domínio e o corpus, como dado
│   ├── datasets/mmh.yaml         # arquivos, separador, encoding, saídas
│   ├── datasets/mmh_opme.yaml    # outro corpus, mesmo perfil
│   ├── perfis/base.yaml          # neutro: português, SI, formato do rótulo CATMAT
│   └── perfis/mmh.yaml           # léxico, famílias, unidades, atributos, prompts
├── requirements.txt
├── dados/mmh/                    # entradas (ground truth), uma pasta por dataset
│   ├── 20260408_ground_truth_mmh_limpa.csv
│   ├── 20260408_ground_truth_mmh_test.csv
│   └── 20260408_ground_truth_mmh_opme_test.csv
├── resultados/mmh/               # saídas do pipeline, uma pasta por dataset
│   ├── resultado_pipeline_completo.csv
│   ├── stats_pipeline.json
│   ├── resultado_pipeline.yaml
│   ├── grafo_unificado.graphml
│   ├── ontologia.owl
│   ├── rede_semantica.graphml
│   ├── rede_semantica_skos.ttl
│   ├── fila_curadoria.csv
│   ├── analise_global.png
│   ├── rede_semantica_agulha_puncao_ossea.png
│   └── rede_semantica_geral.png
├── resultados/mmh_opme/          # o mesmo, para o recorte OPME
├── cache/<dataset>/              # caches de LLM, embeddings e expansão (não versionado)
└── docs/
    ├── NOVO-DATASET.md           # como rodar em outro corpus ou domínio
    ├── apresentacao_pipeline.html
    └── apresentacao_pre_processamento_mmh.ipynb
```


## Como executar

```bash
# Instalar dependências (Java no PATH para o Pellet)
pip install -r requirements.txt

# Execução completa com LLM (requer OPENAI_API_KEY no .env)
python -m catalogo_match.pipeline

# Execução offline (regex do perfil como fallback, sem custo de API)
python -m catalogo_match.pipeline --sem-llm

# Testar conexão com a API antes de rodar
python -m catalogo_match.pipeline --smoke

# Outro corpus, ou o mesmo corpus sem conhecimento curado
python -m catalogo_match.pipeline --dataset mmh_opme
python -m catalogo_match.pipeline --perfil base
```
