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

---

## Estrutura do repositório

```
mmh/
├── pipeline_completo_mmh.py      # orquestrador principal (fases [0]–[9])
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
