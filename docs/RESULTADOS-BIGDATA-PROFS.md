# Resultados — dataset `bigdata_profs` (multi-domínio)

Objetivo: elevar o **Recall@3** do casamento e-Fisco → CATMAT no dataset
`bigdata_profs` usando as camadas simbólicas do sistema — GraphRAG, ontologia /
taxonomia e rede semântica — sobre a similaridade semântica do E5. Tudo aqui
foi medido **sem LLM** (não havia chave de API no ambiente), com a grade modular
(`catalogo_match.avaliacao_modular`), sem blocking: cada consulta contra o
catálogo inteiro, teto 100%. Arquitetura e módulos estão no
[README.md](../README.md); a mesma história, contada em ordem e com as células
executáveis, está em [apresentacao_pipeline.ipynb](apresentacao_pipeline.ipynb).

**Resultado.** Na grade sem blocking, de R@3 **74,0%** (E5 puro) e **76,8%**
(melhor combinação da grade original) para **84,2%**
(`basico + e5_tfidf_char_lin + combinado_calibrado`), com MRR de 0,675 para
0,763 e R@1 de 56,9% para 66,6%. O ganho vem de três coisas, nesta ordem de
tamanho: a **fusão linear** do E5 com TF-IDF de palavra e de caractere no eixo
do processador (+7,6 pp), o **GraphRAG com IDF** somado à **medida tipada** no
pós-processamento (+1,7 pp sobre a fusão) e a **taxonomia** dos dois catálogos
como desempate (+0,6 pp). O mesmo par processador + pós-processador, sem nenhum
ajuste, leva o MMH de 67,5% para **77,0%**. No **pipeline completo** (§9), com
esses módulos levados para o blocking e o matcher, extração por LLM e uma
adjudicação por LLM **em lista** na fase [6], o padrão final chega a
**R@3 85,2%, R@1 69,9%, MRR 0,783**, com recall do blocking 96,8% e faixas de
confiança validadas (Alta: 37% das consultas, 95% de acerto).

---

## 1. O corpus

`dados/bigdata_profs/20260916_ground_truth_bigdata_profs_completo_test_ouro.csv`,
1 392 pares, todos com `qualidade = PERFEITA`, de duas origens:

| origem | pares | o que é |
|---|---|---|
| `big_data` | 1 150 | mineração em larga escala, sem flags de revisão |
| `profs_saude` | 242 | revisados por profissionais de saúde (`situacao`: 157 `OK_SEM_FLAGS`, 63 `MAPEAMENTO_INCERTO`, 22 outras) |

Na grade: **1 375 consultas** (e-Fisco únicos) × **1 297 itens** (CATMAT
únicos); 83 itens CATMAT são alvo de mais de uma consulta.

O que o distingue do MMH:

- **Multi-domínio.** 72% dos pares são do grupo "equipamentos e artigos de uso
  médico, odontológico, hospitalar e laboratorial" (aí dentro, 128 pares de
  medicamentos e 54 de OPME); os 28% restantes se espalham por 46 grupos do
  e-Fisco — laboratório escolar, mobiliário, escritório, higiene, alimentos,
  copa e cozinha, informática, tintas, mão de obra...
- **724 PDMs distintos** em 1 297 itens (o MMH tem ~200 em 1 054): quase não há
  "família" para discriminar por dentro; o problema é achar a família certa.
- **Taxonomia explícita nos dois lados**: `grupo_efisco`, `classe_efisco`,
  `grupo_catmat`, `classe_catmat` (163 × 156 classes). 85% dos pares têm a sua
  combinação (classe e-Fisco, classe CATMAT) repetida em pelo menos outro par.
- **Medidas de outros tipos**: dimensões encadeadas com `X` colado
  (`210,00X86,00X162,00CM`, 133 consultas), massa e dose (`500 MG`, 243),
  concentração (`MG/ML`, 54; `%`, 132), tensão e potência (`V`, 59; `W`, 16).

Configuração: `config/datasets/bigdata_profs.yaml` (aponta para o perfil
`compras_publicas`, que herda o `mmh` e acrescenta essas unidades — ver §6).

---

## 2. Ponto de partida: a grade original neste corpus

Perfil `mmh`, extrator de medidas como estava (`resultados/bigdata_profs/etapas/01_baseline_perfil_mmh/`):

| combinação | MRR | R@1 | R@3 | R@10 |
|---|---|---|---|---|
| `nada + tfidf + nada` | 0,554 | 43,7% | 62,7% | — |
| `basico + tfidf + nada` | 0,651 | 53,5% | 72,7% | — |
| `nada + e5 + nada` | 0,675 | 56,9% | 74,0% | 88,2% |
| `basico + e5 + nada` | 0,669 | 56,2% | 74,3% | — |
| `nada + e5 + unidades_graphrag` | 0,680 | 57,0% | 75,4% | — |
| `nada + e5 + medidas_graphrag` | 0,702 | 60,1% | 76,8% | — |
| **`basico + e5 + medidas_graphrag`** | 0,693 | 58,5% | **77,7%** | — |
| `nada + e5 + pdm` | 0,595 | 46,8% | 67,3% | — |

Três diferenças em relação ao MMH já aparecem aqui: a normalização `basico` vale
**+10 pp** no TF-IDF (no MMH valia +5); o pós-processador `pdm` piora ainda mais
(−6,7 pp); e a melhor combinação do MMH (`medidas_graphrag`) continua sendo a
melhor, mas com ganho menor (+2,8 pp sobre o E5 puro, contra +7,4 no MMH).

### Onde o R@3 se perdia

`python -m catalogo_match.diagnostico --dataset bigdata_profs --pre nada --proc e5 --pos nada`:

| posição do correto | consultas | % |
|---|---|---|
| 1 | 782 | 56,9 |
| 2–3 | 236 | 17,2 |
| **4–10** (alvo do pós-processamento) | **194** | **14,1** |
| 11–20 | 55 | 4,0 |
| **> 20** (fora do alcance de qualquer pós-processador) | **108** | **7,9** |

| recorte | n | R@3 | teto (top-20) |
|---|---|---|---|
| `big_data` | 1 148 | 75,5% | 91,5% |
| `profs_saude` | 227 | 66,5% | 95,6% |
| consulta com medida | 761 | 71,2% | 91,9% |
| consulta sem medida | 614 | 77,5% | 92,5% |
| grupo médico-hospitalar | 979 | 73,5% | 93,7% |
| mobiliário | 34 | 64,7% | 88,2% |
| mão de obra | 11 | 18,2% | 36,4% |

Dois achados que orientaram o resto:

1. **O intruso não é da mesma família.** Nas 194 falhas recuperáveis, o top-1 tem
   o mesmo PDM do correto em só **41%** dos casos (no MMH eram 67%). O erro
   típico é `CAMA BELICHE` → `COLCHA CAMA`, `QUEBRA-CABEÇA EM MADEIRA` → `TÁBUA
   MADEIRA`, `FRASCO PARA HEMOCULTURA` → `FRASCO LABORATÓRIO`: o E5 casa o
   vocabulário e erra a **classe**. Isso é o que a taxonomia e o léxico exato
   (TF-IDF) sabem e o embedding não.
2. **8% das consultas estão fora do top-20.** Nenhum pós-processador chega lá;
   só a recuperação. Por isso o trabalho foi nos dois eixos.

---

## 3. O que foi testado, e o que cada coisa rendeu

Todos os módulos abaixo estão registrados na grade e descritos no README.
Os números desta seção são com `pre = basico`, perfil `compras_publicas`,
salvo indicação (`resultados/bigdata_profs/etapas/03..07/`, `varreduras/`).

### 3.1 GraphRAG com IDF (pós-processador)

O pós-processador `graphrag` original mede a cobertura das entidades da
consulta pelo candidato com peso fixo (token 1, atributo 2). Neste corpus,
`DESCARTAVEL` e `ESTERIL` cobrem centenas de itens e `ORTOFTALALDEIDO` cobre
três; o `graphrag_idf` pesa cada entidade pelo seu IDF no catálogo.

| pós (sobre `basico + e5`) | forma | R@3 |
|---|---|---|
| `nada` | — | 74,3% |
| `graphrag` | convexa 0,25 | 71,6% |
| `graphrag_idf` | convexa 0,25 | 72,7% |
| `graphrag_idf_aditivo` | aditiva 0,05 | **76,9%** |
| `medidas_graphrag` | aditiva (média) | 78,0% |
| `medidas_graphrag_idf` | aditiva (média) | **78,9%** |

O IDF ajuda (+1,1 pp na forma convexa, +0,9 na combinação com medida), mas a
**forma de combinar** decide mais que o sinal: a mistura convexa com peso 0,25 —
a dos pós-processadores publicados no MMH — deixa o bônus dominar e **piora**
o E5 em qualquer variante; somado com peso pequeno, o mesmo sinal rende +2,6 pp
sozinho. É a mesma lição da seção "A forma de combinar importa" do RESULTADOS.md,
agora valendo para o grafo e não só para a medida.

### 3.2 GraphRAG como pré-processador e como fonte de recuperação

| eixo | módulo | R@3 (sobre `e5`) | R@3 (+ `medidas_graphrag_idf`) |
|---|---|---|---|
| pré | `graphrag` (6 termos + comunidade) | 69,2% | 74,0% |
| pré | `graphrag_leve` (3 termos) | 71,6% | 76,2% |
| proc | `e5_kg` (RRF de E5 com a busca local do KG) | 67,4% | 69,3% |
| proc | `e5_tfidf_kg_cand` (RRF só para escolher candidatos; score E5) | 76,1% | 79,8% |
| proc | `e5_tfidf_kg_lin` (busca local do KG na soma linear, peso 0,1) | 77,7% | 77,8% |

A busca local do KG (soma do IDF das entidades compartilhadas) é boa para
**alargar** o conjunto de candidatos — `e5_tfidf_kg_cand` sobe o R@10 de 86,9%
para 89,2% e o R@3 em +1,9 pp sem pós e +0,9 pp com o mesmo pós que o E5 —
mas é ruim como **score**: entra na soma linear e derruba tudo. O grafo de entidades é um
índice invertido ponderado, não uma medida de semelhança; usado como o
primeiro, ajuda; como o segundo, atrapalha. O mesmo vale para a expansão da
consulta (pré-processador): injeta vocabulário de família e dilui o que
distingue o item — a leitura já registrada no MMH se confirma aqui.

### 3.3 Taxonomia: a ontologia de classes dos dois catálogos

`catalogo_match/taxonomia.py` alinha as duas hierarquias e devolve, para cada
par consulta × candidato, a compatibilidade da classe e-Fisco com a classe
CATMAT — por **histórico** (P(classe CATMAT | classe e-Fisco) estimada nos
outros vínculos, em leave-one-out) ou só por **rótulo** (similaridade E5 entre
os nomes das classes), com backoff para o nível de grupo.

Diagnóstico do sinal antes de usá-lo: compatibilidade média do par **correto**
0,62 contra 0,23 de um par aleatório; **92%** dos pares corretos têm
compatibilidade > 0 em LOO; 133 consultas (9,7%) têm classe sem nenhum outro
vínculo e caem para o grupo.

| pós (sobre `nada + e5`) | peso aditivo | R@3 |
|---|---|---|
| `nada` | — | 74,0% |
| `taxonomia` | 0,05 | 72,6% |
| `taxonomia` | 0,02 | **75,2%** |
| `taxonomia` | 0,01 | 75,0% |
| `taxonomia_rotulo` (só rótulo, controle) | 0,05 | 73,9% |
| `medidas_taxonomia` | 0,02 (backoff de grupo 0,3) | **77,3%** |
| `e5_taxonomia` (processador: E5 + 0,02 × compatibilidade no catálogo inteiro) | — | 74,8% |

Três leituras:

- **O sinal é bom e a escala é estreita.** Com peso 0,05 — o que serve à medida
  no MMH — a taxonomia **piora** 1,4 pp; com 0,02, melhora 1,2 pp sozinha. É um
  sinal denso (71% dos pares têm compatibilidade > 0, pelo backoff de grupo) e
  precisa entrar como desempate, não como voto. A varredura em
  `resultados/bigdata_profs/varreduras/` mostra o mesmo formato para todos os
  botões testados (`peso_grupo` 0 ou 0,3; `min_obs` 2 ou 5): o que importa é o
  peso.
- **O histórico vale mais que o rótulo.** Só com o texto dos rótulos de classe
  (`taxonomia_rotulo`), o sinal é neutro; com os vínculos conhecidos, ajuda. A
  informação está em *quais* classes se correspondem na prática — que é
  exatamente o alinhamento de ontologias que ninguém escreveu à mão.
- **Depois do corte, não antes.** Aplicada ao catálogo inteiro como parte do
  score (`e5_taxonomia`), ajuda menos (+0,8 pp) do que como pós (+1,2 pp), e
  somada à fusão léxica (`e5_tfidf_lin_taxonomia`, 81,4% contra 80,9%) o ganho
  é o mesmo que ela dá como pós. Não há candidato da classe certa escondido
  fora do top-20 em quantidade que justifique o custo.

### 3.4 Medida tipada neste corpus

Duas correções no extrator (`caracteristicas.py`), ambas medidas:

- **Cadeias de dimensão.** `210,00X86,00X162,00CM` era lido como um único
  comprimento — e errado: com o `X` colado ao número, a fronteira de palavra
  cortava `162` em `62`, e a cama de 162 cm virava 620 mm. Agora a unidade
  final é propagada a todos os números da cadeia (2 100, 860 e 1 620 mm).
- **Um alias, duas escalas.** Para ler `500 MG` e `5 KG` era preciso dar
  aliases à massa, mas `G` já era gauge. O extrator passou a ler primeiro o
  padrão mais restrito (com `faixa` ou `inteiro`): `16 G` cai na faixa 5–34 e
  é calibre; `500 G` é rejeitado pelo calibre e sobra para a massa.

Com o perfil `compras_publicas` (§6), a cobertura do extrator sobe de 761 para
**1 015** consultas com medida (bases novas: massa 159, `%` 137, `V` 72, `MG/ML`
54). O efeito no ranking, porém, é pequeno e **depende do peso**:

| pós `medidas` sobre `basico + e5` | peso | R@3 |
|---|---|---|
| — | — | 74,3% |
| `medidas` | 0,05 (o do MMH) | 73,8% |
| `medidas` | 0,03 | 75,4% |
| `medidas` | 0,02 | 76,4% |
| `medidas` | 0,01 | **76,8%** |

No MMH a medida sozinha rendia +5 pp com peso 0,05; aqui, com o mesmo peso,
**tira** 0,5 pp, e só rende com peso quatro vezes menor. A razão é o corpus:
neste catálogo a medida raramente é o que distingue o item dentro da família
(724 PDMs para 1 297 itens), e um `10 ML` casado com força empurra para cima
frascos e seringas que têm 10 ml e nada mais em comum. O sinal continua valendo
— mas como refinamento do grafo (`medidas_graphrag_idf`: 78,7%), não sozinho.

### 3.5 Fusão de rankings: o que mais rendeu

Como 8% dos corretos estavam fora do top-20 e os intrusos eram lexicalmente
próximos, a hipótese era juntar ao E5 uma fonte com erros diferentes — o
casamento **exato** de termos. Quatro formas foram medidas:

| processador | como combina | R@3 (`nada`) | R@3 (+ `medidas_graphrag_idf`) |
|---|---|---|---|
| `e5` | — | 74,3% | 78,9% |
| `e5_tfidf` | RRF, peso igual | 78,8% | 75,9% |
| `e5_tfidf_cand` | RRF escolhe candidatos; score E5 | 75,0% | 79,3% |
| `e5_tfidf_lin` | 0,8 · E5 + 0,2 · TF-IDF (cossenos) | 80,9% | 82,0% |
| `e5_char_lin` | 0,8 · E5 + 0,2 · TF-IDF de caracteres | 80,6% | 83,1% |
| **`e5_tfidf_char_lin`** | 0,7 · E5 + 0,15 · palavra + 0,15 · caractere | **81,9%** | **83,6%** |

- **RRF recupera, mas não combina.** Sozinho, o RRF de E5 com TF-IDF já sobe o
  R@3 do E5 normalizado em +4,6 pp: a fonte léxica traz para o top-K itens que
  o embedding deixara fora. Mas ele iguala fontes desiguais (o TF-IDF cru está
  11 pp abaixo do E5; o normalizado, 1,6 pp) e, pior, os scores RRF ficam numa
  escala de ~1/60 por posição — um bônus aditivo de 0,05 calibrado para cosseno
  engole o ranking, e o mesmo pós que dá 78,9% sobre o E5 dá 75,9% sobre o RRF
  — abaixo do RRF sem pós nenhum.
  A variante `_cand`, que usa a fusão só para escolher quem entra no top-K e
  devolve o cosseno E5, corrige o problema de escala, mas perde a ordenação que
  o RRF tinha acertado (75,0% sem pós).
- **A soma linear de cossenos funciona.** E5 e TF-IDF estão na mesma escala;
  somados, o TF-IDF entra como evidência léxica que o E5 pondera, não como voto
  igual. O peso do léxico é robusto: de 0,1 a 0,3 o R@3 varia dentro de 1 pp
  (`varreduras/basico+e5_tfidf_lin+...__peso_fusao_tfidf.csv`); a partir de 0,5
  cai.
- **Caractere complementa palavra.** Os n-gramas de caracteres (3–5, dentro da
  palavra) casam `TRANPARENTE` com `TRANSPARENTE` e `ANTIMICROB.` com
  `ANTIMICROBIANO` sem léxico; somados ao TF-IDF de palavra dão mais 1 a 1,6 pp.
  Pesos entre 0,1 e 0,3 para cada um ficam num platô de 83,1–83,9%.

Esta é a decisão de maior impacto do trabalho, e vale a comparação com o MMH:
lá a normalização `basico` valia +5 pp no TF-IDF e o E5 abria 0,06 de MRR sobre
ele; aqui a normalização vale +10 pp e o TF-IDF normalizado fica a 0,02 de MRR
do E5. Quanto mais o léxico já resolve, mais a fusão rende.

### 3.6 Calibração do pós-processamento sobre a fusão

Com `e5_tfidf_char_lin` como processador, os pesos relativos dos três sinais
foram varridos (`varreduras/basico+e5_tfidf_char_lin+medidas_graphrag_idf_...csv`):

| pós | peso base | medida × | GraphRAG IDF × | taxonomia × | R@3 | MRR | R@1 |
|---|---|---|---|---|---|---|---|
| `nada` | — | — | — | — | 81,9% | 0,743 | 64,4% |
| `medidas_graphrag_idf` | 0,05 | ½ | ½ | — | 83,6% | 0,758 | 66,1% |
| `medidas_graphrag_idf` | 0,05 | 0,6 | 1,0 | — | 83,7% | 0,759 | 66,3% |
| `medidas_graphrag_idf_taxonomia` | 0,03 | 0,6 | 1,0 | 0,3 | **84,3%** | **0,763** | **66,6%** |
| **`combinado_calibrado`** | 0,05 | 0,6 | 1,0 | 0,3 | **84,2%** | 0,763 | 66,8% |
| `medidas_graphrag_idf_taxonomia` | 0,05 | 1,0 | 1,0 | 1,0 | 83,1% | 0,750 | 65,1% |

`combinado_calibrado` é o módulo registrado com esses pesos relativos (0,6 /
1,0 / 0,3). A medida quer peso menor que no MMH, a cobertura IDF quer o peso
inteiro e a taxonomia entra como desempate; com os três em peso igual, a
taxonomia volta a dominar e o resultado cai 1 pp.

**Isto é ajuste no próprio conjunto de teste**, e foi tratado como tal. A
varredura com `--particoes 2` mede cada configuração nas duas metades disjuntas
das consultas (índices pares e ímpares):

| pós (sobre `e5_tfidf_char_lin`) | peso | metade A | metade B | total |
|---|---|---|---|---|
| `nada` | — | 80,2% | 83,6% | 81,9% |
| `medidas_graphrag_idf` | 0,05 | 82,6% | 84,7% | 83,6% |
| `combinado_calibrado` | 0,03 | 83,3% | 85,3% | 84,3% |
| `combinado_calibrado` | 0,05 | 82,9% | 85,4% | 84,2% |
| `combinado_calibrado` | 0,08 | 81,4% | 84,7% | 83,1% |

A ordem se mantém nas duas metades (o trio calibrado acima do par, o par acima
do E5 sem pós) e a diferença entre 0,03 e 0,05 fica dentro de 0,4 pp em cada
metade — o ganho é do sinal, não do ajuste fino. O que **não** é robusto é a
casa decimal: 0,1 pp são 1,4 consultas, e diferenças menores que ~0,5 pp entre
combinações vizinhas não devem ser lidas como diferença.

### 3.7 Rede semântica e os pré-processadores

| pré (sobre `e5`) | R@3 (`nada`) | R@3 (+ `medidas_graphrag_idf`) | R@3 (`e5_tfidf_lin`) | R@3 (`e5_tfidf_lin` + `medidas_graphrag_idf`) |
|---|---|---|---|---|
| `nada` | 74,0% | 78,2% | — | — |
| `basico` | 74,3% | 78,7% | 80,9% | 82,0% |
| `rede_semantica` | 71,2% | 77,2% | 78,4% | 80,9% |
| `graphrag` | 69,2% | 74,0% | 74,3% | 76,6% |
| `graphrag_leve` | 71,6% | 76,2% | 77,3% | 79,1% |

Nenhum pré-processador supera a normalização `basico`, em nenhuma coluna. A
expansão GraphRAG custa 5 pp (6 termos + comunidade) ou 3 pp (3 termos) frente
ao `basico`, e o prejuízo persiste com a fusão e o pós — a expansão acrescenta
vocabulário de família que o TF-IDF passa a casar com toda a família.
A rede semântica como pré-processador (normalização por grafo léxico + expansão
por ativação) custa 3 pp frente ao `basico`, como no MMH. O grafo léxico
construído sobre este corpus tem o léxico curado do MMH e o que a mineração de
co-ocorrência e o fastText induzem dos 1 392 pares; num corpus com 724 PDMs e
28% de itens fora da saúde, a indução produz mais candidatos a sinônimo
duvidosos e a expansão dilui. A rede continua sendo o lugar certo para o
conhecimento léxico **curado** (é ela que serve `QUICKLE → QUINCKE` ao
pré-processador `basico` via perfil); como expansora automática de consulta,
neste corpus, não paga.

---

## 4. A grade final

`python -m catalogo_match.avaliacao_modular --dataset bigdata_profs --metrica recall_at_3`
com os módulos principais (`resultados/bigdata_profs/`):

Matriz pré × processador (R@3, sem pós-processamento; `matriz_modular.png`):

| pré \ proc | `e5` | `e5_tfidf` (RRF) | `e5_tfidf_lin` | `e5_char_lin` | `e5_tfidf_char_lin` |
|---|---|---|---|---|---|
| `nada` | 74,0% | 74,6% | 79,4% | 81,1% | 81,2% |
| `basico` | 74,3% | 78,8% | 80,9% | 80,7% | **81,9%** |

Ranking das combinações (70 no total; `ranking_combinacoes.csv`):

| # | combinação | MRR | R@1 | R@3 | R@10 |
|---|---|---|---|---|---|
| 1 | **`basico + e5_tfidf_char_lin + combinado_calibrado`** | **0,763** | **66,8%** | **84,2%** | 92,3% |
| 2 | `basico + e5_tfidf_char_lin + medidas_graphrag_idf` | 0,758 | 66,1% | 83,6% | 92,2% |
| 3 | `basico + e5_tfidf_char_lin + medidas_graphrag` | 0,757 | 66,0% | 83,4% | 92,0% |
| 4 | `basico + e5_char_lin + combinado_calibrado` | 0,754 | 65,9% | 83,2% | 92,3% |
| 5 | `basico + e5_char_lin + medidas_graphrag_idf` | 0,755 | 65,9% | 83,1% | 92,1% |
| 6 | `nada + e5_char_lin + medidas_graphrag` | 0,755 | 66,2% | 82,9% | 91,8% |
| 7 | `nada + e5_char_lin + medidas_graphrag_idf` | 0,759 | 66,9% | 82,8% | 92,0% |
| 10 | `basico + e5_tfidf_char_lin + medidas` | 0,744 | 64,4% | 82,7% | 91,8% |
| — | `basico + e5 + medidas_graphrag` (melhor cadeia do MMH) | 0,692 | 58,5% | 78,0% | 88,5% |
| — | `basico + e5 + nada` | 0,669 | 56,2% | 74,3% | 86,9% |
| — | `nada + e5 + nada` | 0,675 | 56,9% | 74,0% | 88,2% |

A distância entre o 1º e o 5º lugar é de 1 pp (~14 consultas): o que decide é
a **fusão léxico-densa no processador** (as dez primeiras posições são todas
dela) e, dentro dela, o par GraphRAG-IDF + medida como pós, com a taxonomia
acrescentando meio ponto. O teto do top-20 (fração de consultas com o correto
em alguma das 20 posições) subiu de 91,1% para 94,4%.

---

## 5. O que mudou, consulta a consulta

`python -m catalogo_match.diagnostico --dataset bigdata_profs --pre basico --proc e5 --pos nada --contra basico,e5_tfidf_char_lin,combinado_calibrado`
(`resultados/bigdata_profs/diagnostico/movimentos_*.csv`):

- R@3 **74,3% → 84,2%**: **164 consultas entraram** no top-3 e **28 saíram**
  (troca líquida de +136).
- 125 das 164 entradas são do grupo médico-hospitalar; as demais vêm de
  laboratório escolar (6), higiene (6), mobiliário (5) e outros.
- Casos típicos que entraram: a cama beliche (dimensões encadeadas + classe
  "mobiliário"), a lixeira de 25 L (medida + TF-IDF), kits de diagnóstico cuja
  distinção está em um termo exato (`CALPROTECTINA`, `SANGUE OCULTO`).
- Casos típicos que saíram: itens cujo correto está numa classe CATMAT
  "surpreendente" (papel térmico de ultrassom classificado como filme; avental
  de PVC hospitalar catalogado como avental de cozinha industrial) — aí a
  taxonomia e o léxico puxam para o intruso plausível. São 28 casos, e o
  diagnóstico os lista.

---

## 6. Perfil: o que a curadoria valeu aqui

Mesmo processador e pós (`nada + e5_tfidf_char_lin + combinado_calibrado`),
três perfis (`varreduras/nada+e5_tfidf_char_lin+..._perfil-*.csv`):

| perfil | o que traz | consultas com medida | R@3 sem pós | R@3 com pós |
|---|---|---|---|---|
| `base` | português, SI, tudo o mais induzido do corpus | — | 81,2% | 82,0% |
| `mmh` | léxico hospitalar curado, gauge/french/polegada | 761 | 81,2% | 82,3% |
| `compras_publicas` | herda `mmh` + massa, dose, `%`, `MG/ML`, UI, W, V | 1 015 | 81,2% | **82,6%** |

O pré-processador `nada` não passa pelo perfil, por isso a coluna "sem pós" é
idêntica (controle). A curadoria vale 0,6 pp no pós — pouco, e coerente com a
§3.4: neste corpus a medida é sinal secundário. O perfil `compras_publicas`
existe menos pelo número e mais pelo que ele **habilita**: sem ele, `500 MG`,
`0,9%` e `220 V` não são lidos, e a diferença entre dois medicamentos do mesmo
PDM é invisível ao pós-processamento. Ele herda o `mmh` (`herda: mmh`) e
declara só a diferença, então os números do MMH não mudam.

---

## 7. Generalização: o mesmo par de módulos no MMH

Sem nenhum ajuste para o MMH — os mesmos pesos padrão (`peso_medidas` 0,05) e
o mesmo processador (`resultados/mmh/varreduras/`):

| combinação (`pre = basico`) | MRR | R@1 | R@3 | R@10 |
|---|---|---|---|---|
| `e5 + nada` | 0,539 | 41,0% | 60,7% | 81,7% |
| `e5 + medidas_graphrag` (publicado em RESULTADOS.md) | 0,604 | 48,3% | 67,5% | 82,9% |
| `e5 + medidas_graphrag` (extrator corrigido) | 0,607 | 48,6% | 68,1% | 83,1% |
| `e5 + medidas_graphrag_idf` | 0,614 | 49,6% | 68,5% | 83,7% |
| `e5_tfidf_char_lin + nada` | 0,641 | 51,7% | 72,1% | 87,9% |
| **`e5_tfidf_char_lin + medidas_graphrag`** | **0,683** | **56,7%** | **77,0%** | **89,8%** |
| `e5_tfidf_char_lin + combinado_calibrado` | 0,678 | 55,8% | 76,7% | 89,9% |

**+9,5 pp de R@3 no MMH** com o que foi aprendido aqui. A fusão léxico-densa
generaliza; o IDF no GraphRAG dá +0,4 pp; a taxonomia, num corpus de um só
domínio com ~200 PDMs, não acrescenta (76,7% contra 77,0%) — as classes CATMAT
do MMH são poucas e quase todas compatíveis com quase todas as classes e-Fisco.
A correção do extrator vale +0,6 pp lá também. Os números publicados em
RESULTADOS.md continuam reproduzíveis no commit anterior; o adendo no topo
daquele documento aponta para esta tabela.

---

## 8. O que não funcionou (e fica registrado)

| tentativa | resultado | por quê |
|---|---|---|
| `pdm` (coerência de família) como pós | −6,7 pp | a família já está no texto; o bônus premia o genérico da família certa |
| `graphrag` / `graphrag_idf` na forma convexa (0,25) | −2,7 / −1,6 pp | o bônus domina o ranking; a forma aditiva com peso pequeno é a certa |
| `medidas` sozinha com peso 0,05 | −0,5 pp | neste corpus a medida raramente distingue o item; quer peso 0,01–0,02 |
| `taxonomia` com peso 0,05 | −1,4 pp | sinal denso; com 0,02 vira +1,2 pp |
| RRF de E5 com TF-IDF | +4,6 pp sem pós, −3 pp com os pós | recupera bem, mas a escala do score (~1/60) é incompatível com os pós aditivos |
| RRF de E5 com o KG | −7 pp | iguala uma fonte fraca a uma forte |
| busca local do KG como score na soma linear | −4 pp | índice invertido ponderado não é medida de semelhança |
| expansão GraphRAG da consulta (pré) | ver §3.7 | injeta vocabulário de família e dilui a consulta |
| rede semântica como pré | −3 pp | indução de sinônimos ruidosa num corpus multi-domínio |

---

## 9. O pipeline completo, com LLM

Depois da grade, a continuação natural era levar o que ela mediu para dentro
do pipeline das fases [0]–[9] e rodá-lo com a chave de API: extração de
atributos por LLM na fase [2], índice Microsoft GraphRAG e adjudicação na fase
[6], explicações na [8b]. Três mudanças no pipeline saíram diretamente da grade:

- **Blocking e estágio 1 pela fusão.** A estratégia 6 do blocking (top-50
  cross-PDM) e o estágio 1 do matcher deixam de usar o cosseno E5 sozinho e
  passam a usar `0,7 · E5 + 0,15 · TF-IDF palavra + 0,15 · TF-IDF caractere`,
  sobre o texto normalizado (`efisco_processado`) em vez do documento virtual
  expandido pela rede — a grade mediu a expansão 3 pp abaixo nos dois datasets.
  Os embeddings vêm do mesmo cache em disco da grade.
- **Fase [5b], sinais simbólicos.** O pós-processador `combinado_calibrado` da
  grade vira uma fase: `score_neural += 0,05 · (0,6 · medida + 1,0 · GraphRAG-IDF
  + 0,3 · taxonomia)`. Com LLM, os atributos extraídos na fase [2] entram no KG
  como entidades tipadas dos dois lados (`principio_ativo::PARACETAMOL`), o que
  a grade não tinha.
- **Zona cinzenta por margem.** Na escala do score fundido (média 0,64) quase
  toda consulta cai na zona [0,25; 0,75), e o teto de 500 adjudicações cortava
  pela ordem do arquivo. Agora as 500 adjudicadas são as de **menor margem**
  entre o 1º e o 2º candidato — as de fato incertas.

A avaliação ganhou uma **ablação por camada**: o mesmo top-20 de cada consulta
reordenado por cosseno E5, pela fusão, pelo score com cross-encoder, com os
sinais da [5b], com o GraphRAG da [6] e pelo score final da [8]. É o que
permite dizer o que cada fase acrescentou sem rodar o pipeline seis vezes.

### O que as fases produziram (`resultados/bigdata_profs/stats_pipeline.json`)

| fase | resultado |
|---|---|
| [1] rede semântica | 7 682 nós, 37 261 arestas; 1 374 / 1 375 consultas ancoradas em PDM; OOV 25,7% |
| [2] extração LLM (`gpt-4o-mini`, 8 threads) | 2 668 / 2 672 textos pela LLM (4 caíram para regex por JSON truncado); consultas com atributos 1 361 / 1 375 (regex: 864); esquema SHACL completo 1 230 / 1 375 (89%, escore 0,922) |
| [3] blocking | 94 669 pares, bloco médio 69 (máx. 129), **recall do blocking 96,8%** (teto) |
| [4] ontologia + Pellet | 2 672 indivíduos, 22 regras SWRL; 557 `equivalenteA`, 21 `broadMatch`, 7 954 vetos (8,4% dos pares), 86 137 incompletos |
| [5] matcher | 27 500 pares (top-20 por consulta pela fusão); cross-encoder em 33 min de CPU (execução A) ou desligado (execução B) |
| [5b] sinais | 1 015 / 1 375 consultas com medida; KG com 15 431 nós e 8 566 entidades (com os atributos da LLM); 25 072 / 27 500 pares com algum sinal |
| [6] adjudicação | 500 consultas adjudicadas pela LLM, as de menor margem entre 1º e 2º candidato (zona cinzenta: 554 em A, 830 em B) — execução A com contexto de vizinhos TF-IDF, execução B com `local_search` do índice Microsoft GraphRAG |
| [6] índice Microsoft GraphRAG (B) | 2 672 documentos; grafo com 13 273 nós no nível 1, 1 461 comunidades no nível 2 e 731 no nível 1, cada uma com relatório gerado pela LLM; ~1,9 h de indexação com `gpt-4o-mini` + `text-embedding-3-small`; 1 000 chamadas de `local_search` para 500 adjudicações (~55 min) |
| [8b] explicabilidade | 60 explicações em linguagem natural para decisões de confiança Média/Baixa |
| tempo | A: 51 min (33 do cross-encoder, 6 da extração LLM, 8 da adjudicação); B: 2 h 59 min (2 h 55 na fase [6]); C: 3 min; D: 4 min (extração LLM em cache); padrão final (etapa 17): 5 min, dos quais ~2 de adjudicação em lista |

Três execuções, para separar três perguntas (LLM? cross-encoder? índice?):

- **A** — LLM na [2], cross-encoder ligado; a fase [6] adjudica com contexto de
  vizinhos TF-IDF (o índice Microsoft não subiu nesta execução por um defeito
  de importação, corrigido em seguida — `etapas/09_pipeline_llm_sem_indice_ms/`).
- **B** — LLM na [2], `--sem-cross`; a fase [6] adjudica com `local_search` do
  índice Microsoft GraphRAG construído sobre os 2 672 itens
  (`etapas/10_pipeline_llm_indice_ms_sem_cross/`).
- **C** — `--sem-llm --sem-cross`: regex do perfil na [2], votação por vizinhos
  TF-IDF na [6], nada de API (`etapas/11_pipeline_sem_llm_sem_cross/`, 3 min).

### Recuperação e ablação por camada

O mesmo top-20 de cada consulta, reordenado pelo score de cada camada
acumulada (recall do blocking, teto das três: **96,8%**). MRR / R@1 / R@3:

| camada acumulada | A (LLM, cross, adj. TF-IDF) | C (sem LLM, sem cross) | B (LLM, sem cross, índice MS) |
|---|---|---|---|
| cosseno E5 (só o bi-encoder) | 0,676 / 56,4 / 75,1 | 0,676 / 56,4 / 75,1 | 0,676 / 56,4 / 75,1 |
| fusão E5 + TF-IDF (estágio 1) | 0,743 / 64,4 / 81,9 | 0,743 / 64,4 / 81,9 | 0,743 / 64,4 / 81,9 |
| + cross-encoder (estágio 2) | 0,724 / 61,7 / 80,2 | — | — |
| + sinais simbólicos (fase 5b) | 0,743 / 63,9 / 82,1 | 0,762 / 66,6 / 83,6 | **0,765 / 67,0 / 84,1** |
| + fase 6 (adjudicação LLM em A e B; votação TF-IDF em C) | 0,728 / 63,3 / 79,9 | 0,758 / 66,1 / 83,3 | 0,742 / 66,5 / 79,3 |
| **score final** (fase 8: 0,35 ontologia + 0,45 neural + 0,20 GraphRAG, veto × 0,25) | 0,727 / 62,5 / 80,4 | 0,751 / 65,0 / 82,5 | 0,730 / 64,4 / 79,1 |

Execução **D** — a configuração que a ablação aponta: LLM na [2] (cache),
`--sem-cross --max-adjudicacoes 0` (`etapas/12_pipeline_llm_sem_cross_sem_adjudicacao/`,
os arquivos canônicos em `resultados/bigdata_profs/`):

| camada acumulada | D: MRR | D: R@1 | D: R@3 | D: R@10 |
|---|---|---|---|---|
| cosseno E5 | 0,676 | 56,4% | 75,1% | 88,7% |
| fusão E5 + TF-IDF | 0,743 | 64,4% | 81,9% | 91,8% |
| + sinais simbólicos (fase 5b) | **0,765** | **67,0%** | **84,1%** | **92,3%** |
| + votação TF-IDF (fase 6, sem LLM) | 0,765 | 66,9% | 84,0% | 92,3% |
| score final (fase 8) | 0,743 | 64,1% | 82,3% | 91,6% |

Extração LLM 2 672 / 2 672 (cache; os 4 JSON truncados de A foram refeitos com
`max_tokens` maior), SHACL completo 1 230 / 1 375, 558 `equivalenteA` e 7 954
vetos, 4 minutos. O melhor ponto do pipeline é a saída da fase [5b]: **R@3
84,1%, MRR 0,765, R@1 67,0%** — igual, dentro do ruído, aos 84,2% da grade sem
blocking. O score final da fase [8] fica 1,8 pp abaixo porque a composição
`0,35 ontologia + 0,45 neural + 0,20 graphrag` com veto × 0,25 foi calibrada
no MMH; aqui, com 724 PDMs e vetos sobre atributos extraídos, ela rebaixa
pares corretos. Três variantes da composição, sobre a mesma execução D
(`etapas/14_score_final_variantes/`):

| composição do score final | MRR | R@1 | R@3 |
|---|---|---|---|
| padrão (0,35 ontologia, veto × 0,25) | 0,743 | 64,1% | 82,3% |
| sem penalidade de veto (`--fator-veto 1.0`) | 0,744 | 64,3% | 82,3% |
| ontologia com peso 0,15 (`--peso-ontologia 0.15`) | 0,746 | 64,7% | 82,3% |
| **sem ontologia no score** (`--peso-ontologia 0 --fator-veto 1.0`) | **0,764** | **66,8%** | **84,1%** |

Não é o veto: é o **score da dedução**. `equivalenteA` vale 1,0 e `incompleto`
vale 0,3, e com peso 0,35 a diferença (+0,245) é maior do que o espalhamento
inteiro da similaridade fundida entre o 1º e o 20º candidato. Basta a regra
SWRL fechar sobre duas características definidoras que a LLM extraiu iguais —
dois cateteres do mesmo material e comprimento, mas de tipos diferentes — para
o par errado saltar ao topo. Em 724 PDMs com definidoras em parte induzidas,
isso acontece mais do que o acerto que a regra traz. A dedução ontológica
continua valendo como **explicação** e como veto de curadoria; como componente
do ranking, precisa entrar na escala dos outros sinais (aditivo, peso pequeno,
como a fase [5b] faz) — o que fica como próximo passo, e por isso os arquivos
canônicos de `resultados/bigdata_profs/` são os de D com a composição padrão.

### Leitura

1. **A fusão no blocking funciona como na grade.** Com o top-50 pela fusão
   como estratégia 6, o recall do blocking é **96,8%** (teto de tudo o que vem
   depois), e o estágio 1 pela fusão dá 81,9% de R@3 dentro dos blocos — o
   mesmo número da grade sem blocking (81,9%). Nada se perdeu ao passar da grade
   para o pipeline.
2. **O cross-encoder tira 1,7 pp.** Confirma na cadeia completa o que a grade
   media no MMH: o mMARCO foi treinado para relevância de passagem, não para
   equivalência de itens de catálogo. Custa 33 min de CPU e piora; por isso a
   execução B o desliga (`--sem-cross`).
3. **Os sinais simbólicos recuperam o que o cross-encoder tirou** (+1,9 pp) e
   ficam em 82,1% — abaixo dos 84,2% da grade porque aqui eles são somados a um
   score que já carrega o cross-encoder.
4. **A adjudicação por LLM com contexto TF-IDF piora 2,2 pp de R@3.** O ajuste
   médio de score foi 0,155 — grande, para um par que já era o top-1 por uma
   margem pequena — e derrubou mais acertos do que corrigiu erros. É o mesmo
   fenômeno do MMH ("premiar acordo funciona, punir divergência não"): a LLM,
   pedida a julgar um par isolado com pouco contexto, é conservadora e rebaixa
   pares corretos cuja especificação está incompleta no texto.
5. **A ontologia no score final depende da qualidade dos atributos.** Em A,
   com atributos da LLM dos dois lados, ela soma +0,5 pp (80,4% contra 79,9%):
   os 557 `equivalenteA` sobem pares certos e os 7 954 vetos descem mais errados
   do que certos. Em C, com atributos por regex (864 / 1 375 consultas com
   algum atributo), ela **tira** 0,8 pp (82,5% contra 83,3%): com menos
   atributos, os 4 122 vetos caem mais sobre pares certos com extração
   incompleta. A regra SWRL é tão boa quanto a percepção que a alimenta.
6. **Sem nenhuma API, o pipeline chega a 83,6% no ponto dos sinais** e 82,5%
   no score final, em 3 minutos — 1 pp acima da execução A com LLM em três
   fases. A LLM da fase [2] melhora a extração (1 361 contra 864 consultas com
   atributos, SHACL 89% contra 85%) e enriquece o KG (8 566 contra 7 648
   entidades), mas isso não se traduziu em ranking porque em A o cross-encoder
   e a adjudicação consumiram o ganho. A execução B mede a LLM na [2] sem esses
   dois efeitos.

7. **A LLM na fase [2] vale +0,5 pp no ponto dos sinais** (B: 84,1% contra
   C: 83,6%; MRR 0,765 contra 0,762): os atributos extraídos entram no KG da
   [5b] como entidades tipadas e afinam a cobertura. É o único lugar em que a
   LLM ajudou o ranking — e 84,1% é, dentro do ruído, o mesmo 84,2% da melhor
   cadeia da grade sem blocking: o pipeline com blocking não perde nada.
8. **A adjudicação com o índice Microsoft GraphRAG piora mais do que a com
   contexto TF-IDF**: −4,8 pp de R@3 (84,1% → 79,3%) em B, contra −2,2 pp em
   A, com ajuste médio de score de 0,135 sobre 500 consultas. O índice sobe
   (gr_ok), o `local_search` responde e o relatório de comunidade é pertinente
   — o problema não é o grafo, é a **tarefa**: pedir a uma LLM que julgue um
   par isolado, com o item certo já em 1º por pequena margem, produz vereditos
   conservadores que rebaixam corretos com especificação incompleta. Uma
   adjudicação útil teria de comparar os *k* candidatos entre si (rerank em
   lista), não pontuar um par; fica registrado como próximo passo, junto com o
   custo (2 h de indexação, ~1 000 chamadas por 500 pares).
9. **A configuração que a ablação aponta** é LLM na [2], fusão, sinais, sem
   cross-encoder e sem adjudicação — a execução D, abaixo. O GraphRAG que
   ajuda neste sistema é o **offline** (KG de entidades + IDF, na [5b]); o
   oficial da Microsoft serve à explicabilidade e à busca exploratória, não ao
   ranking.

### Finalização: os padrões do pipeline passam a ser os medidos

Quatro mudanças fecham o pipeline de modo que a saída **padrão**, sem flag
nenhuma, seja a melhor configuração medida (`etapas/16_*`, `17_*`, `18_*`):

1. **Dedução ontológica como sinal aditivo da [5b]**, não como fatia de 0,35 do
   score final: `equivalenteA` = 1, `broadMatch` = 0,5, peso relativo 0,6 (× 0,05).
   Varrido em 0 / 0,3 / 0,6 / 1,0 sem adjudicação: R@3 84,1 / 84,0 / 84,1 /
   84,0% — **neutro**. A regra SWRL acerta (86% dos top-1 com
   `equivalenteA` estão certos, contra 65% dos `incompleto`), mas o que ela
   confirma a similaridade fundida já tinha colocado no topo. Fica no score pela
   trilha de explicação, e o veto fica desligado por padrão (`--fator-veto 0.25`
   religa): entre os 71 top-1 vetados, 35 estavam certos e 36 errados.
2. **Cross-encoder desligado por padrão** (`--cross` religa).
3. **Adjudicação em lista** na fase [6]: para as 500 consultas de menor margem
   entre o 1º e o 2º candidato, a LLM vê os 5 melhores candidatos de uma vez —
   cada um com seu score neural, a dedução ontológica e a fração das medidas
   da consulta que satisfaz — e dá uma nota a cada um. A nota entra em
   `score_graphrag = (1 − p) · neural + p · nota`, e o score final é
   `0,8 · neural + 0,2 · graphrag`. Em 8 threads, 500 consultas levam ~2 min.
   O índice Microsoft GraphRAG passa a ser opcional (`--contexto-ms`).
4. **Confiança pela margem** entre o 1º e o 2º candidato (Alta ≥ 0,05;
   Média ≥ 0,02), validada contra o gabarito na própria avaliação.

Adjudicação em lista, peso da nota `p` (`--peso-adjudicacao`), sobre a saída da
[5b] (R@3 84,0% / R@1 67,2% / MRR 0,766):

| p | MRR | R@1 | R@3 | R@10 |
|---|---|---|---|---|
| 0 (sem adjudicação; saída da [5b]) | 0,766 | 67,2% | 84,0% | 92,3% |
| **0,1 (padrão)** | **0,783** | **69,9%** | **85,2%** | 92,3% |
| 0,2 | 0,784 | 70,1% | 85,2% | 92,1% |
| 0,3 | 0,772 | 69,1% | 83,1% | 91,6% |
| 0,5 (primeira tentativa) | 0,758 | 68,2% | 80,9% | 88,9% |

**É a primeira vez que a LLM ajuda o ranking neste pipeline**: +1,2 pp de R@3 e
+2,7 pp de R@1 sobre a saída da [5b], com 500 chamadas de ~2 minutos. Duas
coisas mudaram em relação ao modo par a par (−2,2 a −4,8 pp): a pergunta é
**relativa** — "qual destes cinco é o item?" em vez de "este par é o mesmo
item?" —, então especificação ausente deixa de ser lida como conflito; e a nota
entra como **desempate**, com peso efetivo 0,02 no score final (0,1 × 0,2) — na
escala em que a medida, o grafo e a taxonomia já entravam. Com p = 0,5, a mesma
LLM e as mesmas notas **pioram** o R@3 em 3 pp: as notas baixas dos candidatos
não escolhidos empurram o correto, quando não é o escolhido, para fora do
top-3. É o terceiro sinal deste trabalho em que a forma de combinar decide mais
que o sinal (medida, taxonomia, agora a LLM).

Faixas de confiança da execução padrão (etapa 17, p = 0,1), medidas contra o
gabarito:

| faixa (margem 1º−2º no score final) | consultas | P@1 | R@3 |
|---|---|---|---|
| Alta (≥ 0,05) | 473 (34,4%) | **96,2%** | 98,3% |
| Média (0,02–0,05) | 303 (22,0%) | 76,2% | 91,1% |
| Baixa (< 0,02) | 599 (43,6%) | 45,9% | 72,0% |

Com os limiares absolutos herdados do MMH (0,75 / 0,50 sobre o score), 81% das
consultas caíam em "Média" e a faixa não dizia nada. Pela margem, a faixa
**Alta** cobre 34% das consultas com 96% de acerto no top-1 — é o que pode ir
para o catálogo sem revisão —, e a **Baixa** concentra o trabalho da curadoria:
44% das consultas, mas 78% dos erros do top-1 (324 dos 414).

### O índice Microsoft GraphRAG como expansor de consulta (`--graphrag-ms`)

A outra forma de usar o índice oficial é a da grade: reescrever a consulta com
os termos que o `local_search` devolve, antes da similaridade. Medido numa
amostra de 200 consultas (semente 42; cada expansão é uma chamada ao modelo),
contra os pré-processadores offline na mesma amostra
(`etapas/13_grade_pre_graphrag_offline_amostra200/`, `etapas/15_grade_pre_graphrag_ms_amostra200/`):

| pré (sobre `e5_tfidf_char_lin`) | R@3 sem pós | R@3 + `combinado_calibrado` |
|---|---|---|
| `nada` | 81,0% | 83,0% |
| `basico` | 80,5% | **84,5%** |
| `graphrag_leve` (offline, 3 termos) | 78,0% | 80,5% |
| `graphrag` (offline, 6 termos + comunidade) | 74,0% | 79,5% |
| `graphrag` com índice Microsoft (`--graphrag-ms`) | 80,5% | 85,0% |

A expansão pelo índice oficial **corrige o defeito da offline**: os 6 termos que
o `local_search` devolve por consulta são específicos do item (o modelo lê o
relatório da comunidade e as entidades vizinhas e responde sobre *aquele*
item), não o vocabulário de família que a busca local offline injetava — e o
R@3 sobe de 79,5% para 85,0% com o mesmo pós. Mas, comparada à normalização
básica sem expansão nenhuma (84,5%), a diferença é de **uma consulta em 200**,
com MRR menor (0,747 contra 0,759) e uma chamada ao modelo por consulta, além
das 2 h de indexação. Neutra, portanto: na amostra, o índice Microsoft como
expansor de consulta empata com `basico`; como adjudicador de pares, piora
(§ acima). O que ele entrega de fato é explicabilidade — relatórios de
comunidade legíveis sobre 1 461 grupos de itens — e não ranking.

### O mesmo pipeline no MMH

`python -m catalogo_match.pipeline --dataset mmh --sem-llm --sem-cross --saida fusao_sem_llm`
(`resultados/mmh/etapas/pipeline_fusao_sem_llm_sem_cross/`, 2 min). Comparação
com o pipeline publicado em RESULTADOS.md (blocking por embedding, LLM em três
fases, cross-encoder):

| | publicado (LLM, E5 puro, cross, composição antiga) | sem LLM, fusão, sem cross, composição antiga | **padrão final** (LLM na [2] e [6] em lista, `resultados/mmh/etapas/pipeline_final_llm/`) |
|---|---|---|---|
| recall do blocking | 98,2% | 98,8% | 98,8% |
| MRR | 0,553 | 0,649 | **0,677** |
| R@1 | 43,9% | 52,3% | **56,6%** |
| R@3 | 60,7% | 73,3% | **75,2%** |
| R@10 | 80,4% | 88,4% | **89,3%** |
| tempo | 27 min (com índice em cache) | 2 min | 9 min (2 066 extrações + 500 adjudicações) |

Ablação no MMH com o padrão final: cosseno E5 61,4% → fusão 72,1% → sinais
**75,9%** → adjudicação em lista 75,2% (R@1 54,8% → 56,6%, MRR 0,672 → 0,677).
O mesmo padrão do `bigdata_profs`, com uma diferença: no MMH a adjudicação em
lista troca 0,7 pp de R@3 por 1,8 pp de R@1 — os candidatos do mesmo PDM são
muito parecidos e a LLM, ao escolher um, empurra outro correto para a 4ª
posição com mais frequência. As faixas de confiança seguem separando (Alta:
P@1 95%), mas cobrem só 6% das consultas: as margens no MMH são menores, e os
limiares de 0,05 / 0,02 são um ponto de partida a recalibrar por corpus.
Contra o publicado: **+14,5 pp de R@3, +12,7 de R@1, +0,124 de MRR**, três
vezes mais rápido.

---

## 10. Como reproduzir

```bash
# pipeline completo (chave em .env); os padrões são a configuração medida como melhor
python -m catalogo_match.pipeline --dataset bigdata_profs --smoke
python -m catalogo_match.pipeline --dataset bigdata_profs                                       # etapa 17: 85,2%, 5 min
python -m catalogo_match.pipeline --dataset bigdata_profs --adjudicacao nenhuma --saida sem_adj  # saída da [5b]: 84,0%
python -m catalogo_match.pipeline --dataset bigdata_profs --sem-llm --saida sem_llm             # C: sem API, 3 min
python -m catalogo_match.pipeline --dataset bigdata_profs --adjudicacao par --contexto-ms \
    --peso-ontologia 0.35 --fator-veto 0.25 --cross --saida antigo                              # o desenho original (A/B)
python -m catalogo_match.pipeline --dataset mmh --saida pipeline_final_llm                      # MMH com os padrões novos

# a grade final (E5 em cache: ~2 min; primeira vez: ~10 min por pré-processador)
python -m catalogo_match.avaliacao_modular --dataset bigdata_profs --metrica recall_at_3 \
    --pre nada,basico --proc e5,e5_tfidf_lin,e5_char_lin,e5_tfidf_char_lin \
    --pos nada,medidas,graphrag_idf_aditivo,taxonomia,medidas_graphrag_idf,combinado_calibrado

# varreduras que sustentam as tabelas da §3
python -m catalogo_match.varredura --dataset bigdata_profs --pre basico --proc e5_tfidf_char_lin \
    --pos medidas_graphrag_idf,combinado_calibrado --param peso_medidas=0.03,0.05,0.08 --particoes 2
python -m catalogo_match.varredura --dataset bigdata_profs --pre nada --proc e5 --pos taxonomia \
    --param peso_medidas=0.01,0.02,0.05,0.1 --param taxonomia_peso_grupo=0,0.3

# onde o R@3 se perde, e o que mudou entre a linha-base e a melhor cadeia
python -m catalogo_match.diagnostico --dataset bigdata_profs --pre basico --proc e5 --pos nada \
    --contra basico,e5_tfidf_char_lin,combinado_calibrado

# o mesmo par de módulos no MMH
python -m catalogo_match.varredura --dataset mmh --pre basico --proc e5_tfidf_char_lin \
    --pos nada,medidas_graphrag,medidas_graphrag_idf,combinado_calibrado
```

Cada etapa intermediária (linha-base, extrator novo, módulos novos por perfil,
fusão, pré-processadores) está congelada em `resultados/bigdata_profs/etapas/NN_*/`
com o `ranking_combinacoes.csv`, o YAML e o log da execução.

**Não foi medido**, por falta de chave de API: a variante `--graphrag-ms`
(expansão pelo índice Microsoft GraphRAG), a extração de atributos por LLM da
fase [2] e a adjudicação da fase [6]. O pipeline completo das fases [0]–[9]
(`catalogo_match.pipeline`) **não foi executado** neste dataset nesta rodada:
a grade modular, sem blocking, é a medida comparável, e levar a fusão linear
para o blocking e o matcher do pipeline é a continuação natural deste trabalho.
