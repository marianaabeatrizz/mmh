# Rodar o pipeline em outro dataset

O pipeline não conhece domínio. O que ele sabe sobre agulhas, gauge e cláusula do
art. 31 da Lei 8.078/90 está em dois arquivos de configuração, não no código:

| Arquivo | O que declara | Quando muda |
|---|---|---|
| `config/datasets/<nome>.yaml` | onde estão os CSVs, separador, encoding, para onde vão as saídas | a cada **corpus** novo |
| `config/perfis/<nome>.yaml` | léxico, famílias, unidades, esquema de atributos, papéis nos prompts | a cada **domínio** novo |

São coisas separadas de propósito: `mmh` e `mmh_opme` são dois corpora do mesmo
domínio e compartilham o perfil `mmh`. Um catálogo de medicamentos seria um
dataset **e** um perfil novos.

---

## Caso 1 — outro corpus, mesmo domínio

Dois arquivos de configuração, zero linha de código.

```bash
mkdir -p dados/meu_corpus
cp .../ground_truth.csv dados/meu_corpus/
```

```yaml
# config/datasets/meu_corpus.yaml
nome: meu_corpus
descricao: Recorte X do catálogo, safra 2026
perfil: mmh              # reaproveita o léxico já curado

separador: "|"
encoding: utf-8-sig
arquivo_padrao: principal

arquivos:
  principal: dados/meu_corpus/ground_truth.csv
```

```bash
python -m catalogo_match.avaliacao_modular --dataset meu_corpus
python -m catalogo_match.pipeline --dataset meu_corpus
```

As saídas vão para `resultados/meu_corpus/`, os caches para `cache/meu_corpus/` e
o índice do Microsoft GraphRAG para `graphrag_workspace/meu_corpus/`. Nada é
compartilhado entre datasets — um índice de outro corpus reaproveitado em
silêncio daria resultado errado com cara de certo.

### Se as colunas tiverem outros nomes

O esquema interno é `codigo_efisco`, `item_efisco`, `classe_efisco` e os três
equivalentes `*_catmat`. Um CSV com outros nomes declara o mapeamento, e o resto
do código continua vendo o esquema interno:

```yaml
colunas:
  item_efisco: descricao_origem
  item_catmat: descricao_destino
  codigo_efisco: cod_origem
  codigo_catmat: cod_destino
```

Só `item_efisco` e `item_catmat` são obrigatórias. Sem as de código, não há
gabarito e as métricas de recuperação não são calculáveis.

---

## Caso 2 — domínio novo, sem perfil

Funciona sem escrever perfil nenhum: aponte o dataset para `perfil: base` e o
pipeline **induz** do próprio corpus o que puder.

```bash
python -m catalogo_match.avaliacao_modular --dataset meu_corpus --perfil base
```

O que é induzido (`catalogo_match/config.py`, `induzir_perfil`):

| Bloco | Como é induzido |
|---|---|
| `atributos` | chaves dos rótulos estruturados do catálogo, com os valores mais frequentes virando a descrição que vai para o prompt |
| `familias` | substantivo-núcleo do rótulo de PDM, quando se repete em 2+ PDMs distintos |
| `atributos_definidores` | atributos presentes em ≥ 70% dos itens de cada família (mínimo 5 itens) |
| `hierarquia` | contenção de rótulo: `AGULHA` ⊃ `AGULHA PUNÇÃO ÓSSEA` |
| `stopwords` | tokens presentes em ≥ 50% dos registros |
| `boilerplate` | frases de 5 tokens repetidas em ≥ 40% dos registros |
| `unidades` | tokens que seguem número em ≥ 50% das vezes em que aparecem |

O que **não** é induzido, e por que:

- **Fator de conversão de unidade.** Uma unidade induzida entra sem fator: o
  pipeline passa a reconhecê-la e a comparar por igualdade, mas não converte.
  Afirmar que 1 CH = 0,33 MM é conhecimento externo, e inventá-lo criaria
  equivalência que ninguém verificou.
- **Regras regex de extração.** Sem elas, a fase [2] depende da LLM; com
  `--sem-llm` o pipeline extrai apenas o que o rótulo estruturado do catálogo já
  entrega.
- **Cláusula legal esparsa.** A indução de boilerplate só pega o que o corpus
  repete em quase todo registro. Cláusula que aparece em 10% dos textos pode ser
  especificação, e apagá-la destruiria sinal.
- **Sinônimo e abreviação.** Não por omissão: a rede semântica já os induz do
  histórico de vínculos (§3.6-b) e por embeddings (§3.6-d). Curar a semente do
  perfil melhora o resultado, não o habilita.

Cada bloco induzido e cada lacuna aparecem no log e no bloco `PERFIL` de
`resultados/<dataset>/avaliacao_modular.yaml` e `stats_pipeline.json` — é o que
permite ler um número sabendo quanto dele depende de conhecimento humano.

O sinal de medida é a exceção feliz da indução: ele não depende de léxico curado,
só das unidades. No MMH, `medidas_graphrag` rende o mesmo ganho de R@3 (+3,1 pp)
com perfil curado e com perfil `base` — ver "Separação de características" no
RESULTADOS.md. Se o seu domínio tem medida no texto, é o primeiro lugar a mexer.

### Quanto custa não ter perfil

Medido no MMH, 150 consultas, perfil `mmh` (curado) contra perfil `base` (tudo
induzido). O ganho da curadoria depende do processador:

| combinação | curado | induzido | Δ |
|---|---|---|---|
| `basico` + `fuzzy` + `unidades` | 0,5155 | 0,4312 | **+0,0843** |
| `basico` + `fuzzy` + `nada` | 0,4643 | 0,3828 | +0,0815 |
| `rede_semantica` + `fuzzy` + `unidades` | 0,4874 | 0,4462 | +0,0412 |
| `basico` + `tfidf` + `nada` | 0,4763 | 0,4678 | +0,0085 |
| `basico` + `tfidf` + `unidades` | **0,5189** | **0,5144** | +0,0045 |
| `nada` + `tfidf` + `nada` | 0,3559 | 0,3559 | 0,0000 |

Leitura: o casamento léxico (`fuzzy`) depende do léxico curado, e perde ~0,08 MRR
sem ele — é ele que faz `QUICKLE` e `QUINCKE` serem a mesma coisa. O TF-IDF
quase não percebe (+0,005), porque distribui o peso por muitos termos. As células
`nada` são idênticas nos dois perfis, o que confirma que o texto cru não passa
pelo perfil — serve de controle do próprio experimento.

---

## Caso 3 — domínio novo, com perfil curado

Copie `config/perfis/mmh.yaml` como ponto de partida e substitua o conteúdo de
domínio. A ordem que rende mais, medida acima, é: `lexico.abreviacoes` e
`lexico.variantes` primeiro (é o que o `fuzzy` sente), depois
`atributos` + `atributos_definidores` (que sustentam a regra SWRL e as shapes
SHACL), depois `texto.boilerplate`.

Blocos declarados no perfil **nunca** são sobrescritos pela indução: curadoria
vence indução, sempre. Deixar um bloco de fora é o que liga a indução dele.

Campos que merecem atenção:

- `iri_ontologia` / `iri_lexico` — deixe em branco se a ontologia não for
  publicada sob um domínio que seja seu. Em branco, o código monta um IRI local
  com o nome do dataset, em vez de emitir RDF sob `mmh.sad.pe.gov.br`, que
  afirmaria uma autoria que não existe.
- `llm.*` — as frases de papel dos prompts. Sem elas, são montadas a partir de
  `descricao`.
- `familias` — é a lista que o blocking usa. Note que ela **não** é a união
  automática com as chaves de `atributos_definidores`: três famílias do MMH
  (ESPARADRAPO, FRASCO, BOLSA) têm definidoras e nunca estiveram no blocking, e
  unir os dois conjuntos mudaria os números publicados. Para incluí-las, basta
  acrescentá-las em `familias`.
- `extracao` — cada regra declara a regex, que atributo preenche e de que grupo
  de captura (`0` = casamento inteiro), ou um `valor_fixo` quando a presença do
  padrão já é a informação.
- `unidades[*].aliases` — **é o que liga a separação de características.** São as
  grafias que o extrator procura no texto livre. Unidade sem `aliases` não é
  extraída, e sem nenhuma o sinal `medidas` fica inativo (bônus sempre zero, e o
  ranking não muda). Campos que acompanham:

  | campo | para quê |
  |---|---|
  | `aliases` | grafias no texto (`[MM, MILIMETRO]`; `['"', POL]`) |
  | `prefixo` | a unidade também vem antes do número (`G16`, não só `16G`) |
  | `fracao` | aceita fração mista (`3 1/2"` → 3,5) |
  | `faixa` | `[min, max]` plausível; fora disso o casamento é descartado |
  | `inteiro` | o valor precisa ser inteiro (calibre 16, não 16,3) |

  A comparação acontece na `base` da unidade, não na `categoria`: gauge e french
  são ambos calibre, e 7 FR não é 7 G. Declare `base` diferente para cada escala
  incomensurável.

  Para conferir o que o extrator está vendo — e o que falta declarar:

  ```bash
  python -m catalogo_match.caracteristicas --dataset meu_corpus
  python -m catalogo_match.caracteristicas --texto '10 ML, 21G X 1 1/2"'
  ```

  O perfil `base` já declara comprimento e volume do SI, então um domínio novo
  ganha esses de graça. Massa fica de fora de propósito: em catálogo médico "G" é
  gauge, não grama. Se o seu domínio pesa (alimentos, insumos), declare massa no
  seu perfil — e reveja a `faixa` de qualquer unidade que colida com ela.

## Conferir o que está em uso

```bash
python -m catalogo_match.avaliacao_modular --listar     # módulos da grade
python -c "from catalogo_match.config import ativar; import json; \
  print(json.dumps(ativar('mmh').perfil.resumo(), indent=2, ensure_ascii=False))"
```

O `resumo()` imprime o tamanho de cada bloco, a proveniência (`curado`,
`induzido`, `vazio`) e as lacunas declaradas.
