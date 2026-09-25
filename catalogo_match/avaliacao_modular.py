"""
avaliacao_modular.py
========================
Grade modular de avaliação — implementação da arquitetura de `pipeline.jpg`:

    Pré-proc 1 (Nada) ┐                            ┌ Pós-proc 1 (Nada) ┐
    Pré-proc 2        ├─→ Processador 1..N ─→ matriz ─→ Pós-proc 2      ├─→ Rank das
    Pré-proc N        ┘      (pré × proc)           └ Pós-proc N        ┘   combinações

Cada célula da matriz é uma métrica de recuperação (MRR por padrão) do par
(pré-processador, processador). Os pós-processadores reordenam o top-K de cada
célula, e o ranking final ordena todas as combinações pré × proc × pós.

Diferenças em relação ao `pipeline.py`
---------------------------------------------------
O pipeline monolítico executa UMA configuração fixa das fases [0]-[9]. Aqui cada
etapa vira um módulo trocável e todas as combinações são medidas lado a lado.
Não há blocking: cada consulta é pontuada contra o catálogo CATMAT inteiro, de
modo que o teto é 100% e as células são comparáveis entre si.

O GraphRAG entra em três eixos, sobre o mesmo KG (`graphrag`):
  - PRÉ-PROCESSADOR (`graphrag`, `graphrag_leve`): expande a consulta com
    entidades vizinhas e o rótulo da comunidade antes da similaridade.
  - PROCESSADOR (`e5_kg`, `e5_tfidf_kg_cand`, ...): a busca local do KG vira um
    ranking completo, fundido ao E5 antes do corte.
  - PÓS-PROCESSADOR (`graphrag`, `graphrag_idf`, `medidas_graphrag_idf`, ...):
    reordena o top-K pela cobertura das entidades da consulta pelo candidato,
    com peso fixo ou ponderada pelo IDF da entidade.

Os processadores híbridos (`e5_tfidf_lin`, `e5_tfidf_char_lin`, ...) fundem o
cosseno E5 com TF-IDF de palavra e de caractere; a taxonomia dos dois catálogos
(`taxonomia.py`) entra como pós (`taxonomia`, `combinado_calibrado`) ou como
fonte da fusão (`e5_taxonomia`). O que cada um rendeu está em RESULTADOS.md
(mmh) e docs/RESULTADOS-BIGDATA-PROFS.md (bigdata_profs).

Uso:
    python -m catalogo_match.avaliacao_modular                       # grade completa
    python -m catalogo_match.avaliacao_modular --listar              # módulos disponíveis
    python -m catalogo_match.avaliacao_modular --amostra 200         # subamostra
    python -m catalogo_match.avaliacao_modular --pre nada,basico --proc tfidf,fuzzy
    python -m catalogo_match.avaliacao_modular --metrica recall_at_3 # métrica das células
    python -m catalogo_match.avaliacao_modular --graphrag-ms         # índice Microsoft
    python -m catalogo_match.avaliacao_modular --dataset mmh_opme    # outro corpus
    python -m catalogo_match.avaliacao_modular --perfil base         # sem léxico curado
    python -m catalogo_match.avaliacao_modular --dataset bigdata_profs \\
        --proc e5_tfidf_char_lin --pos combinado_calibrado           # a melhor cadeia
    python -m catalogo_match.avaliacao_modular --peso-rel taxonomia=0.3 --peso-fusao tfidf=0.3

Ferramentas irmãs: `catalogo_match.varredura` (varre parâmetros de uma
combinação) e `catalogo_match.diagnostico` (onde o R@3 se perde).
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

# Forçar saída UTF-8 no terminal Windows (evita UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as _normalize_linhas

# (Um `sys.path.insert(0, <esta pasta>)` que havia aqui foi removido: com a
# pasta do pacote no sys.path, `import graphrag` passava a resolver para o
# NOSSO catalogo_match/graphrag.py em vez do pacote da Microsoft, e a fase [6]
# do pipeline quebrava com "attempted relative import with no known parent".)
from . import caracteristicas
from .config import contexto, perfil_ativo
from .preprocessamento import (
    extrair_atributos_catmat,
    normalizar_texto,
    preprocessar_texto_catmat,
    preprocessar_texto_efisco,
    remover_stopwords,
    tokenizar,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# Pastas de saida e de cache sao POR DATASET: duas grades de datasets diferentes
# nao devem sobrescrever a matriz uma da outra nem, pior, reusar embeddings do
# corpus errado.


def dir_resultados() -> Path:
    d = contexto().dataset.dir_resultados
    d.mkdir(parents=True, exist_ok=True)
    return d


def dir_cache_embeddings() -> Path:
    d = contexto().dataset.dir_cache_embeddings
    d.mkdir(parents=True, exist_ok=True)
    return d

MODELO_EMBEDDING = "intfloat/multilingual-e5-base"
MODELO_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Candidatos retidos por consulta — os pós-processadores só reordenam esse topo
K_CANDIDATOS = 20
KS_AVALIACAO = (1, 3, 5, 10)

# Profundidade do reranking do cross-encoder (custo ~5 pares/s em CPU)
K_CROSS_RERANK = 5

# Peso do sinal do pós-processador na mistura com o score do processador
PESO_POS_PADRAO = 0.25

# Peso do sinal de medida, na forma ADITIVA (ver `_pos_aditivo`). O valor saiu
# de varredura sobre as 1 012 consultas do MMH: 0,05 e o regime em que o sinal
# desempata sem substituir a similaridade.
#
#   forma / peso              R@3      MRR
#   convexa, peso >= 0,25   0,6601   0,5747   <- bonus domina; 0,25 e 0,60 dao
#                                                 o MESMO resultado, porque o
#                                                 ranking passa a ser ordenado
#                                                 primeiro pelo bonus
#   aditiva, peso 0,02      0,6700   0,5944
#   aditiva, peso 0,05      0,6729   0,5858   <- escolhido
#   aditiva, peso 0,10      0,6640   0,5787
#   aditiva, peso 0,20      0,6621   0,5763
PESO_MEDIDAS_PADRAO = 0.05


class ModuloIndisponivel(RuntimeError):
    """Dependência ausente (modelo, biblioteca): a célula fica sem valor."""


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    """Consultas e-Fisco, catálogo CATMAT e gabarito (só para avaliar)."""

    consultas: pd.DataFrame
    catalogo: pd.DataFrame
    gold: dict[str, set]
    arquivo: str = ""
    dataset: str = ""
    # Chave lógica do arquivo no dataset ("principal", "test"): é o que a rede
    # semântica precisa para construir o léxico sobre o MESMO corpus avaliado.
    chave: str = ""


@dataclass
class Textos:
    """Saída de um pré-processador: os dois lados já prontos para o processador."""

    pre_id: str
    queries: list[str]
    catalogo: list[str]
    meta: dict = field(default_factory=dict)


@dataclass
class Modulo:
    """Um bloco do diagrama: pré-processador, processador ou pós-processador."""

    id: str
    nome: str
    fn: Callable
    descricao: str = ""


# ---------------------------------------------------------------------------
# Carregamento do corpus
# ---------------------------------------------------------------------------

def carregar_corpus(arquivo: str = "",
                    amostra: int | None = None,
                    semente: int = 42) -> Corpus:
    """
    Lê o ground truth e separa catálogo (universo de busca), consultas e gabarito.
    Mesma semântica da fase [0] do pipeline, sem as dependências de LLM/OWL/SHACL.

    `arquivo` é a chave lógica do dataset ativo ("principal", "test") ou um nome
    de arquivo; quem resolve caminho, separador e encoding é o DatasetSpec.
    """
    ctx = contexto()
    caminho = ctx.dataset.caminho(arquivo)

    df = ctx.dataset.ler(arquivo)
    df.fillna("", inplace=True)
    ctx.completar_com_dados(df)

    # Colunas opcionais de taxonomia e proveniência: quando o CSV as traz, os
    # módulos de taxonomia e o diagnóstico as usam; quando não, seguem sem elas.
    extras_cat = [c for c in ("grupo_catmat",) if c in df.columns]
    extras_q = [c for c in ("grupo_efisco", "origem", "situacao") if c in df.columns]

    catalogo = (
        df[["codigo_catmat", "item_catmat", "classe_catmat", *extras_cat]]
        .drop_duplicates(subset="codigo_catmat")
        .reset_index(drop=True)
    )
    catalogo = catalogo[catalogo["codigo_catmat"].str.strip() != ""].reset_index(drop=True)
    catalogo["catmat_atributos"] = catalogo["item_catmat"].apply(extrair_atributos_catmat)
    catalogo["pdm"] = catalogo["catmat_atributos"].apply(
        lambda a: a.get("tipo_produto", "").strip()
    )

    consultas = (
        df[["codigo_efisco", "item_efisco", "classe_efisco", *extras_q]]
        .drop_duplicates(subset="codigo_efisco")
        .reset_index(drop=True)
    )
    consultas = consultas[consultas["codigo_efisco"].str.strip() != ""].reset_index(drop=True)

    gold: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        ce, cc = row.get("codigo_efisco", "").strip(), row.get("codigo_catmat", "").strip()
        if ce and cc:
            gold[ce].add(cc)

    if amostra and amostra < len(consultas):
        consultas = consultas.sample(n=amostra, random_state=semente).reset_index(drop=True)
        log.info("Subamostra: %d consultas (semente %d).", amostra, semente)

    log.info("Corpus [%s]: %d consultas x %d itens de catálogo (%d com gabarito).",
             ctx.dataset.nome, len(consultas), len(catalogo),
             sum(1 for c in consultas["codigo_efisco"] if c in gold))

    return Corpus(consultas=consultas, catalogo=catalogo, gold=dict(gold),
                  arquivo=caminho.name, dataset=ctx.dataset.nome, chave=arquivo)


# ---------------------------------------------------------------------------
# PRÉ-PROCESSADORES — os blocos da esquerda do diagrama
# ---------------------------------------------------------------------------

def _pre_nada(corpus: Corpus, ctx: dict) -> Textos:
    """Pré-proc 1 (Nada): texto cru, como veio das fontes."""
    return Textos(
        pre_id="nada",
        queries=corpus.consultas["item_efisco"].fillna("").astype(str).tolist(),
        catalogo=corpus.catalogo["item_catmat"].fillna("").astype(str).tolist(),
        meta={"transformacao": "nenhuma"},
    )


def _pre_basico(corpus: Corpus, ctx: dict) -> Textos:
    """Normalização do repositório: boilerplate jurídico, sinônimos, stopwords."""
    queries = [preprocessar_texto_efisco(t) for t in corpus.consultas["item_efisco"].fillna("")]
    catalogo = [preprocessar_texto_catmat(t) for t in corpus.catalogo["item_catmat"].fillna("")]
    return Textos(pre_id="basico", queries=queries, catalogo=catalogo,
                  meta={"transformacao": "boilerplate + sinônimos + stopwords"})


def _pre_rede_semantica(corpus: Corpus, ctx: dict) -> Textos:
    """
    Documento virtual da fase [1]: normalização por grafo léxico + expansão por
    propagação de ativação. Só as consultas são expandidas.
    """
    try:
        from .rede_semantica import (
            construir_grafo,
            expandir_por_ativacao,
            normalizar_termo,
        )
    except Exception as exc:                       # dependências do grafo léxico
        raise ModuloIndisponivel(f"rede_semantica indisponível: {exc}") from exc

    G = ctx.get("_grafo_lexico")
    if G is None:
        log.info("[pré/rede] Construindo grafo léxico de domínio (%s)...",
                 corpus.dataset or "dataset ativo")
        # Mesmo corpus da avaliação: o léxico de um dataset não descreve outro.
        G = construir_grafo(corpus.chave)
        ctx["_grafo_lexico"] = G

    base = _pre_basico(corpus, ctx)
    queries = []
    for texto in corpus.consultas["item_efisco"].fillna(""):
        toks = remover_stopwords(tokenizar(normalizar_texto(texto, manter_maiusculas=True)))
        toks_norm = [normalizar_termo(G, t, contexto=toks) for t in toks]
        ativacoes = expandir_por_ativacao(
            G, [t for t in toks_norm if len(t) > 3], limiar=0.15, max_saltos=2
        )
        expandidos = [
            t for t, s in sorted(ativacoes.items(), key=lambda x: -x[1])
            if s > 0.3 and t not in toks_norm
        ][:5]
        queries.append((" ".join(toks_norm) + " " + " ".join(expandidos)).strip())

    return Textos(pre_id="rede_semantica", queries=queries, catalogo=base.catalogo,
                  meta={"transformacao": "grafo léxico: normalização + ativação",
                        "n_nos": G.number_of_nodes(), "n_arestas": G.number_of_edges()})


def _pre_graphrag(corpus: Corpus, ctx: dict, *, pre_id: str = "graphrag",
                  max_termos: int = 6, top_k_vizinhos: int = 6,
                  incluir_comunidade: bool = True) -> Textos:
    """
    GraphRAG como pré-processador: expande a consulta com entidades recuperadas
    por busca local no KG + rótulo da comunidade (`graphrag`).
    """
    from . import graphrag as gr

    base = _pre_basico(corpus, ctx)

    kg = ctx.get("_kg_graphrag")
    if kg is None and not ctx.get("graphrag_ms"):
        kg = gr.construir_kg(corpus.consultas, corpus.catalogo)
        ctx["_kg_graphrag"] = kg

    queries, stats = gr.expandir_consultas(
        corpus.consultas,
        corpus.catalogo,
        textos_base=base.queries,
        max_termos=max_termos,
        top_k_vizinhos=top_k_vizinhos,
        incluir_comunidade=incluir_comunidade,
        usar_ms_graphrag=bool(ctx.get("graphrag_ms")),
        api_key=ctx.get("api_key", ""),
        kg=kg,
    )
    return Textos(pre_id=pre_id, queries=queries, catalogo=base.catalogo,
                  meta={"transformacao": "básico + expansão GraphRAG", **stats})


def _pre_graphrag_leve(corpus: Corpus, ctx: dict) -> Textos:
    """Expansão GraphRAG conservadora: 3 termos, sem rótulo de comunidade."""
    return _pre_graphrag(corpus, ctx, pre_id="graphrag_leve", max_termos=3,
                         top_k_vizinhos=4, incluir_comunidade=False)


PRE_PROCESSADORES: dict[str, Modulo] = {
    "nada":           Modulo("nada", "Pré-proc 1 (Nada)", _pre_nada,
                             "texto cru, sem transformação"),
    "basico":         Modulo("basico", "Normalização básica", _pre_basico,
                             "boilerplate jurídico + sinônimos + stopwords"),
    "rede_semantica": Modulo("rede_semantica", "Rede semântica", _pre_rede_semantica,
                             "grafo léxico: normalização + propagação de ativação"),
    "graphrag":       Modulo("graphrag", "GraphRAG (expansão por grafo)", _pre_graphrag,
                             "6 entidades vizinhas no KG + rótulo da comunidade"),
    "graphrag_leve":  Modulo("graphrag_leve", "GraphRAG conservador", _pre_graphrag_leve,
                             "3 entidades vizinhas, sem rótulo de comunidade"),
}


# ---------------------------------------------------------------------------
# PROCESSADORES — os losangos do diagrama
# ---------------------------------------------------------------------------

def _top_k(matriz: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k por linha, ordenado decrescentemente."""
    k = min(k, matriz.shape[1])
    idx = np.argpartition(-matriz, kth=k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(matriz, idx, axis=1)
    ordem = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(scores, ordem, axis=1), np.take_along_axis(idx, ordem, axis=1)


def _matriz_tfidf(textos: Textos, ctx: dict) -> np.ndarray:
    """Cosseno TF-IDF (1-2 gramas) consulta x catálogo, matriz cheia, memorizada."""
    chave = f"tfidf::{textos.pre_id}"
    if chave not in ctx:
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)
        vec.fit(textos.queries + textos.catalogo)
        Q = _normalize_linhas(vec.transform(textos.queries))
        C = _normalize_linhas(vec.transform(textos.catalogo))
        ctx[chave] = (Q @ C.T).toarray().astype(np.float32)
    return ctx[chave]


def _proc_tfidf(textos: Textos, top_k: int, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Similaridade de cosseno sobre TF-IDF de 1-2 gramas."""
    return _top_k(_matriz_tfidf(textos, ctx), top_k)


def _proc_fuzzy(textos: Textos, top_k: int, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """token_set_ratio do rapidfuzz — casamento léxico tolerante a ordem."""
    try:
        from rapidfuzz import fuzz, process as rprocess
    except ImportError as exc:
        raise ModuloIndisponivel("rapidfuzz não instalado") from exc

    matriz = rprocess.cdist(
        textos.queries, textos.catalogo, scorer=fuzz.token_set_ratio, workers=-1
    ).astype(np.float32) / 100.0
    return _top_k(matriz, top_k)


def _carregar_sentence_transformer(nome: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ModuloIndisponivel("sentence-transformers não instalado") from exc
    try:
        return SentenceTransformer(nome)
    except Exception as exc:
        raise ModuloIndisponivel(f"modelo {nome} indisponível: {str(exc)[:100]}") from exc


def _embeddings_e5(textos: Textos, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Codifica os dois lados com o E5 assimétrico. Cacheado em memória e disco."""
    chave = f"e5::{textos.pre_id}"
    if chave in ctx:
        return ctx[chave]

    assinatura = hashlib.sha1(
        (MODELO_EMBEDDING + "|" + "\n".join(textos.queries)
         + "|||" + "\n".join(textos.catalogo)).encode("utf-8")
    ).hexdigest()[:16]
    arquivo_cache = dir_cache_embeddings() / f"e5_{textos.pre_id}_{assinatura}.npz"

    if arquivo_cache.exists():
        dados = np.load(arquivo_cache)
        emb = (dados["q"], dados["c"])
        log.info("[proc/e5] Embeddings reaproveitados do cache: %s", arquivo_cache.name)
    else:
        modelo = _carregar_sentence_transformer(MODELO_EMBEDDING)
        log.info("[proc/e5] Codificando %d consultas + %d itens (pré=%s)...",
                 len(textos.queries), len(textos.catalogo), textos.pre_id)
        emb_q = modelo.encode([f"query: {t}" for t in textos.queries],
                              batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        emb_c = modelo.encode([f"passage: {t}" for t in textos.catalogo],
                              batch_size=64, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        emb = (emb_q.astype(np.float32), emb_c.astype(np.float32))
        np.savez_compressed(arquivo_cache, q=emb[0], c=emb[1])

    ctx[chave] = emb
    return emb


def _matriz_e5(textos: Textos, ctx: dict) -> np.ndarray:
    """Cosseno E5 consulta x catálogo, matriz cheia."""
    emb_q, emb_c = _embeddings_e5(textos, ctx)
    return emb_q @ emb_c.T


def _proc_e5(textos: Textos, top_k: int, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """Bi-encoder assimétrico multilingual-e5-base (query: / passage:)."""
    return _top_k(_matriz_e5(textos, ctx), top_k)


# ---- Recuperação híbrida: fusão de rankings ------------------------------------
# Os pós-processadores só reordenam o top-K do processador; o que o E5 deixou
# na 30ª posição está perdido para eles. A fusão ataca o TETO: junta rankings
# de fontes com erros diferentes (denso, léxico, grafo de entidades, taxonomia)
# ANTES do corte, para que o item certo entre no top-K por qualquer uma delas.


def _kg_de(ctx: dict):
    """KG do GraphRAG offline, construído uma vez por grade e reaproveitado."""
    from . import graphrag as gr

    kg = ctx.get("_kg_graphrag")
    if kg is None:
        corpus = ctx["_corpus"]
        kg = gr.construir_kg(corpus.consultas, corpus.catalogo)
        ctx["_kg_graphrag"] = kg
    return kg


def _idf_entidades(kg, ctx: dict) -> dict[str, float]:
    """
    IDF de cada entidade do KG, pela sua frequência documental no catálogo.

    Um token que aparece em 400 itens (DESCARTAVEL) não diz qual deles é o
    certo; um que aparece em 3 (ORTOFTALALDEIDO) quase decide sozinho. O peso
    fixo 1,0/2,0 do pós-processador `graphrag` original ignorava isso.
    """
    if "_idf_kg" not in ctx:
        m = max(1, len(kg.ents_catalogo))
        ctx["_idf_kg"] = {ent: float(np.log(1.0 + m / len(idxs)))
                          for ent, idxs in kg.indice_invertido.items()}
    return ctx["_idf_kg"]


def _matriz_kg(textos: Textos, ctx: dict) -> np.ndarray:
    """
    Busca local do GraphRAG como RANKING COMPLETO: soma do IDF das entidades
    compartilhadas entre consulta e item, normalizada pelo IDF total da
    consulta. É o que a `_busca_local` do pré-processador faz para achar
    vizinhos, generalizado ao catálogo inteiro e ponderado por raridade.
    """
    chave = "kg::matriz"
    if chave in ctx:
        return ctx[chave]
    kg = _kg_de(ctx)
    idf = _idf_entidades(kg, ctx)
    teto = perfil_ativo().max_itens_por_entidade
    n, m = len(kg.ents_consulta), len(kg.ents_catalogo)
    matriz = np.zeros((n, m), dtype=np.float32)
    for i, ents in kg.ents_consulta.items():
        relevantes = [e for e in ents if not e.startswith("PDM::")] or list(ents)
        total = sum(idf.get(e, 0.0) for e in relevantes)
        if total <= 0:
            continue
        for ent in relevantes:
            idxs = kg.indice_invertido.get(ent)
            if not idxs or len(idxs) > teto:
                continue
            matriz[i, idxs] += idf[ent]
        matriz[i] /= total
    ctx[chave] = matriz
    return matriz


def _matriz_taxonomia(ctx: dict) -> np.ndarray:
    """Compatibilidade taxonômica (LOO) consulta x catálogo, matriz cheia."""
    if "_taxonomia_matriz" not in ctx:
        from . import taxonomia as tx

        corpus = ctx["_corpus"]
        al = ctx.get("_taxonomia_hist")
        if al is None:
            al = tx.alinhamento_historico(corpus.consultas, corpus.catalogo, corpus.gold)
            # Botões do sinal, ajustáveis pela CLI (--taxonomia-peso-grupo,
            # --taxonomia-min-obs) para a varredura sem editar código.
            if "taxonomia_peso_grupo" in ctx:
                al.peso_grupo = float(ctx["taxonomia_peso_grupo"])
            if "taxonomia_min_obs" in ctx:
                al.min_obs = int(ctx["taxonomia_min_obs"])
            ctx["_taxonomia_hist"] = al
            ctx["_taxonomia_resumo"] = {
                **al.resumo(),
                **tx.diagnostico_cobertura(al, corpus.gold, corpus.consultas, corpus.catalogo),
            }
            log.info("[taxonomia] correto com compatibilidade > 0 (LOO): %.1f%% das consultas; "
                     "classe sem histórico: %d",
                     100 * ctx["_taxonomia_resumo"]["fracao_correto_compativel"],
                     ctx["_taxonomia_resumo"]["classe_sem_historico_loo"])
        ctx["_taxonomia_matriz"] = tx.matriz_compatibilidade(al)
    return ctx["_taxonomia_matriz"]


def _ranks(matriz: np.ndarray) -> np.ndarray:
    """Posição (0 = melhor) de cada item em cada linha."""
    ordem = np.argsort(-matriz, axis=1, kind="stable")
    ranks = np.empty_like(ordem)
    linhas = np.arange(matriz.shape[0])[:, None]
    ranks[linhas, ordem] = np.arange(matriz.shape[1])[None, :]
    return ranks


RRF_K = 60   # constante clássica do Reciprocal Rank Fusion (Cormack et al., 2009)


def _fusao_rrf(matrizes: list[np.ndarray], pesos: list[float]) -> np.ndarray:
    """Σ peso / (RRF_K + posição): só a ORDEM de cada fonte importa, não a escala."""
    total = np.zeros_like(matrizes[0], dtype=np.float32)
    for mat, peso in zip(matrizes, pesos):
        total += peso / (RRF_K + _ranks(mat).astype(np.float32))
    return total


def _matriz_tfidf_char(textos: Textos, ctx: dict) -> np.ndarray:
    """
    Cosseno TF-IDF de n-gramas de CARACTERES (3-5, dentro da palavra).

    Complementa o de palavras: casa QUICKLE com QUINCKE, TRANPARENTE com
    TRANSPARENTE e "ANTIMICROB." com ANTIMICROBIANO sem precisar de léxico.
    É a mesma rede de segurança lexical do blocking da fase [3].
    """
    chave = f"tfidf_char::{textos.pre_id}"
    if chave not in ctx:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60000,
                              sublinear_tf=True)
        vec.fit(textos.queries + textos.catalogo)
        Q = _normalize_linhas(vec.transform(textos.queries))
        C = _normalize_linhas(vec.transform(textos.catalogo))
        ctx[chave] = (Q @ C.T).toarray().astype(np.float32)
    return ctx[chave]


_FONTES = {
    "e5":         lambda textos, ctx: _matriz_e5(textos, ctx),
    "tfidf":      lambda textos, ctx: _matriz_tfidf(textos, ctx),
    "tfidf_char": lambda textos, ctx: _matriz_tfidf_char(textos, ctx),
    "kg":         lambda textos, ctx: _matriz_kg(textos, ctx),
    "taxonomia":  lambda textos, ctx: _matriz_taxonomia(ctx),
}


def _proc_fusao(fontes: tuple[str, ...], modo: str = "rrf",
                pesos: tuple[float, ...] | None = None,
                escores: str = "") -> Callable:
    """
    Processador híbrido: combina os rankings de várias fontes antes do corte.

    `rrf`    — Reciprocal Rank Fusion, robusto a escalas incomparáveis.
    `linear` — soma ponderada dos scores; exige fontes na mesma escala (os
               cossenos E5 e TF-IDF estão ambos em [0,1]; a taxonomia também).

    `escores` — quando dado (p. ex. "e5"), a fusão escolhe QUEM entra no top-K,
    mas o score devolvido é o dessa fonte. Separa os dois papéis: a fusão
    alarga o conjunto de candidatos (teto), o E5 continua a ordená-los e os
    pós-processadores continuam a receber cosseno, na escala para a qual os
    seus pesos foram calibrados. Sem isso, o score RRF (~1/60 por posição)
    faz um bônus aditivo de 0,05 engolir o ranking inteiro — foi o que
    derrubou `e5_tfidf + medidas_graphrag` para 0,735 contra 0,772 do E5 puro.
    """
    pesos = pesos or tuple(1.0 for _ in fontes)

    def aplicar(textos: Textos, top_k: int, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
        matrizes = [_FONTES[f](textos, ctx) for f in fontes]
        # Peso de cada fonte ajustável por execução (CLI: --peso-fusao
        # tfidf=0.3,kg=0.1; varredura: --param peso_fusao_tfidf=0.1,0.2,...).
        efetivos = [ctx.get(f"peso_fusao_{f}", p) for f, p in zip(fontes, pesos)]
        if modo == "rrf":
            combinada = _fusao_rrf(matrizes, efetivos)
        else:
            combinada = sum(p * m for p, m in zip(efetivos, matrizes)).astype(np.float32)
        scores, idx = _top_k(combinada, top_k)
        if escores:
            base = _FONTES[escores](textos, ctx)
            scores = np.take_along_axis(base, idx, axis=1).astype(np.float32)
            ordem = np.argsort(-scores, axis=1, kind="stable")
            scores = np.take_along_axis(scores, ordem, axis=1)
            idx = np.take_along_axis(idx, ordem, axis=1)
        return scores, idx

    aplicar.__name__ = f"_proc_fusao_{'_'.join(fontes)}_{modo}{'_' + escores if escores else ''}"
    return aplicar


def _proc_e5_cross(textos: Textos, top_k: int, ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Cascata do repositório: E5 no estágio 1 e cross-encoder reordenando apenas a
    cabeça do ranking (mistura 0,7 cosseno / 0,3 cross). A profundidade do
    reranking é limitada porque o cross-encoder custa ~5 pares/s nesta máquina.
    """
    scores, idx = _proc_e5(textos, top_k, ctx)
    profundidade = min(ctx.get("k_cross", K_CROSS_RERANK), idx.shape[1])

    chave = f"cross::{textos.pre_id}::{profundidade}"
    assinatura = hashlib.sha1(
        (MODELO_CROSS_ENCODER + f"|{profundidade}|{top_k}|"
         + "\n".join(textos.queries) + "|||" + "\n".join(textos.catalogo)).encode("utf-8")
    ).hexdigest()[:16]
    arquivo_cache = dir_cache_embeddings() / f"cross_{textos.pre_id}_{assinatura}.npy"

    if chave in ctx:
        scores_cross = ctx[chave]
    elif arquivo_cache.exists():
        scores_cross = np.load(arquivo_cache)
        ctx[chave] = scores_cross
        log.info("[proc/e5_cross] Scores reaproveitados do cache: %s", arquivo_cache.name)
    else:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ModuloIndisponivel("sentence-transformers não instalado") from exc
        try:
            ce = CrossEncoder(MODELO_CROSS_ENCODER, max_length=128)
        except Exception as exc:
            raise ModuloIndisponivel(
                f"cross-encoder indisponível: {str(exc)[:100]}") from exc

        pares = [(textos.queries[i], textos.catalogo[j])
                 for i in range(idx.shape[0]) for j in idx[i, :profundidade]]
        log.info("[proc/e5_cross] Cross-encoder em %d pares (pré=%s, profundidade %d)...",
                 len(pares), textos.pre_id, profundidade)
        brutos = np.asarray(ce.predict(pares, batch_size=64, show_progress_bar=False),
                            dtype=np.float32)
        scores_cross = (1.0 / (1.0 + np.exp(-brutos))).reshape(idx.shape[0], profundidade)
        np.save(arquivo_cache, scores_cross)
        ctx[chave] = scores_cross

    # Só a cabeça é remisturada e reordenada; a cauda segue na ordem do E5.
    scores_novos, idx_novos = scores.copy(), idx.copy()
    cabeca = (0.7 * np.clip(scores[:, :profundidade], 0, 1)
              + 0.3 * np.clip(scores_cross, 0, 1))
    ordem = np.argsort(-cabeca, axis=1, kind="stable")
    scores_novos[:, :profundidade] = np.take_along_axis(cabeca, ordem, axis=1)
    idx_novos[:, :profundidade] = np.take_along_axis(idx[:, :profundidade], ordem, axis=1)
    return scores_novos, idx_novos


PROCESSADORES: dict[str, Modulo] = {
    "tfidf":    Modulo("tfidf", "TF-IDF (cosseno)", _proc_tfidf,
                       "1-2 gramas, 20k features"),
    "fuzzy":    Modulo("fuzzy", "Fuzzy token_set_ratio", _proc_fuzzy,
                       "rapidfuzz, casamento léxico"),
    "e5":       Modulo("e5", "E5 bi-encoder", _proc_e5,
                       MODELO_EMBEDDING),
    "e5_cross": Modulo("e5_cross", "E5 + cross-encoder", _proc_e5_cross,
                       f"{MODELO_EMBEDDING} + {MODELO_CROSS_ENCODER} "
                       f"(reranking do top-{K_CROSS_RERANK})"),
    # --- híbridos: fusão de rankings antes do corte (ataca o teto do top-K) ---
    "e5_tfidf":     Modulo("e5_tfidf", "E5 + TF-IDF (RRF)",
                           _proc_fusao(("e5", "tfidf")),
                           "fusão recíproca de ranks: denso + léxico"),
    "e5_kg":        Modulo("e5_kg", "E5 + GraphRAG local search (RRF)",
                           _proc_fusao(("e5", "kg")),
                           "denso + ranking por entidades do KG ponderadas por IDF"),
    "e5_tfidf_kg":  Modulo("e5_tfidf_kg", "E5 + TF-IDF + KG (RRF)",
                           _proc_fusao(("e5", "tfidf", "kg")),
                           "as três fontes com peso igual"),
    "e5_taxonomia": Modulo("e5_taxonomia", "E5 guiado pela taxonomia",
                           _proc_fusao(("e5", "taxonomia"), modo="linear",
                                       pesos=(1.0, 0.02)),
                           "cosseno E5 + 0,02 x compatibilidade de classe (LOO) sobre o "
                           "catálogo inteiro, antes do corte"),
    "e5_tfidf_taxonomia": Modulo("e5_tfidf_taxonomia", "E5 + TF-IDF (RRF) guiado pela taxonomia",
                                 _proc_fusao(("e5", "tfidf", "taxonomia"), pesos=(1.0, 1.0, 0.5)),
                                 "RRF de denso + léxico + taxonomia (peso 0,5)"),
    # --- fusão só para ESCOLHER candidatos; o score que segue é o cosseno E5 ---
    "e5_tfidf_cand":    Modulo("e5_tfidf_cand", "candidatos de E5+TF-IDF (RRF), score E5",
                               _proc_fusao(("e5", "tfidf"), escores="e5"),
                               "a fusão alarga o top-K; o E5 ordena e os pós recebem cosseno"),
    "e5_tfidf_kg_cand": Modulo("e5_tfidf_kg_cand", "candidatos de E5+TF-IDF+KG (RRF), score E5",
                               _proc_fusao(("e5", "tfidf", "kg"), escores="e5"),
                               "idem, com a busca local do KG como terceira fonte"),
    "e5_tfidf_lin":     Modulo("e5_tfidf_lin", "E5 + TF-IDF (linear 0,8/0,2)",
                               _proc_fusao(("e5", "tfidf"), modo="linear", pesos=(0.8, 0.2)),
                               "soma ponderada de cossenos: mesma escala do E5"),
    "e5_char_lin":      Modulo("e5_char_lin", "E5 + TF-IDF de caracteres (linear 0,8/0,2)",
                               _proc_fusao(("e5", "tfidf_char"), modo="linear", pesos=(0.8, 0.2)),
                               "denso + n-gramas de caracteres: robusto a grafia e abreviação"),
    "e5_tfidf_char_lin": Modulo("e5_tfidf_char_lin", "E5 + TF-IDF palavra + caractere (linear)",
                                _proc_fusao(("e5", "tfidf", "tfidf_char"), modo="linear",
                                            pesos=(0.7, 0.15, 0.15)),
                                "as três fontes léxico-densas somadas"),
    "e5_tfidf_kg_lin":  Modulo("e5_tfidf_kg_lin", "E5 + TF-IDF + KG (linear)",
                               _proc_fusao(("e5", "tfidf", "kg"), modo="linear",
                                           pesos=(0.8, 0.2, 0.1)),
                               "soma de cossenos + busca local do KG (IDF normalizado)"),
    "e5_tfidf_lin_taxonomia": Modulo("e5_tfidf_lin_taxonomia", "E5 + TF-IDF + taxonomia (linear)",
                                     _proc_fusao(("e5", "tfidf", "taxonomia"), modo="linear",
                                                 pesos=(0.8, 0.2, 0.02)),
                                     "a fusão léxico-densa guiada pela compatibilidade de classe"),
    "e5_tfidf_char_lin_taxonomia": Modulo(
        "e5_tfidf_char_lin_taxonomia", "E5 + TF-IDF palavra + caractere + taxonomia (linear)",
        _proc_fusao(("e5", "tfidf", "tfidf_char", "taxonomia"), modo="linear",
                    pesos=(0.7, 0.15, 0.15, 0.02)),
        "a melhor fusão léxico-densa, guiada pela compatibilidade de classe (LOO)"),
}


# ---------------------------------------------------------------------------
# PÓS-PROCESSADORES — os círculos do diagrama
# ---------------------------------------------------------------------------

# As duas regex de medida (unidades e calibre) eram literais aqui, com gauge,
# french e charriere dentro. Agora vêm do perfil: `medidas:` no YAML, com queda
# para a tabela de unidades em uso quando o perfil não declara padrão.


def _pos_nada(corpus: Corpus, textos: Textos, scores: np.ndarray,
              idx: np.ndarray, ctx: dict) -> np.ndarray:
    """Pós-proc 1 (Nada): mantém o ranking do processador."""
    return scores


def _medidas(texto: str) -> set:
    """Conjunto de pares (valor, unidade) normalizados presentes no texto."""
    return perfil_ativo().medidas_de(texto)


def _bonus_unidades(corpus: Corpus, textos: Textos, scores: np.ndarray,
                    idx: np.ndarray, ctx: dict) -> np.ndarray:
    """Equivalência de valores e unidades de medida (18G, 3 ML, 40 MM)."""
    medidas_cat = [_medidas(t) for t in textos.catalogo]
    bonus = np.zeros_like(scores)
    for i in range(idx.shape[0]):
        mq = _medidas(textos.queries[i])
        if not mq:
            continue
        for k, j in enumerate(idx[i]):
            mc = medidas_cat[j]
            if mc:
                bonus[i, k] = len(mq & mc) / len(mq | mc)
    return bonus


def _bonus_pdm(corpus: Corpus, textos: Textos, scores: np.ndarray,
               idx: np.ndarray, ctx: dict) -> np.ndarray:
    """Fração dos termos do PDM do candidato que aparecem na consulta."""
    pdms = [set(str(p or "").upper().split()) for p in corpus.catalogo["pdm"]]
    bonus = np.zeros_like(scores)
    for i in range(idx.shape[0]):
        tokens_q = set(tokenizar(normalizar_texto(textos.queries[i], manter_maiusculas=True)))
        if not tokens_q:
            continue
        for k, j in enumerate(idx[i]):
            pdm = pdms[j]
            if pdm:
                bonus[i, k] = len(pdm & tokens_q) / len(pdm)
    return bonus


def _bonus_graphrag(corpus: Corpus, textos: Textos, scores: np.ndarray,
                    idx: np.ndarray, ctx: dict) -> np.ndarray:
    """
    Cobertura das entidades da consulta pelo candidato, medida no mesmo KG do
    pré-processador GraphRAG — quanto do que a consulta especifica o item do
    catálogo de fato tem. Entidades de atributo (`calibre::18G`) valem o dobro
    de tokens soltos (`TOK::AGULHA`): é o atributo que distingue itens da mesma
    família, e é dentro da família que o ranking se decide.
    """
    from . import graphrag as gr

    kg = ctx.get("_kg_graphrag")
    if kg is None:
        kg = gr.construir_kg(corpus.consultas, corpus.catalogo)
        ctx["_kg_graphrag"] = kg

    def _peso(ent: str) -> float:
        return 1.0 if ent.startswith("TOK::") else 2.0

    bonus = np.zeros_like(scores)
    for i in range(idx.shape[0]):
        proprias = kg.ents_consulta.get(i, set())
        relevantes = {e for e in proprias if not e.startswith("PDM::")} or proprias
        if not relevantes:
            continue
        total = sum(_peso(e) for e in relevantes)
        for k, j in enumerate(idx[i]):
            comuns = relevantes & kg.ents_catalogo.get(int(j), set())
            if comuns:
                bonus[i, k] = sum(_peso(e) for e in comuns) / total
    return bonus


def _bonus_graphrag_idf(corpus: Corpus, textos: Textos, scores: np.ndarray,
                        idx: np.ndarray, ctx: dict) -> np.ndarray:
    """
    Cobertura de entidades como em `_bonus_graphrag`, mas cada entidade pesa o
    seu IDF no catálogo (entidade de atributo ainda vale o dobro). Uma consulta
    coberta em DESCARTAVEL e ESTERIL não está coberta em nada; coberta em
    ORTOFTALALDEIDO, está quase resolvida.
    """
    ctx.setdefault("_corpus", corpus)
    kg = _kg_de(ctx)
    idf = _idf_entidades(kg, ctx)

    def _peso(ent: str) -> float:
        base = 1.0 if ent.startswith("TOK::") else 2.0
        return base * idf.get(ent, 1.0)

    bonus = np.zeros_like(scores)
    for i in range(idx.shape[0]):
        proprias = kg.ents_consulta.get(i, set())
        relevantes = {e for e in proprias if not e.startswith("PDM::")} or proprias
        if not relevantes:
            continue
        total = sum(_peso(e) for e in relevantes)
        if total <= 0:
            continue
        for k, j in enumerate(idx[i]):
            comuns = relevantes & kg.ents_catalogo.get(int(j), set())
            if comuns:
                bonus[i, k] = sum(_peso(e) for e in comuns) / total
    return bonus


def _bonus_taxonomia(corpus: Corpus, textos: Textos, scores: np.ndarray,
                     idx: np.ndarray, ctx: dict) -> np.ndarray:
    """
    Compatibilidade entre a classe e-Fisco da consulta e a classe CATMAT do
    candidato, estimada nos vínculos conhecidos em leave-one-out (ver
    `taxonomia.py`). Sinal ontológico: usa o metadado de classificação dos dois
    catálogos, que o texto do item não carrega.
    """
    ctx.setdefault("_corpus", corpus)
    return np.take_along_axis(_matriz_taxonomia(ctx), idx, axis=1)


def _bonus_taxonomia_rotulo(corpus: Corpus, textos: Textos, scores: np.ndarray,
                            idx: np.ndarray, ctx: dict) -> np.ndarray:
    """
    Alinhamento das taxonomias SÓ pelo texto dos rótulos de classe (E5), sem
    nenhum vínculo conhecido: é o controle do `taxonomia` — quanto do ganho é
    a informação dos vínculos e quanto é só o rótulo.
    """
    from . import taxonomia as tx

    al = ctx.get("_taxonomia_rotulos")
    if al is None:
        al = tx.alinhamento_rotulos(corpus.consultas, corpus.catalogo, MODELO_EMBEDDING)
        ctx["_taxonomia_rotulos"] = al
    return tx.bonus_top_k(al, idx)


def _medidas_do_corpus(corpus: Corpus, ctx: dict) -> tuple[list, list]:
    """Medidas tipadas dos dois lados, extraidas do texto CRU e memorizadas.

    Texto cru, e nao o do pre-processador, porque a normalizacao do pipeline
    remove aspas e apaga a polegada de `3 1/2"` -- uma das medidas que mais
    discriminam no corpus.
    """
    if "_medidas" not in ctx:
        perfil = perfil_ativo()
        med_q = caracteristicas.indice(
            corpus.consultas["item_efisco"].fillna("").astype(str).tolist(), perfil)
        med_c = caracteristicas.indice(
            corpus.catalogo["item_catmat"].fillna("").astype(str).tolist(), perfil)
        r_q, r_c = caracteristicas.resumo(med_q), caracteristicas.resumo(med_c)
        log.info("[pós/medidas] consultas com medida: %d/%d | catálogo: %d/%d | bases: %s",
                 r_q["com_medida"], r_q["total"], r_c["com_medida"], r_c["total"],
                 ", ".join(f"{k}={v}" for k, v in list(r_q["por_base"].items())[:6]))
        ctx["_medidas"] = (med_q, med_c)
        ctx["_medidas_resumo"] = {"consultas": r_q, "catalogo": r_c}
    return ctx["_medidas"]


def _bonus_medidas(corpus: Corpus, textos: Textos, scores: np.ndarray,
                   idx: np.ndarray, ctx: dict) -> np.ndarray:
    """
    Fracao das medidas DA CONSULTA que o candidato satisfaz, em [0,1].

    Mesma maquinaria de mistura do `unidades`, para que a comparacao entre os
    dois isole o que mudou: a EXTRACAO. Onde `unidades` compara o par
    (valor, unidade) como texto -- e por isso nunca casa `3 1/2"` com `90 MM`,
    nem reconhece `G16` --, aqui os dois lados vao para a unidade base da
    categoria antes de comparar, com tolerancia relativa.

    Ancorado na consulta: o denominador e o numero de categorias que a CONSULTA
    enuncia, nao as do candidato. Normalizar pelo candidato pune o item mais
    especifico, que e justamente o certo -- foi o erro que derrubou as variantes
    de atributo categorico (26% x 28% contra o acaso).
    """
    med_q, med_c = _medidas_do_corpus(corpus, ctx)
    tol = ctx.get("tolerancia_medidas", caracteristicas.TOLERANCIA_PADRAO)
    bonus = np.zeros_like(scores)
    for i in range(idx.shape[0]):
        mq = med_q[i]
        if not mq:
            continue
        for k, j in enumerate(idx[i]):
            bonus[i, k] = caracteristicas.satisfacao(mq, med_c[int(j)], tol)
    return bonus


def _pos_medidas_conflito(corpus: Corpus, textos: Textos, scores: np.ndarray,
                          idx: np.ndarray, ctx: dict) -> np.ndarray:
    """
    Variante com PENALIDADE de conflito, somada com sinal ao score.

    Mantida para registro porque a medicao a reprovou: R@3 0,601 contra 0,604 do
    ranking sem pos-processamento. O diagnostico que a motivou olhava so as
    falhas (o sinal acerta 46 contra 9 entre as consultas cujo correto estava no
    top4-10) e nao media o dano nas 401 consultas que ja acertavam no top-1 --
    onde penalizar conflito derruba acerto. Premiar acordo funciona; punir
    divergencia, neste corpus, nao.
    """
    med_q, med_c = _medidas_do_corpus(corpus, ctx)
    peso = ctx.get("peso_medidas", PESO_MEDIDAS_PADRAO)
    tol = ctx.get("tolerancia_medidas", caracteristicas.TOLERANCIA_PADRAO)
    ajustados = scores.copy()
    for i in range(idx.shape[0]):
        mq = med_q[i]
        if not mq:
            continue
        for k, j in enumerate(idx[i]):
            c = caracteristicas.concordancia(mq, med_c[int(j)], tol)
            if c:
                ajustados[i, k] = scores[i, k] + peso * c
    return ajustados


def _misturar(scores: np.ndarray, bonus: np.ndarray, peso: float) -> np.ndarray:
    """Mistura convexa entre o score do processador e o sinal do pós-processador."""
    return (1.0 - peso) * scores + peso * bonus


def _pos_aditivo(*bonus_fns: Callable, pesos: tuple[float, ...] | None = None) -> Callable:
    """
    Monta um pos-processador que SOMA o sinal ao score, com peso pequeno.

    Difere de `_pos_de`, que faz mistura convexa. A diferenca foi medida e
    importa: com peso >= 0,25 na forma convexa o bonus domina a similaridade e o
    ranking vira "ordena por bonus, desempata por E5" -- o que explica 0,25 e
    0,60 darem resultado identico. Somado com peso 0,05, o sinal fica na escala
    do espalhamento do E5 (~0,1 entre o 1o e o 10o colocado) e refina o ranking
    em vez de substitui-lo: R@3 0,6729 contra 0,6601 da forma convexa.

    Aqui `peso_medidas` e o parametro, e nao `peso_pos`, para que ajustar este
    sinal nao mexa nos pos-processadores publicados em RESULTADOS.md.

    Sem `pesos`, os sinais entram pela MEDIA (comportamento original, que e o
    dos numeros publicados). Com `pesos`, cada sinal entra multiplicado pelo
    seu peso relativo -- e o que permite somar um terceiro sinal sem diluir os
    dois que ja funcionavam.

    O peso relativo de cada sinal pode ainda ser ajustado por execucao com
    `ctx["peso_rel_<sinal>"]` (CLI: --peso-rel medidas=1,taxonomia=0.4), onde
    <sinal> e o nome da funcao de bonus sem o prefixo `_bonus_`. E o que a
    varredura usa para achar a escala certa de um sinal novo sem editar codigo.
    """
    nomes = [fn.__name__.removeprefix("_bonus_") for fn in bonus_fns]

    def aplicar(corpus: Corpus, textos: Textos, scores: np.ndarray,
                idx: np.ndarray, ctx: dict) -> np.ndarray:
        peso = ctx.get("peso_medidas", PESO_MEDIDAS_PADRAO)
        relativos = [ctx.get(f"peso_rel_{nome}", 1.0) for nome in nomes]
        if pesos is None:
            total = sum(r * fn(corpus, textos, scores, idx, ctx)
                        for r, fn in zip(relativos, bonus_fns))
            return scores + peso * (total / len(bonus_fns))
        total = sum(p * r * fn(corpus, textos, scores, idx, ctx)
                    for p, r, fn in zip(pesos, relativos, bonus_fns))
        return scores + peso * total
    return aplicar


def _pos_de(*bonus_fns: Callable) -> Callable:
    """
    Monta um pós-processador a partir de um ou mais sinais de bônus.

    Com mais de um, os sinais entram com peso igual dentro da mesma fatia
    `peso_pos` — como a lista `POSTPROCESSORS` com `WEIGHT` da avaliação modular.
    """
    def aplicar(corpus: Corpus, textos: Textos, scores: np.ndarray,
                idx: np.ndarray, ctx: dict) -> np.ndarray:
        bonus = sum(fn(corpus, textos, scores, idx, ctx) for fn in bonus_fns)
        return _misturar(scores, bonus / len(bonus_fns),
                         ctx.get("peso_pos", PESO_POS_PADRAO))
    return aplicar


POS_PROCESSADORES: dict[str, Modulo] = {
    "nada":     Modulo("nada", "Pós-proc 1 (Nada)", _pos_nada,
                       "ranking do processador, sem reordenação"),
    "unidades": Modulo("unidades", "Equivalência de unidades",
                       _pos_de(_bonus_unidades),
                       f"valores + unidades de medida (peso {PESO_POS_PADRAO})"),
    "pdm":      Modulo("pdm", "Coerência de PDM", _pos_de(_bonus_pdm),
                       f"família do catálogo vs. consulta (peso {PESO_POS_PADRAO})"),
    "graphrag": Modulo("graphrag", "GraphRAG (reranking por KG)",
                       _pos_de(_bonus_graphrag),
                       f"cobertura de entidades da consulta (peso {PESO_POS_PADRAO})"),
    "unidades_graphrag": Modulo("unidades_graphrag", "Unidades + GraphRAG",
                                _pos_de(_bonus_unidades, _bonus_graphrag),
                                f"os dois sinais com peso igual (peso {PESO_POS_PADRAO})"),
    "medidas": Modulo("medidas", "Medida tipada (com conversão)",
                      _pos_aditivo(_bonus_medidas),
                      "valor+unidade convertidos à base da categoria, ancorado "
                      f"na consulta (aditivo, peso {PESO_MEDIDAS_PADRAO})"),
    "medidas_graphrag": Modulo("medidas_graphrag", "Medida tipada + GraphRAG",
                               _pos_aditivo(_bonus_medidas, _bonus_graphrag),
                               "medida convertida + cobertura de entidades do KG "
                               "(a melhor combinação medida na grade)"),
    "medidas_unidades": Modulo("medidas_unidades", "Medida tipada + unidades",
                               _pos_aditivo(_bonus_medidas, _bonus_unidades),
                               "medida convertida e equivalência textual de unidade"),
    "medidas_conflito": Modulo("medidas_conflito", "Medida tipada, com penalidade",
                               _pos_medidas_conflito,
                               "variante reprovada na medição; ver docstring"),
    # --- GraphRAG com IDF -------------------------------------------------------
    "graphrag_idf": Modulo("graphrag_idf", "GraphRAG (cobertura ponderada por IDF)",
                           _pos_de(_bonus_graphrag_idf),
                           f"cobertura de entidades, cada uma pesando seu IDF (peso {PESO_POS_PADRAO})"),
    "graphrag_idf_aditivo": Modulo("graphrag_idf_aditivo", "GraphRAG IDF (aditivo)",
                                   _pos_aditivo(_bonus_graphrag_idf),
                                   "a mesma cobertura IDF, somada com peso pequeno em vez de "
                                   "misturada (isola o efeito da forma de combinar)"),
    "medidas_graphrag_idf": Modulo("medidas_graphrag_idf", "Medida tipada + GraphRAG IDF",
                                   _pos_aditivo(_bonus_medidas, _bonus_graphrag_idf),
                                   "medida convertida + cobertura IDF (aditivo)"),
    # --- taxonomia (ontologia de classes dos dois catálogos) ---------------------
    "taxonomia": Modulo("taxonomia", "Taxonomia (classe e-Fisco -> classe CATMAT, LOO)",
                        _pos_aditivo(_bonus_taxonomia),
                        "compatibilidade de classe estimada nos vínculos, leave-one-out "
                        f"(aditivo, peso {PESO_MEDIDAS_PADRAO})"),
    "taxonomia_rotulo": Modulo("taxonomia_rotulo", "Taxonomia só por rótulo (E5)",
                               _pos_aditivo(_bonus_taxonomia_rotulo),
                               "alinhamento das classes pelo texto do rótulo, sem histórico "
                               "(controle do `taxonomia`)"),
    "medidas_taxonomia": Modulo("medidas_taxonomia", "Medida tipada + taxonomia",
                                _pos_aditivo(_bonus_medidas, _bonus_taxonomia, pesos=(1.0, 1.0)),
                                "os dois sinais somados, cada um com peso inteiro"),
    "medidas_graphrag_taxonomia": Modulo(
        "medidas_graphrag_taxonomia", "Medida + GraphRAG + taxonomia",
        _pos_aditivo(_bonus_medidas, _bonus_graphrag, _bonus_taxonomia, pesos=(1.0, 1.0, 1.0)),
        "medida convertida + cobertura de entidades + compatibilidade de classe"),
    "medidas_graphrag_idf_taxonomia": Modulo(
        "medidas_graphrag_idf_taxonomia", "Medida + GraphRAG IDF + taxonomia",
        _pos_aditivo(_bonus_medidas, _bonus_graphrag_idf, _bonus_taxonomia, pesos=(1.0, 1.0, 1.0)),
        "idem, com a cobertura ponderada por IDF"),
    # Pesos relativos calibrados por varredura no bigdata_profs (ver
    # docs/RESULTADOS-BIGDATA-PROFS.md): a medida quer peso menor que no MMH, a
    # cobertura IDF quer o peso inteiro e a taxonomia entra como desempate.
    "combinado_calibrado": Modulo(
        "combinado_calibrado", "Medida 0,6 + GraphRAG IDF 1,0 + taxonomia 0,3",
        _pos_aditivo(_bonus_medidas, _bonus_graphrag_idf, _bonus_taxonomia, pesos=(0.6, 1.0, 0.3)),
        "os três sinais com pesos relativos calibrados (x peso_medidas)"),
}


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def avaliar(corpus: Corpus, idx: np.ndarray, ks=KS_AVALIACAO) -> dict:
    """
    Métricas de recuperação contra o gabarito.

    MRR        : média de 1/posição do primeiro acerto
    recall@k   : fração de consultas com o CATMAT correto no top-k
    precisão@1 : fração de consultas cujo top-1 está correto
    """
    codigos_cat = corpus.catalogo["codigo_catmat"].tolist()
    codigos_q = corpus.consultas["codigo_efisco"].tolist()

    acertos = {k: 0 for k in ks}
    rr_total = 0.0
    n_aval = 0

    for i, cod_e in enumerate(codigos_q):
        alvos = corpus.gold.get(cod_e)
        if not alvos:
            continue
        n_aval += 1
        pos = next((p for p, j in enumerate(idx[i]) if codigos_cat[j] in alvos), None)
        if pos is None:
            continue
        rr_total += 1.0 / (pos + 1)
        for k in ks:
            if pos < k:
                acertos[k] += 1

    resultado = {
        "n_avaliados": n_aval,
        "mrr": round(rr_total / max(n_aval, 1), 4),
        **{f"recall_at_{k}": round(acertos[k] / max(n_aval, 1), 4) for k in ks},
    }
    resultado["precisao_at_1"] = resultado.get("recall_at_1", 0.0)
    return resultado


# ---------------------------------------------------------------------------
# Execução da grade
# ---------------------------------------------------------------------------

def executar_grade(corpus: Corpus, ids_pre: list[str], ids_proc: list[str],
                   ids_pos: list[str], top_k: int = K_CANDIDATOS,
                   ctx: dict | None = None) -> list[dict]:
    """
    Produto cartesiano pré × proc × pós.

    O pré-processamento é calculado uma vez por pré-processador e o ranking do
    processador uma vez por célula (pré, proc) — os pós-processadores apenas
    reordenam o top-K, como no diagrama.
    """
    ctx = ctx if ctx is not None else {}
    # Processadores híbridos e o sinal de taxonomia precisam do corpus (o
    # contrato `fn(textos, top_k, ctx)` não o recebe): fica no contexto.
    ctx["_corpus"] = corpus
    combinacoes: list[dict] = []
    total = len(ids_pre) * len(ids_proc) * len(ids_pos)
    log.info("Grade: %d pré × %d proc × %d pós = %d combinações.",
             len(ids_pre), len(ids_proc), len(ids_pos), total)

    for id_pre in ids_pre:
        mod_pre = PRE_PROCESSADORES[id_pre]
        t0 = time.time()
        try:
            textos = _pre_com_cache(mod_pre, corpus, ctx)
            tempo_pre = time.time() - t0
            erro_pre = None
        except Exception as exc:   # um módulo quebrado não derruba a grade inteira
            tempo_pre = time.time() - t0
            erro_pre = f"{type(exc).__name__}: {exc}"
            log.warning("[pré/%s] falhou: %s", id_pre, erro_pre)
            textos = None

        for id_proc in ids_proc:
            mod_proc = PROCESSADORES[id_proc]

            if textos is None:
                for id_pos in ids_pos:
                    combinacoes.append(_celula_vazia(id_pre, id_proc, id_pos, erro_pre))
                continue

            t1 = time.time()
            try:
                scores, idx = mod_proc.fn(textos, top_k, ctx)
                tempo_proc = time.time() - t1
            except Exception as exc:
                erro = f"{type(exc).__name__}: {exc}"
                log.warning("[proc/%s] falhou (pré=%s): %s", id_proc, id_pre, erro)
                for id_pos in ids_pos:
                    combinacoes.append(_celula_vazia(id_pre, id_proc, id_pos, erro))
                continue

            for id_pos in ids_pos:
                mod_pos = POS_PROCESSADORES[id_pos]
                t2 = time.time()
                try:
                    scores_pos = mod_pos.fn(corpus, textos, scores, idx, ctx)
                except Exception as exc:
                    erro = f"{type(exc).__name__}: {exc}"
                    log.warning("[pós/%s] falhou (pré=%s, proc=%s): %s",
                                id_pos, id_pre, id_proc, erro)
                    combinacoes.append(_celula_vazia(id_pre, id_proc, id_pos, erro))
                    continue
                ordem = np.argsort(-scores_pos, axis=1, kind="stable")
                idx_pos = np.take_along_axis(idx, ordem, axis=1)
                tempo_pos = time.time() - t2

                metricas = avaliar(corpus, idx_pos)
                combinacoes.append({
                    "pre": id_pre, "proc": id_proc, "pos": id_pos,
                    "combinacao": f"{id_pre} + {id_proc} + {id_pos}",
                    "metricas": metricas,
                    "tempo_s": {"pre": round(tempo_pre, 2),
                                "proc": round(tempo_proc, 2),
                                "pos": round(tempo_pos, 2)},
                    "meta_pre": textos.meta,
                    "erro": None,
                })
                log.info("  %-14s | %-9s | %-9s -> MRR %.4f  R@1 %.3f  R@3 %.3f",
                         id_pre, id_proc, id_pos, metricas["mrr"],
                         metricas["recall_at_1"], metricas["recall_at_3"])

    return combinacoes


def _assinatura_pre(mod: Modulo, corpus: Corpus) -> str:
    """
    Assinatura do pré-processador: identidade + corpus + código-fonte relevante.

    Inclui o fonte para o cache não sobreviver a uma mudança de lógica. Fica de
    fora o resto deste arquivo, para editar gráfico ou CLI não invalidar textos
    caros de recomputar.
    """
    ctx = contexto()
    partes = [mod.id, corpus.arquivo, ctx.dataset.nome, ctx.perfil.nome,
              str(len(corpus.consultas)), str(len(corpus.catalogo))]
    for fn in (mod.fn, _pre_basico, _pre_graphrag):
        try:
            partes.append(inspect.getsource(fn))
        except OSError:
            pass
    for nome in ("preprocessamento.py", "rede_semantica.py", "graphrag.py", "config.py"):
        caminho = BASE_DIR / "catalogo_match" / nome
        if caminho.exists():
            partes.append(caminho.read_text(encoding="utf-8"))
    # O YAML do perfil entra na assinatura: mudar uma stopword muda os textos,
    # e um cache que sobrevivesse a isso devolveria resultado de outro domínio.
    # A cadeia de herança inteira entra (compras_publicas herda mmh): editar o
    # pai também muda os textos do filho.
    nome = ctx.perfil.nome
    vistos: list[str] = []
    while nome and nome not in vistos:
        vistos.append(nome)
        caminho = BASE_DIR / "config" / "perfis" / f"{nome}.yaml"
        if not caminho.exists():
            break
        conteudo = caminho.read_text(encoding="utf-8")
        partes.append(conteudo)
        m = re.search(r"^herda:\s*(\S+)", conteudo, re.MULTILINE)
        nome = m.group(1).strip("'\"") if m else "base"
    if "base" not in vistos:
        caminho = BASE_DIR / "config" / "perfis" / "base.yaml"
        if caminho.exists():
            partes.append(caminho.read_text(encoding="utf-8"))
    return hashlib.sha1("|".join(partes).encode("utf-8")).hexdigest()[:16]


def _pre_com_cache(mod: Modulo, corpus: Corpus, ctx: dict) -> Textos:
    """
    Executa o pré-processador, guardando os textos em disco.

    Além de economizar tempo, isso torna a grade reprodutível: a construção do
    grafo léxico da fase [1] tem passo estocástico, então sem cache a linha
    `rede_semantica` muda de uma execução para outra.
    """
    arquivo = dir_cache_embeddings() / f"pre_{mod.id}_{_assinatura_pre(mod, corpus)}.json"

    if arquivo.exists() and not ctx.get("sem_cache"):
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        log.info("[pré/%s] Textos reaproveitados do cache: %s", mod.id, arquivo.name)
        return Textos(pre_id=mod.id, queries=dados["queries"],
                      catalogo=dados["catalogo"], meta=dados["meta"])

    textos = mod.fn(corpus, ctx)
    arquivo.write_text(
        json.dumps({"queries": textos.queries, "catalogo": textos.catalogo,
                    "meta": textos.meta}, ensure_ascii=False),
        encoding="utf-8",
    )
    return textos


def _celula_vazia(id_pre: str, id_proc: str, id_pos: str, erro: str) -> dict:
    return {
        "pre": id_pre, "proc": id_proc, "pos": id_pos,
        "combinacao": f"{id_pre} + {id_proc} + {id_pos}",
        "metricas": None, "tempo_s": None, "meta_pre": None, "erro": erro,
    }


# ---------------------------------------------------------------------------
# Saídas: matriz, ranking e YAML
# ---------------------------------------------------------------------------

def montar_matriz(combinacoes: list[dict], ids_pre: list[str], ids_proc: list[str],
                  metrica: str = "mrr") -> pd.DataFrame:
    """Matriz do diagrama: linhas = pré-processadores, colunas = processadores.

    As células usam o pós-processador neutro (`nada`) quando ele está na grade —
    é o estado do ranking antes do pós-processamento, como na figura.
    """
    base = "nada" if any(c["pos"] == "nada" for c in combinacoes) else None
    matriz = pd.DataFrame(index=ids_pre, columns=ids_proc, dtype=float)
    for c in combinacoes:
        if base is not None and c["pos"] != base:
            continue
        if c["metricas"] is not None:
            matriz.loc[c["pre"], c["proc"]] = c["metricas"].get(metrica, np.nan)
    return matriz


def imprimir_matriz(matriz: pd.DataFrame, metrica: str) -> None:
    """Renderiza a matriz no terminal, marcando as três melhores células."""
    valores = matriz.stack().dropna() if not matriz.empty else pd.Series(dtype=float)
    melhores = set(valores.nlargest(3).index) if len(valores) else set()

    largura = max(14, *(len(c) + 2 for c in matriz.columns)) if len(matriz.columns) else 14
    print(f"\n=== MATRIZ pré × proc ({metrica}, sem pós-processamento) ===")
    print(" " * 16 + "".join(f"{c:>{largura}}" for c in matriz.columns))
    for pre in matriz.index:
        linha = f"{pre:<16}"
        for proc in matriz.columns:
            v = matriz.loc[pre, proc]
            if pd.isna(v):
                celula = "n/d"
            else:
                celula = f"{v:.4f}" + ("*" if (pre, proc) in melhores else "")
            linha += f"{celula:>{largura}}"
        print(linha)
    if melhores:
        print("  (*) três melhores células")


def imprimir_ranking(combinacoes: list[dict], metrica: str, limite: int = 10) -> list[dict]:
    """Rank das melhores combinações — o bloco final do diagrama."""
    validas = [c for c in combinacoes if c["metricas"] is not None]
    ordenadas = sorted(validas, key=lambda c: -c["metricas"].get(metrica, 0.0))

    print(f"\n=== RANK DAS MELHORES COMBINAÇÕES (por {metrica}) ===")
    print(f"{'#':<4}{'combinação':<46}{metrica:>10}{'R@1':>8}{'R@3':>8}{'R@10':>8}{'s':>8}")
    for posicao, c in enumerate(ordenadas[:limite], start=1):
        m, t = c["metricas"], c["tempo_s"]
        print(f"{posicao:<4}{c['combinacao']:<46}{m.get(metrica, 0):>10.4f}"
              f"{m['recall_at_1']:>8.3f}{m['recall_at_3']:>8.3f}{m['recall_at_10']:>8.3f}"
              f"{sum(t.values()):>8.1f}")

    invalidas = [c for c in combinacoes if c["metricas"] is None]
    if invalidas:
        print(f"\n  {len(invalidas)} combinações sem valor (dependência ausente):")
        for c in invalidas[:5]:
            print(f"    - {c['combinacao']}: {c['erro']}")
        if len(invalidas) > 5:
            print(f"    ... e mais {len(invalidas) - 5}")

    return ordenadas


# Rampa sequencial de um só matiz (azul, claro -> escuro): a célula codifica
# magnitude, então a cor varia em luminosidade, nunca em matiz.
_RAMPA_SEQUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                     "#256abf", "#184f95", "#0d366b"]
_SUPERFICIE = "#fcfcfb"
_TINTA = "#0b0b0b"
_TINTA_MUTED = "#898781"


def plotar_matriz(matriz: pd.DataFrame, metrica: str,
                  caminho: Path | None = None, subtitulo: str = "") -> Path:
    """Heatmap da matriz pré × proc — a figura do diagrama, para apresentação."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    caminho = caminho or (dir_resultados() / "matriz_modular.png")
    dados = matriz.astype(float)
    valores = dados.values
    n_lin, n_col = valores.shape

    cmap = LinearSegmentedColormap.from_list("mmh_seq", _RAMPA_SEQUENCIAL)
    vmin, vmax = np.nanmin(valores), np.nanmax(valores)
    piso = vmin - 0.12 * (vmax - vmin)

    fig, ax = plt.subplots(figsize=(1.7 * n_col + 3.4, 0.78 * n_lin + 2.6), dpi=200)
    fig.patch.set_facecolor(_SUPERFICIE)
    ax.set_facecolor(_SUPERFICIE)

    malha = ax.pcolormesh(valores, cmap=cmap, vmin=piso, vmax=vmax,
                          edgecolors=_SUPERFICIE, linewidth=2)

    melhor_por_coluna = np.nanargmax(valores, axis=0)
    melhor_global = np.unravel_index(np.nanargmax(valores), valores.shape)

    def _tinta_sobre(cor) -> str:
        """Escolhe a tinta pelo brilho real da célula (luminância relativa)."""
        canais = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                  for c in cor[:3]]
        lum = 0.2126 * canais[0] + 0.7152 * canais[1] + 0.0722 * canais[2]
        return _TINTA if lum > 0.179 else _SUPERFICIE

    for i in range(n_lin):
        for j in range(n_col):
            v = valores[i, j]
            if np.isnan(v):
                ax.text(j + 0.5, i + 0.5, "n/d", ha="center", va="center",
                        fontsize=10, color=_TINTA_MUTED)
                continue
            cor_celula = cmap((v - piso) / max(vmax - piso, 1e-9))
            ax.text(j + 0.5, i + 0.5, f"{v:.3f}", ha="center", va="center",
                    fontsize=11, color=_tinta_sobre(cor_celula),
                    fontweight="bold" if melhor_por_coluna[j] == i else "normal")

    # Melhor combinação: anel, não cor — a cor já está ocupada pela magnitude.
    ax.add_patch(Rectangle((melhor_global[1], melhor_global[0]), 1, 1,
                           fill=False, edgecolor=_TINTA, linewidth=2.2, zorder=5))

    ax.set_xticks(np.arange(n_col) + 0.5)
    ax.set_xticklabels(dados.columns, fontsize=11, color=_TINTA)
    ax.set_yticks(np.arange(n_lin) + 0.5)
    ax.set_yticklabels(dados.index, fontsize=11, color=_TINTA)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0)
    ax.invert_yaxis()
    for lado in ax.spines.values():
        lado.set_visible(False)

    barra = fig.colorbar(malha, ax=ax, fraction=0.030, pad=0.02)
    barra.outline.set_visible(False)
    barra.ax.tick_params(length=0, labelsize=9, colors=_TINTA_MUTED)
    barra.set_label(metrica.upper(), fontsize=10, color=_TINTA_MUTED)

    # Espaço reservado no topo: rótulos das colunas + título + subtítulo. As
    # posições são em POLEGADAS a partir do topo, e não frações da altura:
    # com poucas linhas a figura é baixa e frações fixas faziam o título
    # cair em cima do subtítulo.
    altura = fig.get_size_inches()[1]
    y_titulo = 1 - 0.12 / altura          # topo do título a 0,12" da borda
    y_sub = 1 - 0.48 / altura             # topo do subtítulo a 0,48" (título tem ~0,25")
    fig.subplots_adjust(top=1 - 1.0 / altura, left=0.20, right=0.94, bottom=0.06)
    fig.suptitle(f"Matriz pré-processador × processador — {metrica.upper()}",
                 fontsize=15, color=_TINTA, x=0.02, ha="left", y=y_titulo, va="top")
    legenda = "linhas = pré-processador · colunas = processador"
    fig.text(0.02, y_sub, f"{legenda}{' · ' + subtitulo if subtitulo else ''}",
             fontsize=9.5, color=_TINTA_MUTED, ha="left", va="top")

    fig.savefig(caminho, bbox_inches="tight", facecolor=_SUPERFICIE)
    plt.close(fig)
    return caminho


def exportar(corpus: Corpus, combinacoes: list[dict], ordenadas: list[dict],
             matriz: pd.DataFrame, metrica: str, top_k: int,
             ids_pre: list[str], ids_proc: list[str], ids_pos: list[str],
             ctx: dict | None = None) -> None:
    """Grava o YAML da grade + a matriz e o ranking em CSV."""
    ctx = ctx or {}

    documento = {
        "INFO": {
            "DATA": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ARQUITETURA": "pré-proc x processador x pós-proc (pipeline.jpg)",
            "METRICA_MATRIZ": metrica,
            "TOP_K_CANDIDATOS": top_k,
            "DADOS": {
                "DATASET": corpus.dataset,
                "GROUND_TRUTH": corpus.arquivo,
                "N_CONSULTAS": len(corpus.consultas),
                "N_CATALOGO": len(corpus.catalogo),
                "BLOCKING": "nenhum — catálogo inteiro por consulta",
            },
            # Proveniência do conhecimento de domínio: quanto deste resultado
            # vem de curadoria humana e quanto foi induzido do corpus.
            "PERFIL": perfil_ativo().resumo(),
            "MODULOS": {
                "PRE": [{"ID": i, "NOME": PRE_PROCESSADORES[i].nome,
                         "DESCRICAO": PRE_PROCESSADORES[i].descricao} for i in ids_pre],
                "PROC": [{"ID": i, "NOME": PROCESSADORES[i].nome,
                          "DESCRICAO": PROCESSADORES[i].descricao} for i in ids_proc],
                "POS": [{"ID": i, "NOME": POS_PROCESSADORES[i].nome,
                         "DESCRICAO": POS_PROCESSADORES[i].descricao} for i in ids_pos],
            },
        },
        "META_PRE": {
            id_pre: next((c["meta_pre"] for c in combinacoes
                          if c["pre"] == id_pre and c["meta_pre"]), None)
            for id_pre in ids_pre
        },
        # O que os sinais de pós-processamento viram no corpus: cobertura do
        # extrator de medidas e do alinhamento taxonômico. Sem isto um R@3 não
        # diz se o sinal estava ativo ou calado.
        "META_SINAIS": {
            "MEDIDAS": ctx.get("_medidas_resumo"),
            "TAXONOMIA": ctx.get("_taxonomia_resumo"),
        },
        "MATRIZ": {
            pre: {proc: (None if pd.isna(matriz.loc[pre, proc])
                         else float(matriz.loc[pre, proc]))
                  for proc in matriz.columns}
            for pre in matriz.index
        },
        "RANKING": [
            {"POSICAO": p, "COMBINACAO": c["combinacao"],
             "PRE": c["pre"], "PROC": c["proc"], "POS": c["pos"],
             "METRICAS": c["metricas"], "TEMPO_S": c["tempo_s"]}
            for p, c in enumerate(ordenadas, start=1)
        ],
        "SEM_VALOR": [
            {"COMBINACAO": c["combinacao"], "MOTIVO": c["erro"]}
            for c in combinacoes if c["metricas"] is None
        ],
    }

    caminho_yaml = dir_resultados() / "avaliacao_modular.yaml"
    with open(caminho_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(documento, fh, allow_unicode=True, sort_keys=False, width=100)

    caminho_matriz = dir_resultados() / "matriz_pre_x_proc.csv"
    matriz.to_csv(caminho_matriz, encoding="utf-8-sig")

    caminho_rank = dir_resultados() / "ranking_combinacoes.csv"
    pd.DataFrame([
        {"posicao": p, "combinacao": c["combinacao"], "pre": c["pre"],
         "proc": c["proc"], "pos": c["pos"], **c["metricas"],
         "tempo_total_s": round(sum(c["tempo_s"].values()), 2)}
        for p, c in enumerate(ordenadas, start=1)
    ]).to_csv(caminho_rank, index=False, encoding="utf-8-sig")

    gerados = [caminho_yaml, caminho_matriz, caminho_rank]
    try:
        gerados.append(plotar_matriz(
            matriz, metrica,
            subtitulo=(f"{len(corpus.consultas)} consultas e-Fisco × "
                       f"{len(corpus.catalogo)} itens CATMAT · {corpus.arquivo}"),
        ))
    except Exception as exc:
        log.warning("Heatmap não gerado (%s): %s", type(exc).__name__, exc)

    print("\n=== ARQUIVOS GERADOS ===")
    for caminho in gerados:
        print(f"  {caminho}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _listar_modulos() -> None:
    for titulo, registro in (("PRÉ-PROCESSADORES", PRE_PROCESSADORES),
                             ("PROCESSADORES", PROCESSADORES),
                             ("PÓS-PROCESSADORES", POS_PROCESSADORES)):
        print(f"\n{titulo}")
        for mod in registro.values():
            print(f"  {mod.id:<16} {mod.nome:<32} {mod.descricao}")


def _selecionar(pedidos: str | None, registro: dict[str, Modulo], eixo: str) -> list[str]:
    if not pedidos:
        return list(registro)
    ids = [p.strip() for p in pedidos.split(",") if p.strip()]
    desconhecidos = [i for i in ids if i not in registro]
    if desconhecidos:
        raise SystemExit(
            f"{eixo} desconhecido(s): {', '.join(desconhecidos)}. "
            f"Disponíveis: {', '.join(registro)}"
        )
    return ids


def main() -> None:
    from .config import ativar, listar_datasets, listar_perfis

    ap = argparse.ArgumentParser(
        description="Grade modular pré × proc × pós para o casamento CATMAT <-> e-Fisco."
    )
    ap.add_argument("--dataset", default="",
                    help=f"dataset a avaliar. Disponíveis: {', '.join(listar_datasets())}")
    ap.add_argument("--perfil", default="",
                    help=f"força outro perfil de domínio ({', '.join(listar_perfis())}). "
                         "Usar `base` mede o pipeline sem conhecimento curado.")
    ap.add_argument("--pre", help="pré-processadores (csv de ids)")
    ap.add_argument("--proc", help="processadores (csv de ids)")
    ap.add_argument("--pos", help="pós-processadores (csv de ids)")
    ap.add_argument("--metrica", default="mrr",
                    help="métrica das células e do ranking (padrão: mrr)")
    ap.add_argument("--amostra", type=int, help="usa só N consultas (execução rápida)")
    ap.add_argument("--top-k", type=int, default=K_CANDIDATOS,
                    help=f"candidatos retidos por consulta (padrão: {K_CANDIDATOS})")
    ap.add_argument("--arquivo", default="",
                    help="chave lógica do arquivo no dataset (padrão: a do YAML)")
    ap.add_argument("--peso-pos", type=float, default=PESO_POS_PADRAO,
                    help=f"peso do pós-processador na mistura (padrão: {PESO_POS_PADRAO})")
    ap.add_argument("--peso-medidas", type=float, default=PESO_MEDIDAS_PADRAO,
                    help=f"peso do ajuste de medida tipada (padrão: {PESO_MEDIDAS_PADRAO})")
    ap.add_argument("--tolerancia-medidas", type=float,
                    default=caracteristicas.TOLERANCIA_PADRAO,
                    help="tolerância relativa no casamento de medidas "
                         f"(padrão: {caracteristicas.TOLERANCIA_PADRAO})")
    ap.add_argument("--k-cross", type=int, default=K_CROSS_RERANK,
                    help=f"profundidade do cross-encoder (padrão: {K_CROSS_RERANK})")
    ap.add_argument("--peso-fusao", default="",
                    help="pesos por fonte nos processadores híbridos, ex. 'tfidf=0.3,kg=0.1'")
    ap.add_argument("--peso-rel", default="",
                    help="pesos relativos por sinal nos pós aditivos, ex. "
                         "'medidas=1,taxonomia=0.4,graphrag_idf=1' (padrão: todos 1)")
    ap.add_argument("--taxonomia-peso-grupo", type=float,
                    help="peso do nível de grupo no sinal de taxonomia quando a classe é "
                         "conhecida (padrão: 0.3; 0 desliga o backoff parcial)")
    ap.add_argument("--taxonomia-min-obs", type=int,
                    help="vínculos mínimos da classe e-Fisco para o sinal opinar (padrão: 2)")
    ap.add_argument("--graphrag-ms", action="store_true",
                    help="usa o índice Microsoft GraphRAG na expansão (requer API key)")
    ap.add_argument("--sem-cache", action="store_true",
                    help="ignora os textos cacheados dos pré-processadores e recalcula")
    ap.add_argument("--listar", action="store_true", help="lista os módulos e sai")
    args = ap.parse_args()

    if args.listar:
        _listar_modulos()
        return

    ativar(args.dataset, perfil=args.perfil)

    ids_pre = _selecionar(args.pre, PRE_PROCESSADORES, "pré-processador")
    ids_proc = _selecionar(args.proc, PROCESSADORES, "processador")
    ids_pos = _selecionar(args.pos, POS_PROCESSADORES, "pós-processador")

    ctx = {
        "peso_pos": args.peso_pos,
        "peso_medidas": args.peso_medidas,
        "tolerancia_medidas": args.tolerancia_medidas,
        "k_cross": args.k_cross,
        "sem_cache": args.sem_cache,
        "graphrag_ms": args.graphrag_ms,
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
    }
    for par in filter(None, (p.strip() for p in args.peso_rel.split(","))):
        nome, _, valor = par.partition("=")
        ctx[f"peso_rel_{nome.strip()}"] = float(valor)
    for par in filter(None, (p.strip() for p in args.peso_fusao.split(","))):
        nome, _, valor = par.partition("=")
        ctx[f"peso_fusao_{nome.strip()}"] = float(valor)
    if args.taxonomia_peso_grupo is not None:
        ctx["taxonomia_peso_grupo"] = args.taxonomia_peso_grupo
    if args.taxonomia_min_obs is not None:
        ctx["taxonomia_min_obs"] = args.taxonomia_min_obs

    corpus = carregar_corpus(args.arquivo, amostra=args.amostra)
    inicio = time.time()
    combinacoes = executar_grade(corpus, ids_pre, ids_proc, ids_pos,
                                 top_k=args.top_k, ctx=ctx)

    matriz = montar_matriz(combinacoes, ids_pre, ids_proc, args.metrica)
    imprimir_matriz(matriz, args.metrica)
    ordenadas = imprimir_ranking(combinacoes, args.metrica)
    exportar(corpus, combinacoes, ordenadas, matriz, args.metrica,
             args.top_k, ids_pre, ids_proc, ids_pos, ctx=ctx)

    print(f"\nTempo total: {time.time() - inicio:.1f}s")


if __name__ == "__main__":
    main()
