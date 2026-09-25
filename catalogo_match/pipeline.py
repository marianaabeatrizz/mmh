"""
pipeline.py
===========
Implementação integral do fluxo fim a fim descrito em:
  "Rede Semântica e Ontologia para o casamento CATMAT <-> e-Fisco"
  Seção 4.3 — Fases [0] a [9]

[0] Fontes           — ingestão; separa CATÁLOGO (CATMAT) de CONSULTAS (e-Fisco)
[1] Rede semântica   — normalização + expansão + desambiguação + ancoragem
[2] Extração         — texto livre -> atributos PDM (LLM; regex como fallback);
                       validação de esquema SHACL (pyshacl) contra as
                       características definidoras por família (§2.2)
[3] Blocking         — GERAÇÃO DE CANDIDATOS por Classe/PDM + subsunção + top-50
                       pela similaridade FUNDIDA (E5 + TF-IDF palavra + caractere)
[4] Ontologia        — OWL + reasoner Pellet (SWRL da §2.2) sobre os candidatos
[5] Matcher neural   — estágio 1 = fusão léxico-densa; estágio 2 = cross-encoder
                       (opcional, --sem-cross: medido como prejudicial)
[5b] Sinais          — medida tipada + cobertura de entidades (IDF) + taxonomia
                       LOO somados ao score neural (peso 0,05)
[6] GraphRAG         — adjudicação ancorada no grafo + LLM na zona cinzenta, da
                       menor margem para a maior (--max-adjudicacoes 0 desliga)
[7] Grafo unificado  — itens + PDMs + arestas :equivalenteA (só as que passam)
[8] Confiança        — ranqueia candidatos, escolhe o top-1, Alta/Média/Baixa
[*] Avaliação        — recall@k, MRR, precisão do top-1, recall do blocking
[9] Análise global   — comunidades, deduplicação, OOV, anomalias

DIFERENÇA CENTRAL para a versão anterior
----------------------------------------
A versão anterior pontuava apenas a DIAGONAL do gabarito: cada item e-Fisco era
comparado somente com o CATMAT que já estava na mesma linha. Não havia geração
de candidatos, logo não havia recuperação, ranking, precisão nem F1 — apenas
cobertura@limiar sobre pares que já se sabiam corretos.

Agora a fase [3] gera candidatos de verdade a partir do catálogo inteiro, e as
fases [4]–[8] operam sobre esses candidatos. O gabarito passa a ser usado só
para AVALIAR, nunca para construir o conjunto avaliado.

MULTI-DATASET
-------------
Nenhuma fase conhece o domínio: o que é conhecimento (léxico, famílias,
unidades, esquema de atributos, papéis nos prompts) vem do PERFIL, e o que é
corpus (arquivos, separador, saídas) vem do DATASET. Ver catalogo_match/config.py
e config/{datasets,perfis}/.

A avaliação reporta, além das métricas do ranking final, a ABLAÇÃO por camada:
o mesmo top-K reordenado por cosseno E5, fusão, cross-encoder, sinais, fase [6]
e score final — o que cada fase acrescentou, medido numa só execução.

Uso:
    python -m catalogo_match.pipeline                       # dataset padrão
    python -m catalogo_match.pipeline --dataset mmh_opme    # outro corpus
    python -m catalogo_match.pipeline --perfil base         # sem léxico curado
    python -m catalogo_match.pipeline --smoke               # testa a conexão
    python -m catalogo_match.pipeline --sem-llm             # offline (regex do perfil)
    python -m catalogo_match.pipeline --sem-rede            # ablação §3.10
    python -m catalogo_match.pipeline --dataset bigdata_profs   # padrões = a configuração medida
                                                                # como melhor (fusão, sinais, sem
                                                                # cross, adjudicação em lista)
    python -m catalogo_match.pipeline --adjudicacao par --contexto-ms --cross   # o desenho antigo
    python -m catalogo_match.pipeline --saida teste --peso-ontologia 0.35 --fator-veto 0.25
"""

import re
import os
import ast
import sys
import json
import time
import yaml
import hashlib
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

# Forçar saída UTF-8 no terminal Windows (evita UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from rdflib import Graph, Namespace, RDF, Literal, BNode
from rdflib.namespace import SH
import pyshacl

# Módulos locais do projeto
from .config import contexto, perfil_ativo
from .preprocessamento import (
    carregar_dados,
    normalizar_texto,
    preprocessar_texto_efisco,
    preprocessar_texto_catmat,
    extrair_atributos_catmat,
    tokenizar,
    remover_stopwords,
)
from .rede_semantica import (
    construir_grafo,
    normalizar_termo,
    expandir_por_ativacao,
    ancorar_pdm,
    ancorar_registro,
    exportar_graphml,
    exportar_skos,
    exportar_fila_curadoria,
    visualizar_grafo,
    pdm_mais_frequente,
    dir_resultados,
    _id_no,
    HOJE,
)
from .ontologia_owl import (
    caracteristicas_definidoras,
    iri_ontologia,
    _familia_de,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# O `.env` (OPENAI_API_KEY) é carregado por `config.py` no import do pacote,
# para que grade, varredura e serviço vejam a mesma chave que o pipeline.
from .config import carregar_dotenv as _carregar_dotenv

_carregar_dotenv()

# --- Modelos ---------------------------------------------------------------

# Bi-encoder ASSIMÉTRICO (§4.3 [5] / camada "Costura fuzzy"): E5 multilingual.
# Prefixos "query:" / "passage:" casam texto curto-e-sujo (e-Fisco = query) com
# entrada estruturada (CATMAT = passage) — exatamente o cenário do documento.
MODELO_EMBEDDING = "intfloat/multilingual-e5-base"
_USA_PREFIXO_E5 = "e5" in MODELO_EMBEDDING.lower()

# Cross-encoder treinado em mMARCO (inclui português) para re-ranking
MODELO_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

LLM_MODEL_PADRAO = "gpt-4o-mini"

# --- Parâmetros de recuperação --------------------------------------------

# Estágio 1 (barato): bi-encoder sobre TODOS os candidatos do bloco.
# Estágio 2 (caro): cross-encoder só sobre os K melhores de cada consulta.
K_RERANK          = 20     # candidatos que chegam ao cross-encoder e aos sinais
K_AVALIACAO       = 10     # maior k reportado em recall@k
MAX_CANDIDATOS    = 400    # teto por consulta (protege blocos gigantes)
_N_SEM_BLOCKING   = 50    # candidatos cross-PDM por similaridade fundida (estratégia 6)

# --- Fusão léxico-densa (grade modular: processador `e5_tfidf_char_lin`) ----
# A similaridade que guia o blocking (estratégia 6), o corte dos blocos grandes
# e o estágio 1 do matcher é a soma de cossenos E5 + TF-IDF de palavra + TF-IDF
# de caractere. Medido na grade sem blocking: R@3 de 74,3% (E5) para 81,9% no
# bigdata_profs e de 60,7% para 72,1% no MMH. Ver docs/RESULTADOS-BIGDATA-PROFS.md.
PESOS_FUSAO = {"e5": 0.70, "tfidf": 0.15, "tfidf_char": 0.15}

# Texto que alimenta a fusão: `basico` (efisco_processado) ou `rede`
# (doc_virtual_expandido, a expansão da fase [1]). A grade mediu a expansão
# 3 pp abaixo da normalização básica nos dois datasets; a rede continua
# ancorando o PDM (blocking) e servindo o léxico curado ao `basico`.
TEXTO_NEURAL = "basico"

# --- Sinais simbólicos sobre o top-K (grade: pós `combinado_calibrado`) -----
# Somados ao score neural com peso pequeno: medida tipada convertida,
# cobertura de entidades do KG ponderada por IDF, compatibilidade de classe
# entre as taxonomias (histórico de vínculos, leave-one-out) e a dedução
# ontológica da fase [4] (equivalenteA = 1, broadMatch = 0,5). Na execução D,
# 86% dos top-1 com `equivalenteA` estavam certos, contra 65% dos `incompleto`:
# a regra SWRL informa — só não podia entrar na escala em que entrava.
PESO_SINAIS = 0.05
PESOS_REL_SINAIS = {"medidas": 0.6, "graphrag_idf": 1.0, "taxonomia": 0.3, "ontologia": 0.6}
BONUS_DEDUCAO = {"equivalenteA": 1.0, "broadMatch": 0.5}

# --- Fase [6]: como a LLM adjudica ------------------------------------------
#   lista   — vê os K_LISTA melhores candidatos de uma consulta de uma vez e
#             pontua cada um (padrão). Julgar um par isolado ("é o mesmo item?")
#             rebaixava corretos com especificação incompleta: −2,2 a −4,8 pp.
#   par     — o comportamento original (um par por consulta, top-1).
#   nenhuma — sem LLM na fase [6] (votação por vizinhos TF-IDF).
ADJUDICACAO = "lista"
K_LISTA = 5
# score_graphrag = (1-p) neural + p nota da LLM, e o score final pesa 0,2 nele.
# Varrido no bigdata_profs (etapa 18): p = 0,1 e 0,2 dão R@3 85,2% (contra 84,0%
# sem adjudicação); 0,3 já cai para 83,1% e 0,5 para 80,9% — a nota da LLM é
# boa para DESEMPATAR o topo, não para substituir a similaridade.
PESO_ADJUDICACAO_LISTA = 0.1
USAR_CONTEXTO_MS = False         # --contexto-ms: liga o índice Microsoft GraphRAG

# Limiares de confiança (§4.6)
LIMIAR_ALTA   = 0.75
LIMIAR_MEDIA  = 0.50

# Zona cinzenta que o GraphRAG adjudica (§4.2-2)
ZONA_CINZENTA = (0.25, LIMIAR_ALTA)

# Composição do escore final (§4.1), como ficou depois da medição no bigdata_profs
# (docs/RESULTADOS-BIGDATA-PROFS.md, §9): a dedução ontológica deixou de entrar
# como fatia convexa de 0,35 — nessa escala, um `equivalenteA` (1,0) contra um
# `incompleto` (0,3) valia mais que todo o espalhamento da similaridade fundida e
# puxava pares errados ao topo (−1,8 pp de R@3) — e passou a ser um SINAL ADITIVO
# da fase [5b], na escala dos outros. O peso convexo continua disponível
# (--peso-ontologia) para reproduzir a composição antiga.
PESOS_SCORE_FINAL = {
    "ontologia": 0.0,
    "neural":    0.80,
    "graphrag":  0.20,
}
# Veto: entre os 71 top-1 vetados da execução D, 35 estavam certos e 36 errados
# — o veto não separa nada neste corpus. Fica desligado (1,0); --fator-veto 0.25
# reativa a penalidade original.
FATOR_VETO = 1.0

# Faixas de confiança pela MARGEM entre o 1º e o 2º candidato (score neural),
# medida contra o gabarito na execução D: margem >= 0,05 -> P@1 ~0,93-0,99;
# 0,02-0,05 -> ~0,73; < 0,02 -> ~0,41. O limiar absoluto (0,75/0,50) do MMH não
# separa nada na escala do score fundido (81% das consultas caíam em "Média").
LIMIAR_MARGEM_ALTA  = 0.05
LIMIAR_MARGEM_MEDIA = 0.02

# Chaves canônicas do esquema de atributos: são as do PERFIL de domínio. Eram
# uma tupla literal aqui (calibre, bisel, conector...), o que amarrava o esquema
# a material médico-hospitalar.


def atributos_pdm_keys() -> tuple[str, ...]:
    return perfil_ativo().chaves_atributos


# Caches de LLM por dataset: o conteúdo é o texto do corpus, e misturar corpora
# no mesmo arquivo só cria confusão na hora de invalidar.


def _cache_path(nome: str) -> str:
    d = contexto().dataset.dir_cache
    d.mkdir(parents=True, exist_ok=True)
    return str(d / nome)


def cache_extracao_path() -> str:
    return _cache_path("extracao_llm.json")


def cache_graphrag_path() -> str:
    return _cache_path("graphrag_llm.json")

# Teto de chamadas LLM por execução na fase [6] (custo previsível)
LLM_MAX_ADJUDICACOES = 500

# Paralelismo das chamadas de extração da fase [2] (uma por texto único)
LLM_WORKERS = 8

# Flags de execução (definidas por linha de comando)
USAR_LLM    = True
USAR_REDE   = True
USAR_CROSS  = False   # estágio 2 do matcher (cross-encoder): medido como prejudicial
                      # (−1,7 pp de R@3, 33 min de CPU); --cross religa
USAR_SINAIS = True    # fase [5b]; --sem-sinais desliga


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _parse_dict(v) -> dict:
    """Converte com segurança um valor de coluna (dict ou repr-string) em dict."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            parsed = ast.literal_eval(v)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _construir_cliente_openai():
    """Constrói o cliente OpenAI. Retorna (client, motivo) ou (None, motivo)."""
    if not USAR_LLM:
        return None, "desligado_por_flag"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None, "sem_api_key"
    try:
        import openai
    except ImportError:
        return None, "sdk_ausente"
    return openai.OpenAI(api_key=api_key), "ok"


def _hash_texto(texto: str) -> str:
    """Chave de cache estável (sha1 do texto normalizado)."""
    base = normalizar_texto(texto or "", manter_maiusculas=True).strip()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _carregar_cache_extracao(caminho: str = "") -> dict:
    if os.path.exists(caminho):
        try:
            with open(caminho, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cache LLM ilegível (%s) — recomeçando vazio.", exc)
    return {}


def _salvar_cache_extracao(cache: dict, caminho: str = "") -> None:
    try:
        with open(caminho, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=0)
    except OSError as exc:
        log.warning("Falha ao salvar cache LLM: %s", exc)


class _Cronometro:
    """Mede wall/CPU por fase, para a tabela de tempos do relatório."""

    def __init__(self):
        self.tempos: dict[str, dict] = {}

    def medir(self, nome: str):
        crono = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.time()
                self.c0 = time.process_time()
                return self

            def __exit__(self, *exc):
                crono.tempos[nome] = {
                    "wall_s": round(time.time() - self.t0, 3),
                    "cpu_s": round(time.process_time() - self.c0, 3),
                }
                return False

        return _Ctx()


CRONO = _Cronometro()


# ---------------------------------------------------------------------------
# FASE [0] — FONTES
# Separa o que é CATÁLOGO (universo de busca) do que é CONSULTA (o que se quer
# casar), e guarda o gabarito à parte — só para avaliar, nunca para construir.
# ---------------------------------------------------------------------------

def fase_0_fontes() -> dict:
    """
    Fase [0]: Ingestão e caracterização.

    Produz três estruturas distintas, que antes eram uma só:
      catalogo  : itens CATMAT únicos — o universo onde se busca
      consultas : itens e-Fisco únicos — o que se quer casar
      gold_map  : codigo_efisco -> {codigos CATMAT corretos}, só para avaliação
    """
    log.info("=" * 68)
    log.info("[0] FONTES — Ingestão, catálogo e consultas")
    log.info("=" * 68)

    ctx = contexto()
    dfs = {}
    for nome in ctx.dataset.arquivos:
        caminho = ctx.dataset.caminho(nome)
        if caminho.exists():
            d = ctx.dataset.ler(nome)
            d.fillna("", inplace=True)
            dfs[nome] = d
            log.info("  %-10s: %d registros", nome, len(d))

    principal = ctx.dataset.arquivo_padrao
    if principal not in dfs:
        raise SystemExit(
            f"O dataset '{ctx.dataset.nome}' não tem o arquivo '{principal}'. "
            f"Encontrados: {sorted(dfs) or 'nenhum'}."
        )
    df = dfs[principal]

    # O perfil ganha aqui o que der para induzir do corpus (sem efeito quando o
    # perfil já traz o bloco). Precisa ser antes das fases que leem o domínio.
    ctx.completar_com_dados(df)

    # Colunas opcionais de taxonomia/proveniência: quando o CSV as traz, os
    # sinais da fase [5b] e a análise as usam.
    extras_cat = [c for c in ("grupo_catmat",) if c in df.columns]
    extras_q = [c for c in ("grupo_efisco", "origem") if c in df.columns]

    # --- Catálogo CATMAT (universo de busca) -------------------------------
    catalogo = (
        df[["codigo_catmat", "item_catmat", "classe_catmat", *extras_cat]]
        .drop_duplicates(subset="codigo_catmat")
        .reset_index(drop=True)
    )
    catalogo = catalogo[catalogo["codigo_catmat"].str.strip() != ""].reset_index(drop=True)
    catalogo["catmat_processado"] = catalogo["item_catmat"].apply(preprocessar_texto_catmat)
    catalogo["catmat_atributos"] = catalogo["item_catmat"].apply(extrair_atributos_catmat)
    catalogo["pdm"] = catalogo["catmat_atributos"].apply(
        lambda a: a.get("tipo_produto", "").strip()
    )

    # --- Consultas e-Fisco -------------------------------------------------
    consultas = (
        df[["codigo_efisco", "item_efisco", "classe_efisco", *extras_q]]
        .drop_duplicates(subset="codigo_efisco")
        .reset_index(drop=True)
    )
    consultas = consultas[consultas["codigo_efisco"].str.strip() != ""].reset_index(drop=True)
    consultas["efisco_processado"] = consultas["item_efisco"].apply(preprocessar_texto_efisco)

    # --- Gabarito (SOMENTE avaliação) --------------------------------------
    gold_map: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        ce, cc = row.get("codigo_efisco", "").strip(), row.get("codigo_catmat", "").strip()
        if ce and cc:
            gold_map[ce].add(cc)

    situacao_map = {}
    for _, row in df.iterrows():
        ce = row.get("codigo_efisco", "").strip()
        if ce and ce not in situacao_map:
            situacao_map[ce] = row.get("situacao", "")

    n_ok = int(df["situacao"].str.contains("OK_SEM_FLAGS").sum())
    stats = {
        "n_pares_gabarito": len(df),
        "n_ok_sem_flags": n_ok,
        "n_catalogo": len(catalogo),
        "n_consultas": len(consultas),
        "n_classes_efisco": int(df["classe_efisco"].nunique()),
        "n_classes_catmat": int(df["classe_catmat"].nunique()),
        "n_pdms_catalogo": int(catalogo["pdm"].nunique()),
    }

    print("\n=== [0] FONTES ===")
    print(f"  Pares no gabarito       : {len(df)}  (só para avaliação)")
    print(f"  Catálogo CATMAT (busca) : {len(catalogo)} itens únicos")
    print(f"  Consultas e-Fisco       : {len(consultas)} itens únicos")
    print(f"  PDMs distintos          : {catalogo['pdm'].nunique()}")
    print(f"  OK sem flags            : {n_ok}  ({100*n_ok/len(df):.1f}%)")

    return {
        "dfs": dfs,
        "df_gold": df,
        "catalogo": catalogo,
        "consultas": consultas,
        "gold_map": dict(gold_map),
        "situacao_map": situacao_map,
        "stats_fase0": stats,
    }


# ---------------------------------------------------------------------------
# FASE [1] — REDE SEMÂNTICA
# Normaliza, expande, desambigua e ancora — dos DOIS lados. O lado CATMAT ganha
# ancoragem oficial (tipo_produto); o lado e-Fisco ganha a ancoragem inferida.
# ---------------------------------------------------------------------------

def fase_1_rede_semantica(estado: dict) -> dict:
    """
    Fase [1]: rede semântica — os 5 primeiros passos do serviço da §3.7.

    Saídas:
        consultas.doc_virtual_expandido : forma canônica + termos ativados
        consultas.pdm_ancoragem         : PDM inferido (chave do blocking)
        catalogo.pdm                    : PDM oficial (tipo_produto)
    """
    log.info("=" * 68)
    log.info("[1] REDE SEMÂNTICA — Construção e aplicação")
    log.info("=" * 68)

    consultas = estado["consultas"]
    catalogo  = estado["catalogo"]

    if not USAR_REDE:
        # Ablação da §3.10: o pipeline roda sem a rede, para medir o ganho dela.
        log.warning("[1] ABLAÇÃO ATIVA — rede semântica desligada (--sem-rede).")
        consultas["doc_virtual_expandido"] = consultas["efisco_processado"]
        consultas["efisco_normalizado"]    = consultas["efisco_processado"]
        consultas["pdm_ancoragem"]         = ""
        estado.update({
            "consultas": consultas,
            "grafo_semantico": nx.MultiDiGraph(),
            "oov_por_consulta": [[] for _ in range(len(consultas))],
            "stats_fase1": {"ablacao": True, "n_nos": 0, "n_arestas": 0, "taxa_oov": None},
        })
        return estado

    log.info("Construindo grafo léxico de domínio...")
    G = construir_grafo()

    exportar_graphml(G, dir_resultados() / "rede_semantica.graphml")
    exportar_skos(G, dir_resultados() / "rede_semantica_skos.ttl")
    exportar_fila_curadoria(G, dir_resultados() / "fila_curadoria.csv")

    log.info("Gerando visualizações do grafo...")
    # O PDM em foco sai do perfil ou é o mais frequente do grafo — estava fixo
    # em "AGULHA PUNCAO OSSEA", que não existe em nenhum outro domínio.
    foco = perfil_ativo().pdm_foco_exemplo or pdm_mais_frequente(G)
    try:
        if foco:
            from .config import sanitizar
            visualizar_grafo(
                G, pdm_foco=foco,
                titulo=f"Vizinhança: {foco}",
                salvar_em=dir_resultados() / f"rede_semantica_{sanitizar(foco)[:30].lower()}.png",
            )
        visualizar_grafo(
            G, pdm_foco=None, max_nos=80,
            titulo="Rede Semântica CATMAT <-> e-Fisco (top 80 nós)",
            salvar_em=dir_resultados() / "rede_semantica_geral.png",
        )
    except Exception as exc:
        log.warning("Visualização falhou (%s) — seguindo.", exc)

    log.info("Normalizando, expandindo e ancorando %d consultas...", len(consultas))

    docs_virtuais, normalizados, ancoras, oov_por_consulta = [], [], [], []

    for texto in consultas["item_efisco"]:
        toks = remover_stopwords(tokenizar(normalizar_texto(texto, manter_maiusculas=True)))

        # Passo 1 + 3: normalização COM desambiguação pelo contexto do registro.
        # O contexto é o próprio conjunto de tokens do item — é ele que decide
        # o sentido de um termo ambíguo (§3.7-3).
        toks_norm = [normalizar_termo(G, t, contexto=toks) for t in toks]
        texto_norm = " ".join(toks_norm)

        # Passo 2: expansão por propagação de ativação (constrained).
        seeds = [t for t in toks_norm if len(t) > 3]
        ativacoes = expandir_por_ativacao(G, seeds, limiar=0.15, max_saltos=2)
        expandidos = [
            t for t, s in sorted(ativacoes.items(), key=lambda x: -x[1])
            if s > 0.3 and t not in toks_norm
        ][:5]

        # Passo 5: documento virtual expandido.
        docs_virtuais.append((texto_norm + " " + " ".join(expandidos)).strip())
        normalizados.append(texto_norm)

        # Passo 4: ancoragem em nível de registro (consenso, não primeiro token).
        ancoras.append(ancorar_registro(G, toks_norm) or "")

        oov_por_consulta.append([
            orig for orig, norm in zip(toks, toks_norm)
            if orig == norm and _id_no("Conceito", orig) not in G
        ])

    consultas["efisco_normalizado"]    = normalizados
    consultas["doc_virtual_expandido"] = docs_virtuais
    consultas["pdm_ancoragem"]         = ancoras

    # Catálogo: quando o rótulo não traz tipo_produto, ancora pela rede.
    pdm_catalogo = []
    for pdm_oficial, texto in zip(catalogo["pdm"], catalogo["item_catmat"]):
        if pdm_oficial:
            pdm_catalogo.append(pdm_oficial)
        else:
            toks = remover_stopwords(tokenizar(normalizar_texto(texto, manter_maiusculas=True)))
            pdm_catalogo.append(ancorar_registro(G, toks) or "")
    catalogo["pdm"] = pdm_catalogo

    n_oov = sum(len(o) for o in oov_por_consulta)
    n_toks = sum(
        len(tokenizar(normalizar_texto(t, manter_maiusculas=True)))
        for t in consultas["item_efisco"]
    )
    taxa_oov = n_oov / max(n_toks, 1)
    n_com_pdm = int((consultas["pdm_ancoragem"] != "").sum())

    print("\n=== [1] REDE SEMÂNTICA ===")
    print(f"  Nós no grafo            : {G.number_of_nodes()}")
    print(f"  Arestas no grafo        : {G.number_of_edges()}")
    print(f"  Consultas com âncora PDM: {n_com_pdm} / {len(consultas)}")
    print(f"  Taxa OOV (tokens)       : {100*taxa_oov:.1f}%")
    print(f"  Exemplo doc. expandido  : {docs_virtuais[0][:90]}...")

    estado.update({
        "consultas": consultas,
        "catalogo": catalogo,
        "grafo_semantico": G,
        "oov_por_consulta": oov_por_consulta,
        "stats_fase1": {
            "n_nos": G.number_of_nodes(),
            "n_arestas": G.number_of_edges(),
            "n_com_pdm": n_com_pdm,
            "taxa_oov": float(taxa_oov),
        },
    })
    return estado


# ---------------------------------------------------------------------------
# FASE [2] — EXTRAÇÃO (camada de PERCEPÇÃO)
# A LLM lê o texto livre e devolve atributos no esquema PDM. Roda sobre textos
# ÚNICOS (não sobre pares), então o custo é linear no catálogo, não quadrático.
# ---------------------------------------------------------------------------

# As nove regex do extrator determinístico saíram daqui para o perfil
# (`extracao:` no YAML), onde cada regra declara que atributo preenche e de que
# grupo de captura. Eram elas que traziam gauge, bisel e Luer para dentro do
# orquestrador.


def _canonicalizar_atributos_llm(bruto: dict) -> dict:
    """Normaliza o JSON da LLM para o mesmo formato do extrator regex, de modo
    que a ontologia compare chaves e valores alinhados dos dois lados.

    As regras por atributo (extrair número, concatenar sufixo, caixa alta) são
    declaradas no perfil; ver `PerfilDominio.canonicalizar`.
    """
    return perfil_ativo().canonicalizar(bruto)


def extrair_atributos_llm(texto: str, client, cache: dict,
                          model: str = LLM_MODEL_PADRAO) -> dict:
    """Camada de percepção: texto livre -> atributos do esquema PDM.
    Cache por hash. Levanta exceção em falha — o chamador decide o fallback."""
    if not texto or not str(texto).strip():
        return {}
    chave = _hash_texto(texto)
    if chave in cache:
        return cache[chave]

    # O papel e o esquema de chaves vêm do perfil: o prompt é conteúdo de
    # domínio, e um prompt que fala de gauge e bisel extrai mal um catálogo de
    # medicamentos. Ver `PerfilDominio.papel` e `bloco_prompt_atributos`.
    perfil = perfil_ativo()
    prompt = (
        f"{perfil.papel('extrator')} A partir da descrição do item, extraia os "
        "atributos no esquema PDM e responda APENAS com um objeto JSON. Use "
        "SOMENTE estas chaves (omita a chave se o atributo não aparecer no "
        "texto; não invente valores):\n"
        f"{perfil.bloco_prompt_atributos()}\n"
        "Todos os valores em MAIÚSCULAS e sem acento.\n\n"
        f"Descrição do item:\n{str(texto)[:400]}"
    )
    # 320 tokens: com o esquema estendido do perfil multi-domínio (19 chaves),
    # 200 cortava o JSON no meio ("Unterminated string") em ~1% dos itens, que
    # caíam para a regex sem necessidade.
    resp = client.chat.completions.create(
        model=model,
        max_tokens=320,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    atribs = _canonicalizar_atributos_llm(json.loads(resp.choices[0].message.content or "{}"))
    cache[chave] = atribs
    return atribs


def extrair_atributos_regex(texto: str) -> dict:
    """Fallback determinístico da camada de percepção (sem LLM/chave/API).

    As regras vêm do perfil de domínio; ver `PerfilDominio.extrair_regex`. Um
    perfil sem regras devolve dicionário vazio, e a fase [2] registra a lacuna
    em vez de fingir extração.
    """
    return perfil_ativo().extrair_regex(texto)

    if (m := _RE_CALIBRE.search(t)):
        atribs["calibre"] = m.group(1) + "G"
    if (m := _RE_COMPRIMENTO.search(t)):
        atribs["comprimento_valor"] = m.group(1)
        atribs["comprimento_unidade"] = m.group(2).upper()
    if (m := _RE_VOLUME.search(t)):
        atribs["volume_valor"] = m.group(1)
        atribs["volume_unidade"] = m.group(2).upper()
    if (m := _RE_DIMENSAO.search(t)):
        atribs["dimensao"] = m.group(0).upper().strip()
    if _RE_LUER.search(t):
        atribs["conector"] = "LUER LOCK"
    if _RE_ESTERIL.search(t):
        atribs["esterilidade"] = "ESTERIL USO UNICO"
    if (m := _RE_MATERIAL.search(t)):
        atribs["material"] = normalizar_texto(m.group(0), manter_maiusculas=True)
    if _RE_BISEL.search(t):
        atribs["bisel"] = "BISEL CORTANTE"
    if (m := _RE_MODELO.search(t)):
        atribs["modelo"] = m.group(0).upper()
    return atribs


# Espaço de nomes da ontologia — o mesmo IRI da fase [4], para que uma
# violação SHACL aqui e uma dedução OWL lá apontem para o mesmo :atributo.
def _ns_dominio() -> Namespace:
    """Espaço de nomes da ontologia do domínio ativo."""
    return Namespace(f"{iri_ontologia()}#")


def _construir_shapes_pdm() -> Graph:
    """
    Shapes SHACL de verdade (o papel do SHACL na Figura 1): uma sh:NodeShape
    por família de PDM, com sh:minCount 1 para cada característica
    DEFINIDORA — a MESMA lista que a regra SWRL da fase [4] usa para deduzir
    equivalência (`atributos_definidores` do perfil). Gerado a partir dela, e não
    escrito à mão, para que as duas fases nunca divirjam sobre o que é
    "obrigatório" por família.
    """
    DOM = _ns_dominio()
    shapes = Graph()
    shapes.bind(perfil_ativo().prefixo, DOM)
    shapes.bind("sh", SH)
    for familia, obrigatorios in caracteristicas_definidoras().items():
        forma = DOM[f"Forma{familia}"]
        shapes.add((forma, RDF.type, SH.NodeShape))
        shapes.add((forma, SH.targetClass, DOM[familia]))
        for atributo in obrigatorios:
            prop = BNode()
            shapes.add((forma, SH.property, prop))
            shapes.add((prop, SH.path, DOM[atributo]))
            shapes.add((prop, SH.minCount, Literal(1)))
            shapes.add((prop, SH.severity, SH.Violation))
            shapes.add((prop, SH.message, Literal(
                f"{familia}: atributo definidor '{atributo}' ausente no esquema PDM.",
                lang="pt")))
    return shapes


# Montado sob demanda e memorizado por domínio: no import o perfil ativo ainda
# não é conhecido (a CLI só escolhe o dataset depois).
_CACHE_SHAPES: dict[str, Graph] = {}


def shapes_pdm() -> Graph:
    nome = perfil_ativo().nome
    if nome not in _CACHE_SHAPES:
        _CACHE_SHAPES[nome] = _construir_shapes_pdm()
    return _CACHE_SHAPES[nome]


def validar_shacl_lote(atributos: list[dict], pdms: list[str]) -> list[dict]:
    """
    Valida o esquema PDM extraído contra `shapes_pdm()` com o motor pyshacl
    (SHACL-Core; sem inferência OWL, já que as shapes usam só sh:minCount).

    Roda numa ÚNICA passada sobre todas as linhas — um grafo de dados com um
    indivíduo por item — em vez de validar item a item: o custo de montar o
    engine do pyshacl é fixo por chamada, então validar N grafos de 1 item
    cada é ~N vezes mais caro que validar 1 grafo de N indivíduos.
    """
    DOM = _ns_dominio()
    definidoras = caracteristicas_definidoras()
    chaves = atributos_pdm_keys()
    dados = Graph()
    dados.bind(perfil_ativo().prefixo, DOM)
    nos = []  # (uri, obrigatorios) na ordem de entrada, para remontar a saída
    for i, (atribs, pdm) in enumerate(zip(atributos, pdms)):
        familia = _familia_de(pdm or "")
        obrigatorios = definidoras.get(familia, [])
        no = DOM[f"item_{i}"]
        nos.append((no, obrigatorios))
        if not familia:
            continue
        dados.add((no, RDF.type, DOM[familia]))
        for chave, valor in (atribs or {}).items():
            if chave in chaves and valor:
                dados.add((no, DOM[chave], Literal(str(valor))))

    conforms, relatorio, _ = pyshacl.validate(
        dados, shacl_graph=shapes_pdm(), inference="none",
        allow_warnings=True, meta_shacl=False,
    )

    ausentes_por_no: dict[str, set] = defaultdict(set)
    if not conforms:
        for resultado, _, foco in relatorio.triples((None, SH.focusNode, None)):
            caminho = relatorio.value(resultado, SH.resultPath)
            if caminho is not None:
                ausentes_por_no[str(foco)].add(str(caminho).rsplit("#", 1)[-1])

    saida = []
    for no, obrigatorios in nos:
        if not obrigatorios:
            saida.append({"completo": True, "ausentes": [], "escore_completude": 1.0})
            continue
        ausentes = sorted(ausentes_por_no.get(str(no), set()))
        saida.append({
            "completo": not ausentes,
            "ausentes": ausentes,
            "escore_completude": 1.0 - len(ausentes) / len(obrigatorios),
        })
    return saida


def fase_2_extracao(estado: dict) -> dict:
    """
    Fase [2]: extração de atributos dos DOIS lados + validação de esquema.

    Roda sobre textos únicos: 1 chamada por item de catálogo e por consulta,
    com cache em disco. O CATMAT já tem rótulo estruturado, então a LLM entra
    para canonizar as chaves no mesmo vocabulário do e-Fisco — sem isso a
    ontologia compararia nomenclaturas diferentes e nunca deduziria nada.
    """
    log.info("=" * 68)
    log.info("[2] EXTRAÇÃO — Percepção (LLM) + validação de esquema")
    log.info("=" * 68)

    consultas, catalogo = estado["consultas"], estado["catalogo"]
    client, motivo = _construir_cliente_openai()
    if client is None:
        log.warning("[2] LLM indisponível (%s) — extração por regex.", motivo)
    else:
        log.info("[2] Extração via %s (cache: %s)", LLM_MODEL_PADRAO, cache_extracao_path())

    cache = _carregar_cache_extracao(cache_extracao_path())
    stats_fonte = {"llm": 0, "regex": 0, "erros_llm": 0}

    import threading
    from concurrent.futures import ThreadPoolExecutor

    trava = threading.Lock()

    def _extrair(texto: str) -> dict:
        if client is not None:
            try:
                a = extrair_atributos_llm(texto, client, cache, LLM_MODEL_PADRAO)
                with trava:
                    stats_fonte["llm"] += 1
                return a
            except Exception as exc:
                with trava:
                    stats_fonte["erros_llm"] += 1
                    n_erros = stats_fonte["erros_llm"]
                if n_erros <= 3:
                    log.warning("[2] LLM falhou (%s) — regex neste item.", str(exc)[:140])
        with trava:
            stats_fonte["regex"] += 1
        return extrair_atributos_regex(texto)

    def _extrair_lote(textos: list[str]) -> list[dict]:
        """Uma chamada por texto; em paralelo quando há LLM (I/O-bound), na
        ordem de entrada. O cache em memória é compartilhado entre threads —
        escrita de dict sob o GIL é segura, e repetir um texto só custa uma
        chamada a mais."""
        if client is None or LLM_WORKERS <= 1:
            return [_extrair(t) for t in textos]
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            return list(pool.map(_extrair, textos))

    log.info("Extraindo atributos de %d consultas e-Fisco (%d threads)...",
             len(consultas), LLM_WORKERS if client is not None else 1)
    consultas["atributos"] = _extrair_lote(consultas["item_efisco"].tolist())

    log.info("Extraindo atributos de %d itens do catálogo CATMAT...", len(catalogo))
    canons = _extrair_lote(catalogo["item_catmat"].tolist())
    attrs_catalogo = []
    for canon, parser in zip(canons, catalogo["catmat_atributos"]):
        # O parser do rótulo é autoritativo para tipo_produto; a LLM canoniza
        # as características. Os dois se somam, com a LLM por cima.
        merged = {k: v for k, v in parser.items() if k == "tipo_produto"}
        merged.update(canon)
        attrs_catalogo.append(merged)
    catalogo["atributos"] = attrs_catalogo

    if client is not None:
        _salvar_cache_extracao(cache, cache_extracao_path())

    # Validação de esquema — SHACL real (rdflib + pyshacl), papel da Figura 1
    consultas["shacl"] = validar_shacl_lote(
        list(consultas["atributos"]), list(consultas["pdm_ancoragem"])
    )
    n_completos = sum(1 for s in consultas["shacl"] if s["completo"])
    escore_medio = float(np.mean([s["escore_completude"] for s in consultas["shacl"]]))

    n_com_attr_e = sum(1 for a in consultas["atributos"] if a)
    n_com_attr_c = sum(1 for a in catalogo["atributos"] if a)

    print("\n=== [2] EXTRAÇÃO ===")
    print(f"  Fonte                   : LLM {stats_fonte['llm']} | "
          f"regex {stats_fonte['regex']} | erros {stats_fonte['erros_llm']}")
    print(f"  Consultas com atributos : {n_com_attr_e} / {len(consultas)}")
    print(f"  Catálogo com atributos  : {n_com_attr_c} / {len(catalogo)}")
    print(f"  Esquema completo (SHACL): {n_completos} / {len(consultas)}")
    print(f"  Escore médio completude : {escore_medio:.3f}")

    estado.update({
        "consultas": consultas,
        "catalogo": catalogo,
        "stats_fase2": {
            "fonte_extracao": stats_fonte,
            "n_completos": n_completos,
            "escore_medio_completude": escore_medio,
            "n_consultas_com_atributos": n_com_attr_e,
            "n_catalogo_com_atributos": n_com_attr_c,
        },
    })
    return estado


# ---------------------------------------------------------------------------
# SIMILARIDADE FUNDIDA — compartilhada pelas fases [3] e [5]
# ---------------------------------------------------------------------------

def _similaridade_fusao(estado: dict) -> np.ndarray:
    """
    Matriz consulta x catálogo com a fusão linear léxico-densa da grade
    (`e5_tfidf_char_lin`): PESOS_FUSAO['e5'] x cosseno E5 + TF-IDF de palavra
    + TF-IDF de caractere, todos em [0, 1]. Calculada uma vez e guardada no
    estado; os embeddings E5 vêm do mesmo cache em disco da grade modular
    (indexado pelo texto), então uma grade já rodada poupa o encode aqui.
    """
    if "_sim_fusao" in estado:
        return estado["_sim_fusao"]

    from . import avaliacao_modular as am

    consultas, catalogo = estado["consultas"], estado["catalogo"]
    if TEXTO_NEURAL == "rede" and "doc_virtual_expandido" in consultas.columns:
        col_q, pre_id = "doc_virtual_expandido", "pipeline_rede"
    else:
        col_q, pre_id = "efisco_processado", "basico"
    textos = am.Textos(
        pre_id=pre_id,
        queries=consultas[col_q].fillna("").astype(str).tolist(),
        catalogo=catalogo["catmat_processado"].fillna("").astype(str).tolist(),
    )
    ctx = estado.setdefault("_ctx_fusao", {})
    try:
        emb_e, emb_c = am._embeddings_e5(textos, ctx)
        sim_e5 = emb_e @ emb_c.T
        modelo = MODELO_EMBEDDING
    except am.ModuloIndisponivel as exc:
        log.warning("E5 indisponível (%s) — fusão só com TF-IDF.", exc)
        emb_e = emb_c = None
        sim_e5 = np.zeros((len(consultas), len(catalogo)), dtype=np.float32)
        modelo = "tfidf_fallback"

    sim = (PESOS_FUSAO["e5"] * sim_e5
           + PESOS_FUSAO["tfidf"] * am._matriz_tfidf(textos, ctx)
           + PESOS_FUSAO["tfidf_char"] * am._matriz_tfidf_char(textos, ctx)).astype(np.float32)

    estado.update({"_sim_fusao": sim, "_sim_e5": sim_e5, "_emb_e": emb_e, "_emb_c": emb_c,
                   "_modelo_neural": modelo, "_texto_neural": col_q})
    log.info("[fusão] %s sobre '%s': %d x %d.", " + ".join(f"{p:.2f}x{k}" for k, p in PESOS_FUSAO.items()),
             col_q, *sim.shape)
    return sim


# ---------------------------------------------------------------------------
# FASE [3] — BLOCKING / GERAÇÃO DE CANDIDATOS
# É aqui que o pipeline deixa de olhar a diagonal do gabarito: para cada
# consulta, monta-se um conjunto de candidatos vindo do CATÁLOGO INTEIRO.
# ---------------------------------------------------------------------------

def fase_3_blocking(estado: dict) -> dict:
    """
    Fase [3]: gera candidatos por Classe/PDM, com subsunção quando o bloco
    fica pequeno demais (§4.3 [3]: "subsunção amplia o bloco quando o recall cai").

    Estratégia em cascata, da chave mais específica para a mais frouxa:
      1. PDM âncora idêntico
      2. Subsunção: hiperônimos/hipônimos do PDM (hierarquia do perfil)
      3. PDM que compartilha a família (AGULHA, SERINGA, ...)
      4. Classe CATMAT igual à classe e-Fisco
      5. Último recurso: catálogo inteiro truncado por afinidade lexical

    O par correto NÃO é injetado no bloco. O recall do blocking passa a ser
    medido de verdade — é o teto de tudo que vem depois.
    """
    log.info("=" * 68)
    log.info("[3] BLOCKING — Geração de candidatos")
    log.info("=" * 68)

    consultas, catalogo = estado["consultas"], estado["catalogo"]
    gold_map = estado["gold_map"]

    # Índices invertidos do catálogo
    por_pdm: dict[str, list[int]] = defaultdict(list)
    por_classe: dict[str, list[int]] = defaultdict(list)
    por_familia: dict[str, list[int]] = defaultdict(list)

    # A lista de famílias era literal aqui. Vem do perfil (`familias:` + as
    # chaves da hierarquia curada); num domínio sem perfil, é induzida dos
    # rótulos de PDM do próprio catálogo.
    perfil = perfil_ativo()
    familias = list(perfil.familias)

    for i, row in catalogo.iterrows():
        pdm = (row["pdm"] or "").strip().upper()
        if pdm:
            por_pdm[pdm].append(i)
        classe = (row["classe_catmat"] or "").strip()
        if classe:
            por_classe[classe].append(i)
        for fam in familias:
            if fam in pdm:
                por_familia[fam].append(i)

    # Subsunção: mapa PDM -> PDMs relacionados na hierarquia curada
    relacionados: dict[str, set] = defaultdict(set)
    for hiper, hipos in perfil.hierarquia.items():
        h_up = hiper.upper()
        for hipo in hipos:
            relacionados[h_up].add(hipo.upper())
            relacionados[hipo.upper()].add(h_up)

    # Afinidade lexical como rede de segurança (TF-IDF sobre o catálogo)
    vec_cat = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=20000)
    mat_cat = vec_cat.fit_transform(catalogo["catmat_processado"].fillna(""))

    # --- Estratégia 6: blocking semântico cross-PDM pela similaridade FUNDIDA
    # (E5 + TF-IDF palavra + TF-IDF caractere). Era só o cosseno E5; a grade
    # mediu a fusão +7,6 pp de R@3 acima dele, e o que ela recupera a mais são
    # itens que o embedding deixava fora do top-50 — exatamente o teto que o
    # blocking define. A matriz fica no estado e a fase [5] a reutiliza.
    log.info("[blocking] Similaridade fundida para blocking semântico...")
    sim_fusao = _similaridade_fusao(estado)

    candidatos: dict[str, list[int]] = {}
    origens: list[str] = []
    tamanhos: list[int] = []

    for pos_e, (_, row) in enumerate(consultas.iterrows()):
        cod_e = row["codigo_efisco"]
        pdm = (row["pdm_ancoragem"] or "").strip().upper()
        classe = (row["classe_efisco"] or "").strip()

        cands: set[int] = set()
        origem = "sem_candidato"

        # 1. PDM exato
        if pdm and pdm in por_pdm:
            cands.update(por_pdm[pdm])
            origem = "pdm"

        # 2. Subsunção
        if len(cands) < 5 and pdm:
            for rel in relacionados.get(pdm, ()):
                cands.update(por_pdm.get(rel, []))
            if cands and origem == "sem_candidato":
                origem = "subsuncao"

        # 3. Família do PDM
        if len(cands) < 5 and pdm:
            for fam in familias:
                if fam in pdm:
                    cands.update(por_familia.get(fam, []))
                    break
            if cands and origem == "sem_candidato":
                origem = "familia"

        # 4. Classe
        if len(cands) < 5 and classe and classe in por_classe:
            cands.update(por_classe[classe])
            if origem == "sem_candidato":
                origem = "classe"

        # 5. Rede de segurança lexical
        if len(cands) < 5:
            v = vec_cat.transform([row["efisco_processado"] or row["item_efisco"]])
            sims = (mat_cat @ v.T).toarray().ravel()
            cands.update(np.argsort(-sims)[:50].tolist())
            if origem == "sem_candidato":
                origem = "lexical"

        # 6. Blocking semântico cross-PDM pela similaridade fundida
        # Complementa sempre — captura equivalências que escapam ao PDM idêntico.
        sims_fus = sim_fusao[pos_e]
        top_sem = set(np.argsort(-sims_fus)[:_N_SEM_BLOCKING].tolist())
        cands.update(top_sem)
        if cands and origem == "sem_candidato":
            origem = "fusao"

        # Teto por consulta: mantém os mais afins pela similaridade fundida
        if len(cands) > MAX_CANDIDATOS:
            idxs = np.array(sorted(cands))
            cands = set(idxs[np.argsort(-sims_fus[idxs])[:MAX_CANDIDATOS]].tolist())

        candidatos[cod_e] = sorted(cands)
        origens.append(origem)
        tamanhos.append(len(cands))

    consultas["origem_bloco"] = origens
    consultas["n_candidatos"] = tamanhos

    # --- Recall do blocking (teto de tudo que vem depois) -------------------
    cod_por_idx = catalogo["codigo_catmat"].tolist()
    n_aval = n_cobertos = 0
    for cod_e, cands in candidatos.items():
        alvos = gold_map.get(cod_e, set())
        if not alvos:
            continue
        n_aval += 1
        cods_cand = {cod_por_idx[i] for i in cands}
        if alvos & cods_cand:
            n_cobertos += 1
    recall_blocking = n_cobertos / max(n_aval, 1)

    total_pares = int(sum(tamanhos))
    print("\n=== [3] BLOCKING ===")
    print(f"  Consultas               : {len(consultas)}")
    print(f"  Pares candidatos gerados: {total_pares}")
    print(f"  Bloco médio             : {np.mean(tamanhos):.1f}  (máx {max(tamanhos)})")
    print(f"  Origem do bloco         : {dict(Counter(origens))}")
    print(f"  RECALL DO BLOCKING      : {100*recall_blocking:.1f}%  "
          f"({n_cobertos}/{n_aval})  <- teto do pipeline")

    estado.update({
        "consultas": consultas,
        "candidatos": candidatos,
        "cod_por_idx": cod_por_idx,
        "stats_fase3": {
            "n_pares_candidatos": total_pares,
            "bloco_medio": float(np.mean(tamanhos)),
            "bloco_max": int(max(tamanhos)),
            "origem_bloco": dict(Counter(origens)),
            "recall_blocking": float(recall_blocking),
        },
    })
    return estado


# ---------------------------------------------------------------------------
# FASE [4] — ONTOLOGIA + REASONER
# Deixa de ser heurística em Python: monta a ABox OWL e roda o Pellet sobre as
# regras SWRL da §2.2. O que o reasoner deduz vem marcado como deduzido.
# ---------------------------------------------------------------------------

def fase_4_ontologia(estado: dict) -> dict:
    """
    Fase [4]: ontologia OWL + reasoner (§2.2, §4.3 [4]).

    - equivalenteA : deduzido pelo Pellet a partir das regras SWRL
    - broadMatch   : deduzido por subsunção na árvore de PDMs
    - veto         : conflito de característica definidora (checagem, não SWRL)
    - incompleto   : o reasoner não fechou e não há conflito
    """
    log.info("=" * 68)
    log.info("[4] ONTOLOGIA — OWL + reasoner Pellet (SWRL §2.2)")
    log.info("=" * 68)

    consultas, catalogo = estado["consultas"], estado["catalogo"]
    candidatos = estado["candidatos"]
    cod_por_idx = estado["cod_por_idx"]

    try:
        from .ontologia_owl import OntologiaDominio
    except ImportError as exc:
        log.error("owlready2 indisponível (%s) — fase [4] sem dedução.", exc)
        estado["deducoes"] = {}
        estado["stats_fase4"] = {"erro": "owlready2_ausente"}
        return estado

    pdms = sorted({p for p in catalogo["pdm"] if p} |
                  {p for p in consultas["pdm_ancoragem"] if p})

    onto = OntologiaDominio(pdms, perfil_ativo().hierarquia)

    # Só entram na ABox os itens que realmente participam de algum par
    # candidato — reasoning sobre o universo inteiro seria desperdício.
    idxs_usados = sorted({i for cands in candidatos.values() for i in cands})
    for i in idxs_usados:
        row = catalogo.iloc[i]
        onto.adicionar_item("catmat", row["codigo_catmat"], row["pdm"], row["atributos"])

    for _, row in consultas.iterrows():
        if candidatos.get(row["codigo_efisco"]):
            onto.adicionar_item("efisco", row["codigo_efisco"],
                                row["pdm_ancoragem"], row["atributos"])

    log.info("ABox: %s", onto.estatisticas())

    with CRONO.medir("reasoner_pellet"):
        deducoes_reasoner = onto.deduzir()

    # Projeta as deduções sobre os pares candidatos e completa com veto.
    deducoes: dict[tuple, dict] = {}
    contagem = Counter()

    for cod_e, cands in candidatos.items():
        for i in cands:
            cod_c = cod_por_idx[i]
            chave = (cod_e, cod_c)

            if chave in deducoes_reasoner:
                deducoes[chave] = deducoes_reasoner[chave]
                contagem[deducoes[chave]["deducao"]] += 1
                continue

            veto = onto.verificar_veto(cod_e, cod_c)
            if veto:
                deducoes[chave] = veto
                contagem["veto"] += 1
                continue

            # Subsunção -> broadMatch, lida da hierarquia inferida na TBox.
            sub = onto.subsuncao_entre(cod_e, cod_c)
            if sub:
                deducoes[chave] = sub
                contagem["broadMatch"] += 1
                continue

            deducoes[chave] = {
                "deducao": "incompleto",
                "score": 0.3,
                "explicacao": ["Reasoner não fechou equivalência e não há conflito "
                               "de característica definidora (atributos ausentes)."],
                "fonte": "sem_deducao",
            }
            contagem["incompleto"] += 1

    try:
        onto.salvar(dir_resultados() / "ontologia.owl")
    except Exception as exc:
        log.warning("Falha ao salvar a ontologia: %s", exc)

    n_total = sum(contagem.values())
    print("\n=== [4] ONTOLOGIA ===")
    print(f"  Indivíduos na ABox      : {onto.estatisticas()['n_individuos']}")
    print(f"  Regras SWRL             : {onto.estatisticas()['n_regras_swrl']}")
    print(f"  Pares avaliados         : {n_total}")
    for ded, cnt in contagem.most_common():
        print(f"    {ded:15s}: {cnt:6d}  ({100*cnt/max(n_total,1):.1f}%)")

    estado.update({
        "deducoes": deducoes,
        "ontologia": onto,
        "stats_fase4": {
            **onto.estatisticas(),
            "por_deducao": dict(contagem),
        },
    })
    return estado


# ---------------------------------------------------------------------------
# FASE [5] — MATCHER NEURAL (dois estágios)
# Estágio 1: bi-encoder assimétrico sobre todo o bloco (barato, vetorizado).
# Estágio 2: cross-encoder só sobre os K melhores (caro, preciso).
# ---------------------------------------------------------------------------

def _carregar_modelo_embedding():
    try:
        from sentence_transformers import SentenceTransformer
        log.info("Carregando bi-encoder: %s", MODELO_EMBEDDING)
        return SentenceTransformer(MODELO_EMBEDDING)
    except Exception as exc:
        log.warning("sentence-transformers indisponível (%s) — TF-IDF.", exc)
        return None


_cross_encoder_model = None


def _carregar_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is not None:
        return _cross_encoder_model
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder(MODELO_CROSS_ENCODER, max_length=128)
        log.info("Cross-encoder carregado: %s", MODELO_CROSS_ENCODER)
    except Exception as exc:
        log.warning("Cross-encoder indisponível (%s) — fallback fuzzy.", exc)
        _cross_encoder_model = None
    return _cross_encoder_model


def _score_fuzzy(texto_e: str, texto_c: str) -> float:
    """Fallback do estágio 2 quando o cross-encoder neural não carrega."""
    from rapidfuzz import fuzz as rfuzz
    te = normalizar_texto(texto_e or "", manter_maiusculas=True)
    tc = normalizar_texto(texto_c or "", manter_maiusculas=True)
    if not te or not tc:
        return 0.0
    return rfuzz.token_set_ratio(te, tc) / 100.0


def fase_5_matcher_neural(estado: dict) -> dict:
    """
    Fase [5]: recuperação densa + reranking (§4.3 [5]).

    Produz `pares`: um DataFrame com uma linha por par candidato sobrevivente
    ao estágio 1, com score_cosine, score_cross e score_neural.
    """
    log.info("=" * 68)
    log.info("[5] MATCHER NEURAL — bi-encoder + cross-encoder")
    log.info("=" * 68)

    consultas, catalogo = estado["consultas"], estado["catalogo"]
    candidatos, cod_por_idx = estado["candidatos"], estado["cod_por_idx"]

    # Estágio 1 pela similaridade FUNDIDA (a mesma do blocking). O cosseno E5
    # puro fica registrado em `score_cosine` para a trilha e para a ablação.
    sim_fusao = _similaridade_fusao(estado)
    sim_e5 = estado["_sim_e5"]
    modelo = estado.get("_modelo_neural", MODELO_EMBEDDING)

    log.info("Estágio 1: pontuando blocos pela fusão e retendo top-%d por consulta...",
             K_RERANK)

    linhas = []
    for pos_e, row in consultas.iterrows():
        cod_e = row["codigo_efisco"]
        cands = candidatos.get(cod_e, [])
        if not cands:
            continue
        idxs = np.asarray(cands, dtype=int)
        sims_f = sim_fusao[pos_e, idxs]
        sims_cos = sim_e5[pos_e, idxs]
        ordem = np.argsort(-sims_f)[:K_RERANK]
        for rank1, j in enumerate(ordem):
            i_cat = int(idxs[j])
            linhas.append({
                "codigo_efisco": cod_e,
                "codigo_catmat": cod_por_idx[i_cat],
                "idx_consulta": pos_e,
                "idx_catalogo": i_cat,
                "score_cosine": float(sims_cos[j]),
                "score_fusao": float(sims_f[j]),
                "rank_estagio1": rank1,
            })

    pares = pd.DataFrame(linhas)
    log.info("Estágio 1 -> %d pares mantidos (de %d candidatos).",
             len(pares), int(consultas["n_candidatos"].sum()))

    # --- Estágio 2: cross-encoder ------------------------------------------
    # Opcional (--sem-cross): na grade o reranking por cross-encoder mMARCO
    # derrubou o E5 em todas as linhas do MMH. Fica ligado por fidelidade ao
    # desenho de dois estágios, e a avaliação reporta a ablação sem ele.
    itens_e = consultas["item_efisco"].fillna("").tolist()
    itens_c = catalogo["item_catmat"].fillna("").tolist()
    entradas = [
        (itens_e[r.idx_consulta], itens_c[r.idx_catalogo])
        for r in pares.itertuples()
    ]

    ce = _carregar_cross_encoder() if USAR_CROSS else None
    if ce is not None and entradas:
        # Cache em disco por par de textos: o cross-encoder custa ~30 min em CPU
        # neste corpus e o resultado não muda entre execuções.
        cache_cross = _carregar_cache_extracao(_cache_path("cross_encoder.json"))
        chaves = [_hash_texto(a + "||" + b) for a, b in entradas]
        faltam = [k for k in range(len(entradas)) if chaves[k] not in cache_cross]
        log.info("Estágio 2: cross-encoder em %d pares (%d em cache)...",
                 len(entradas), len(entradas) - len(faltam))
        t0 = time.time()
        if faltam:
            brutos = ce.predict([entradas[k] for k in faltam], batch_size=64,
                                show_progress_bar=False)
            novos = 1.0 / (1.0 + np.exp(-np.asarray(brutos, dtype=float)))
            for k, s in zip(faltam, novos):
                cache_cross[chaves[k]] = float(s)
            _salvar_cache_extracao(cache_cross, _cache_path("cross_encoder.json"))
            log.info("  -> %.1fs (%.0f pares/s)", time.time() - t0,
                     len(faltam) / max(time.time() - t0, 1e-6))
        scores_cross = [cache_cross[k] for k in chaves]
        estagio2 = MODELO_CROSS_ENCODER
    elif USAR_CROSS:
        log.info("Estágio 2: fallback fuzzy em %d pares...", len(entradas))
        scores_cross = [_score_fuzzy(a, b) for a, b in entradas]
        estagio2 = "fuzzy_fallback"
    else:
        log.info("Estágio 2 desligado (--sem-cross): score neural = fusão.")
        scores_cross = pares["score_fusao"].tolist()
        estagio2 = "desligado"

    pares["score_cross"] = scores_cross
    if USAR_CROSS:
        pares["score_neural"] = (
            0.7 * pares["score_fusao"].clip(0, 1) + 0.3 * pares["score_cross"].clip(0, 1)
        ).clip(0, 1)
    else:
        pares["score_neural"] = pares["score_fusao"].clip(0, 1)

    print("\n=== [5] MATCHER NEURAL ===")
    print(f"  Estágio 1 (fusão)       : {modelo} + TF-IDF palavra + TF-IDF caractere "
          f"({', '.join(f'{k} {p:.2f}' for k, p in PESOS_FUSAO.items())}) "
          f"sobre '{estado.get('_texto_neural', '')}'")
    print(f"  Estágio 2               : {estagio2}")
    print(f"  Pares pontuados         : {len(pares)}")
    print(f"  Score cosine E5 médio   : {pares['score_cosine'].mean():.3f}")
    print(f"  Score fusão médio       : {pares['score_fusao'].mean():.3f}")
    print(f"  Score cross médio       : {pares['score_cross'].mean():.3f}")
    print(f"  Score neural médio      : {pares['score_neural'].mean():.3f}")

    estado.update({
        "pares": pares,
        "emb_consultas": estado.get("_emb_e"),
        "stats_fase5": {
            "modelo_embedding": modelo,
            "encoder_assimetrico": bool(modelo == MODELO_EMBEDDING and _USA_PREFIXO_E5),
            "fusao": dict(PESOS_FUSAO),
            "texto_neural": estado.get("_texto_neural", ""),
            "cross_encoder": estagio2,
            "n_pares_pontuados": len(pares),
            "score_cosine_medio": float(pares["score_cosine"].mean()),
            "score_fusao_medio": float(pares["score_fusao"].mean()),
            "score_cross_medio": float(pares["score_cross"].mean()),
            "score_neural_medio": float(pares["score_neural"].mean()),
        },
    })
    return estado


# ---------------------------------------------------------------------------
# FASE [5b] — SINAIS SIMBÓLICOS SOBRE O TOP-K
# O que a grade modular chamou de pós-processador `combinado_calibrado`: três
# sinais que o bi-encoder não vê, somados ao score neural com peso pequeno.
# ---------------------------------------------------------------------------

def fase_5b_sinais_simbolicos(estado: dict) -> dict:
    """
    Fase [5b]: medida tipada + cobertura de entidades (IDF) + taxonomia.

    - medidas       : fração das medidas da consulta (valor+unidade convertidos
                      à base) que o candidato satisfaz — `caracteristicas.py`.
    - graphrag_idf  : cobertura das entidades da consulta pelo candidato, no KG
                      offline do GraphRAG, cada entidade pesando o seu IDF; com
                      LLM, os atributos extraídos na fase [2] entram como
                      entidades tipadas dos dois lados.
    - taxonomia     : P(classe CATMAT | classe e-Fisco) estimada nos vínculos
                      conhecidos em leave-one-out — a MESMA fonte (histórico de
                      vínculos) que a fase [1] usa para minerar jargão, e por
                      isso admitida aqui; a consulta avaliada nunca contribui
                      para a própria estimativa.

    score_neural += PESO_SINAIS x (0,6 medidas + 1,0 graphrag_idf + 0,3 taxonomia)
    """
    log.info("=" * 68)
    log.info("[5b] SINAIS SIMBÓLICOS — medida, GraphRAG-IDF, taxonomia")
    log.info("=" * 68)

    pares = estado["pares"]
    pares["score_neural_base"] = pares["score_neural"]
    for col in ("bonus_medidas", "bonus_graphrag_idf", "bonus_taxonomia",
                "bonus_ontologia", "score_sinais"):
        pares[col] = 0.0
    if not USAR_SINAIS or pares.empty:
        log.warning("[5b] desligada (--sem-sinais) ou sem pares.")
        estado["stats_fase5b"] = {"ativa": False}
        return estado

    from . import caracteristicas
    from . import graphrag as gr
    from . import taxonomia as tx

    consultas, catalogo = estado["consultas"], estado["catalogo"]
    perfil = perfil_ativo()

    # 1. Medidas tipadas do texto CRU (a normalização apaga a polegada).
    med_q = caracteristicas.indice(consultas["item_efisco"].fillna("").astype(str).tolist(), perfil)
    med_c = caracteristicas.indice(catalogo["item_catmat"].fillna("").astype(str).tolist(), perfil)
    r_q, r_c = caracteristicas.resumo(med_q), caracteristicas.resumo(med_c)

    # 2. KG offline + IDF (com os atributos da fase [2], quando existem).
    kg = gr.construir_kg(consultas, catalogo)
    m = max(1, len(kg.ents_catalogo))
    idf = {ent: float(np.log(1.0 + m / len(idxs))) for ent, idxs in kg.indice_invertido.items()}

    def _cobertura(i: int, j: int) -> float:
        proprias = kg.ents_consulta.get(i, set())
        relevantes = {e for e in proprias if not e.startswith("PDM::")} or proprias
        if not relevantes:
            return 0.0
        peso = lambda e: (1.0 if e.startswith("TOK::") else 2.0) * idf.get(e, 1.0)
        total = sum(peso(e) for e in relevantes)
        comuns = relevantes & kg.ents_catalogo.get(j, set())
        return sum(peso(e) for e in comuns) / total if total > 0 else 0.0

    # 3. Taxonomia (LOO sobre o histórico de vínculos).
    al = tx.alinhamento_historico(consultas, catalogo, estado["gold_map"])
    cobertura_tx = tx.diagnostico_cobertura(al, estado["gold_map"], consultas, catalogo)

    # 4. Dedução ontológica da fase [4], como bônus na escala dos outros sinais.
    deducoes = estado.get("deducoes", {})

    b_med, b_gr, b_tx, b_ont = [], [], [], []
    for r in pares.itertuples():
        i, j = int(r.idx_consulta), int(r.idx_catalogo)
        b_med.append(caracteristicas.satisfacao(med_q[i], med_c[j]) if med_q[i] else 0.0)
        b_gr.append(_cobertura(i, j))
        b_tx.append(al.compatibilidade(i, j))
        ded = deducoes.get((r.codigo_efisco, r.codigo_catmat), {}).get("deducao", "")
        b_ont.append(BONUS_DEDUCAO.get(ded, 0.0))

    pares["bonus_medidas"] = b_med
    pares["bonus_graphrag_idf"] = b_gr
    pares["bonus_taxonomia"] = b_tx
    pares["bonus_ontologia"] = b_ont
    pares["score_sinais"] = (
        PESOS_REL_SINAIS["medidas"] * pares["bonus_medidas"]
        + PESOS_REL_SINAIS["graphrag_idf"] * pares["bonus_graphrag_idf"]
        + PESOS_REL_SINAIS["taxonomia"] * pares["bonus_taxonomia"]
        + PESOS_REL_SINAIS.get("ontologia", 0.0) * pares["bonus_ontologia"]
    )
    # Sem clip: o que importa é a ORDEM dentro de cada consulta, e cortar em 1
    # achataria justamente os primeiros colocados.
    pares["score_neural"] = pares["score_neural_base"] + PESO_SINAIS * pares["score_sinais"]

    print("\n=== [5b] SINAIS SIMBÓLICOS ===")
    print(f"  Consultas com medida    : {r_q['com_medida']} / {r_q['total']}  "
          f"(catálogo {r_c['com_medida']} / {r_c['total']})")
    print(f"  KG                      : {kg.G.number_of_nodes()} nós, "
          f"{len(kg.indice_invertido)} entidades, {len(set(kg.comunidade.values()))} comunidades")
    print(f"  Taxonomia (LOO)         : correto compatível em "
          f"{100 * cobertura_tx['fracao_correto_compativel']:.1f}% das consultas; "
          f"{cobertura_tx['classe_sem_historico_loo']} classes sem histórico")
    print(f"  Dedução ontológica      : {int((pares['bonus_ontologia'] > 0).sum())} pares com "
          f"equivalenteA/broadMatch entre os {len(pares)} do top-K")
    print(f"  Pares com algum sinal   : {int((pares['score_sinais'] > 0).sum())} / {len(pares)}")
    print(f"  Pesos                   : {PESO_SINAIS} x {PESOS_REL_SINAIS}")

    estado.update({
        "pares": pares,
        "stats_fase5b": {
            "ativa": True,
            "peso": PESO_SINAIS,
            "pesos_relativos": dict(PESOS_REL_SINAIS),
            "medidas": {"consultas": r_q, "catalogo": r_c},
            "kg": {"n_nos": kg.G.number_of_nodes(), "n_entidades": len(kg.indice_invertido),
                   "n_comunidades": len(set(kg.comunidade.values()))},
            "taxonomia": {**al.resumo(), **cobertura_tx,
                          "fonte": "histórico de vínculos, leave-one-out"},
            "ontologia": {"pares_com_deducao_positiva": int((pares["bonus_ontologia"] > 0).sum()),
                          "bonus": dict(BONUS_DEDUCAO)},
            "pares_com_sinal": int((pares["score_sinais"] > 0).sum()),
        },
    })
    return estado


# ---------------------------------------------------------------------------
# FASE [6] — GRAPHRAG
# Implementação completa seguindo a arquitetura GraphRAG (Microsoft/IBM):
#   Indexação: extração de entidades → grafo de conhecimento bipartito →
#              detecção de comunidades (Louvain) → resumos de comunidade (LLM)
#   Busca:     local (vizinhança de entidades no KG) + global (resumo de comunidade)
#              + contextual (top-3 por vizinho TF-IDF, inclusive matches prováveis)
# ---------------------------------------------------------------------------

_SW_GRAPHRAG = {
    "PARA", "COMO", "TIPO", "COM", "SEM", "USO", "CADA", "DEVE",
    "ESTE", "ESSA", "PRODUTO", "ITEM", "UNIDADE", "CAIXA",
}


def _extrair_entidades_item(texto: str, atributos: dict, pdm: str) -> set:
    """Entidades de um item: PDM + valores de atributos PDM + tokens relevantes."""
    entidades: set = set()
    if pdm:
        entidades.add(f"PDM::{pdm.upper().strip()}")
    for k, v in (atributos or {}).items():
        if v and k != "tipo_produto":
            entidades.add(f"{k}::{str(v).upper().strip()[:40]}")
    tokens = [t for t in re.findall(r'[A-ZÀ-Ú]{4,}', texto.upper())
              if t not in _SW_GRAPHRAG]
    for t in tokens[:10]:
        entidades.add(f"TOK::{t}")
    return entidades


def _construir_kg_graphrag(consultas, catalogo):
    """
    Grafo de conhecimento bipartito itens ↔ entidades.
    Nós: 'e:{i}' (e-Fisco), 'c:{j}' (CATMAT), strings de entidade.
    Returns: G_kg, ents_e {i->set}, ents_c {j->set}
    """
    G_kg = nx.Graph()
    ents_e: dict = {}
    ents_c: dict = {}

    pdm_col_e = "pdm_ancoragem" if "pdm_ancoragem" in consultas.columns else None
    pdm_col_c = "pdm" if "pdm" in catalogo.columns else None
    tem_attr_e = "atributos" in consultas.columns
    tem_attr_c = "atributos" in catalogo.columns

    for i, row in enumerate(consultas.itertuples(index=False)):
        pdm = (getattr(row, pdm_col_e, "") or "") if pdm_col_e else ""
        attr = (getattr(row, "atributos", {}) or {}) if tem_attr_e else {}
        ents = _extrair_entidades_item(
            getattr(row, "item_efisco", "") or "", attr, pdm
        )
        ents_e[i] = ents
        node = f"e:{i}"
        G_kg.add_node(node, tipo="efisco", idx=i)
        for e in ents:
            if not G_kg.has_node(e):
                G_kg.add_node(e, tipo="entidade")
            G_kg.add_edge(node, e)

    for j, row in enumerate(catalogo.itertuples(index=False)):
        pdm = (getattr(row, pdm_col_c, "") or "") if pdm_col_c else ""
        attr = (getattr(row, "atributos", {}) or {}) if tem_attr_c else {}
        ents = _extrair_entidades_item(
            getattr(row, "item_catmat", "") or "", attr, pdm
        )
        ents_c[j] = ents
        node = f"c:{j}"
        G_kg.add_node(node, tipo="catmat", idx=j)
        for e in ents:
            if not G_kg.has_node(e):
                G_kg.add_node(e, tipo="entidade")
            G_kg.add_edge(node, e)

    return G_kg, ents_e, ents_c


def _detectar_comunidades_graphrag(G_kg, n_efisco):
    """
    Projeta o KG bipartito para grafo e-Fisco ↔ e-Fisco (itens que compartilham
    >= 2 entidades), aplica Louvain e retorna {idx_efisco -> community_id}.
    """
    G_proj = nx.Graph()
    G_proj.add_nodes_from(range(n_efisco))

    ent_to_efisco: dict = defaultdict(list)
    for i in range(n_efisco):
        node = f"e:{i}"
        if not G_kg.has_node(node):
            continue
        for ent in G_kg.neighbors(node):
            if G_kg.nodes[ent].get("tipo") == "entidade":
                ent_to_efisco[ent].append(i)

    for idxs in ent_to_efisco.values():
        if len(idxs) > 200:   # entidade muito genérica — ignora
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if G_proj.has_edge(ia, ib):
                    G_proj[ia][ib]["peso"] += 1
                else:
                    G_proj.add_edge(ia, ib, peso=1)

    G_proj.remove_edges_from(
        [(u, v) for u, v, d in G_proj.edges(data=True) if d.get("peso", 0) < 2]
    )

    try:
        coms = nx.community.louvain_communities(G_proj, seed=42)
        idx_para_com = {m: cid for cid, membros in enumerate(coms) for m in membros}
    except Exception:
        idx_para_com = {i: 0 for i in range(n_efisco)}

    return idx_para_com, G_proj


def _resumir_comunidades_graphrag(idx_para_com, consultas, client, cache, model,
                                   max_comunidades=60):
    """LLM gera resumo de cada comunidade (busca global). Returns {com_id -> texto}."""
    resumos: dict = {}
    if client is None:
        return resumos

    com_para_idxs: dict = defaultdict(list)
    for idx, com in idx_para_com.items():
        com_para_idxs[com].append(idx)

    itens_e = consultas["item_efisco"].fillna("").tolist()
    coms_por_tamanho = sorted(com_para_idxs.items(), key=lambda x: -len(x[1]))

    for com_id, membros in coms_por_tamanho[:max_comunidades]:
        if len(membros) < 2:
            continue
        samples = [itens_e[i][:90] for i in membros[:5] if i < len(itens_e)]
        chave_hash = _hash_texto(f"com_resumo_{com_id}_" + "|".join(samples))
        if chave_hash in cache:
            resumos[com_id] = cache[chave_hash].get("resumo", "")
            continue
        prompt = (
            f"{perfil_ativo().papel('especialista')} "
            "Descreva em UMA frase curta o que esses itens e-Fisco tem em comum "
            "(produto, material, finalidade):\n"
            + "\n".join(f"- {s}" for s in samples if s)
        )
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=60,
                messages=[{"role": "user", "content": prompt}]
            )
            resumo = (resp.choices[0].message.content or "").strip()
            cache[chave_hash] = {"resumo": resumo}
            resumos[com_id] = resumo
        except Exception:
            pass

    return resumos


def _busca_local_kg(ents_query: set, ents_catmat_all: dict, top_k: int = 6) -> list:
    """
    Busca local no KG: CATMATs que compartilham entidades relevantes com a query.
    Entidades relevantes = atributos PDM e tokens (exclui PDM generico sozinho).
    Returns: [(idx_cat, n_shared)]
    """
    ents_relevantes = {e for e in ents_query if not e.startswith("PDM::")}
    if not ents_relevantes:
        ents_relevantes = ents_query
    scores: dict = {}
    for j, ents_c in ents_catmat_all.items():
        n = len(ents_relevantes & ents_c)
        if n > 0:
            scores[j] = n
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def _adjudicar_graphrag_llm(par: dict, contexto_local: list, contexto_global: str,
                             client, cache: dict,
                             model: str = LLM_MODEL_PADRAO) -> dict:
    """
    Adjudicacao por LLM com contexto local (KG + vizinhos TF-IDF) e global (comunidade).
    Inclui matches provaveis/parciais — nao so os de alta confianca.

    A chave do cache inclui o contexto global (a resposta do local_search do
    Microsoft GraphRAG): um veredito dado SEM o indice nao pode ser reaproveitado
    numa execucao COM ele, senao a segunda execucao nunca consulta o indice.
    """
    chave = _hash_texto(par["efisco"] + "||" + par["catmat"] + "||ctx:" + (contexto_global or "")[:300])
    if chave in cache:
        return cache[chave]

    linhas_ctx = ""
    if contexto_local:
        linhas_ctx = "Contexto de itens relacionados no grafo de conhecimento:\n"
        for v in contexto_local[:8]:
            tipo = v.get("tipo", "vizinho")
            score_str = (f" (score {v['score']:.2f})" if "score" in v else
                         (f" ({v['n_entidades']} entidades comuns)" if "n_entidades" in v else ""))
            linhas_ctx += (
                f"  [{tipo}] '{str(v['efisco'])[:90]}'"
                f" ~ '{str(v['catmat'])[:90]}'{score_str}\n"
            )

    ctx_global = (
        f"\nFamilia de produtos desta consulta: {contexto_global}"
        if contexto_global else ""
    )

    prompt = (
        f"{perfil_ativo().papel('auditor')} "
        "Decida se o item e-Fisco corresponde ao item CATMAT abaixo.\n\n"
        "PAR EM ANALISE:\n"
        f"  e-Fisco : {str(par['efisco'])[:220]}\n"
        f"  CATMAT  : {str(par['catmat'])[:220]}\n"
        f"  Score neural: {par['score_neural']:.2f} | "
        f"Deducao ontologica: {par.get('deducao', 'nenhuma')}"
        f"{ctx_global}\n\n"
        + (linhas_ctx or "  (sem contexto de vizinhos)\n")
        + "\nCriterios:\n"
        "  - Equivalencia total (mesmo produto e especificacao): 0.75-1.0\n"
        "  - Match provavel (mesmo produto, especificacao diferente ou ambigua): 0.40-0.74\n"
        "  - Produtos distintos: < 0.40\n"
        "Responda APENAS JSON: "
        '{"score": <float 0-1>, "justificativa": "<uma frase>"}'
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=160,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    dados = json.loads(resp.choices[0].message.content or "{}")
    try:
        score = float(dados.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    veredito = {
        "score": max(0.0, min(1.0, score)),
        "justificativa": str(dados.get("justificativa", "")).strip(),
    }
    cache[chave] = veredito
    return veredito


def _adjudicar_lista_llm(consulta: str, candidatos: list[dict], contexto_global: str,
                         client, cache: dict, model: str = LLM_MODEL_PADRAO) -> dict:
    """
    Adjudicacao EM LISTA: a LLM ve os K melhores candidatos de uma consulta de
    uma vez e da uma nota a cada um, comparando-os entre si.

    E a correcao do modo par a par, que perguntava "este par e o mesmo item?"
    a um par isolado e rebaixava corretos com especificacao incompleta (-2,2 a
    -4,8 pp de R@3). Em lista a pergunta e relativa -- "qual destes e o item?"
    -- e ausencia de atributo deixa de contar como conflito. Cada candidato
    chega com o que o pipeline ja sabe dele: score neural, deducao da
    ontologia e fracao das medidas da consulta que satisfaz.
    """
    codigos = "|".join(c["codigo"] for c in candidatos)
    chave = _hash_texto("lista||" + consulta + "||" + codigos + "||ctx:" + (contexto_global or "")[:300])
    if chave in cache:
        return cache[chave]

    linhas = []
    for k, c in enumerate(candidatos, start=1):
        pistas = [f"score neural {c['score_neural']:.2f}"]
        if c.get("deducao") and c["deducao"] != "incompleto":
            pistas.append(f"ontologia: {c['deducao']}")
        if c.get("medidas") is not None:
            pistas.append(f"medidas da consulta satisfeitas: {c['medidas']:.0%}")
        linhas.append(f"  {k}. [{' | '.join(pistas)}] {str(c['texto'])[:220]}")

    ctx_global = (f"\nContexto do grafo de conhecimento sobre a consulta: {contexto_global[:600]}"
                  if contexto_global else "")
    prompt = (
        f"{perfil_ativo().papel('auditor')} "
        "Abaixo, um item do e-Fisco e os candidatos do CATMAT que o sistema considera "
        "mais provaveis. De a CADA candidato uma nota de 0 a 1 para 'e o mesmo material "
        "(mesmo produto, especificacao compativel)'. Compare os candidatos ENTRE SI: o "
        "melhor recebe a maior nota; candidatos de outro produto recebem nota abaixo de "
        "0,3. Especificacao AUSENTE num candidato nao e conflito -- so divergencia "
        "explicita e. Nao invente atributos.\n\n"
        f"ITEM E-FISCO:\n  {str(consulta)[:300]}\n\n"
        f"CANDIDATOS CATMAT:\n" + "\n".join(linhas) + f"{ctx_global}\n\n"
        'Responda APENAS JSON: {"notas": [<nota do 1>, <nota do 2>, ...], '
        '"melhor": <numero do melhor>, "justificativa": "<uma frase>"}'
    )
    resp = client.chat.completions.create(
        model=model, max_tokens=220,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    dados = json.loads(resp.choices[0].message.content or "{}")
    brutas = dados.get("notas") or []
    notas = []
    for k in range(len(candidatos)):
        try:
            v = float(brutas[k])
        except (IndexError, TypeError, ValueError):
            v = 0.5   # sem nota: neutro, nao pune
        notas.append(max(0.0, min(1.0, v)))
    veredito = {
        "notas": notas,
        "melhor": dados.get("melhor"),
        "justificativa": str(dados.get("justificativa", "")).strip(),
    }
    cache[chave] = veredito
    return veredito


def fase_6_graphrag(estado: dict) -> dict:
    """
    Fase [6]: GraphRAG via Microsoft GraphRAG (github.com/microsoft/graphrag).

    INDEXACAO (uma vez, cached em graphrag_workspace/output/):
    1. Prepara corpus: um .txt por item (e-Fisco + CATMAT)
    2. Executa `graphrag index`: extrai entidades, relações, detecta comunidades
       (Leiden), gera resumos hierárquicos de comunidade via LLM
    3. Artefatos persistidos em parquet para reutilização

    BUSCA / ADJUDICACAO (por consulta na zona cinzenta):
    4. graphrag.api.local_search(query=texto_efisco) — recupera entidades
       vizinhas, relações e resumo de comunidade do KG oficial
    5. LLM adjudica com o contexto GraphRAG + score neural + dedução OWL
    """
    log.info("=" * 68)
    log.info("[6] GRAPHRAG — Microsoft GraphRAG (indexacao + local_search + LLM)")
    log.info("=" * 68)

    consultas, catalogo, pares = estado["consultas"], estado["catalogo"], estado["pares"]
    deducoes = estado.get("deducoes", {})

    if pares.empty:
        estado["stats_fase6"] = {"n_zona_cinzenta": 0}
        return estado

    pares["score_graphrag"] = pares["score_neural"]
    pares["graphrag_justificativa"] = ""

    # ------------------------------------------------------------------ #
    # 1-3. Indexação Microsoft GraphRAG (cached)                          #
    # ------------------------------------------------------------------ #
    from . import graphrag as gr
    from pathlib import Path

    client, motivo = _construir_cliente_openai()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    adjudica = client is not None and LLM_MAX_ADJUDICACOES > 0 and ADJUDICACAO != "nenhuma"
    if not adjudica:
        # Sem LLM (ou teto zero, ou --adjudicacao nenhuma) não há adjudicação,
        # e indexar o corpus no Microsoft GraphRAG (horas e milhares de
        # chamadas) não teria quem o lesse.
        log.warning("[6] sem adjudicacao LLM (%s) — votacao por vizinhos TF-IDF.",
                    f"LLM indisponivel: {motivo}" if client is None
                    else ("teto de adjudicacoes = 0" if LLM_MAX_ADJUDICACOES <= 0
                          else "--adjudicacao nenhuma"))
        gr_ok = False
        gr_config = gr_artefatos = None
    elif not USAR_CONTEXTO_MS:
        # O índice oficial só é construído/consultado com --contexto-ms: medido
        # como adjudicador de pares, piorou o ranking (−4,8 pp) e custou ~2 h de
        # indexação mais duas chamadas por consulta. Fica como opção.
        log.info("[6] Índice Microsoft GraphRAG não usado (--contexto-ms liga).")
        gr_ok = False
        gr_config = gr_artefatos = None
    else:
        try:
            gr_config, gr_artefatos = gr.inicializar(consultas, catalogo, api_key)
            gr_ok = all(v is not None for v in gr_artefatos.values())
        except Exception as exc:
            log.warning("[6] GraphRAG inicialização falhou: %s — usando fallback TF-IDF.", exc)
            gr_ok = False
            gr_config = gr_artefatos = None

    # ------------------------------------------------------------------ #
    # 4. Estruturas de consulta                                          #
    # ------------------------------------------------------------------ #
    itens_e = consultas["item_efisco"].fillna("").tolist()
    itens_c = catalogo["item_catmat"].fillna("").tolist()

    melhor = pares.loc[pares.groupby("codigo_efisco")["score_neural"].idxmax()]

    # Grafo TF-IDF como fallback / complemento
    docs = consultas["doc_virtual_expandido"].fillna("").tolist()
    vec  = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    mat  = vec.fit_transform(docs)
    G_sim = nx.Graph()
    G_sim.add_nodes_from(range(len(consultas)))
    por_bloco: dict = defaultdict(list)
    for i, pdm in enumerate(consultas["pdm_ancoragem"].fillna("")):
        por_bloco[pdm or "SEM_PDM"].append(i)
    for _, idxs in por_bloco.items():
        if len(idxs) < 2 or len(idxs) > 1500:
            continue
        sub = mat[idxs]
        sim = (sub @ sub.T).toarray()
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if sim[a, b] >= 0.35:
                    G_sim.add_edge(idxs[a], idxs[b], peso=float(sim[a, b]))

    top3_por_consulta: dict = {}
    for _, grupo in pares.groupby("codigo_efisco"):
        pos_e = int(grupo["idx_consulta"].iloc[0])
        top3 = grupo.nlargest(3, "score_neural")
        top3_por_consulta[pos_e] = [
            {"idx_catalogo": int(r.idx_catalogo), "score": float(r.score_neural)}
            for r in top3.itertuples() if r.score_neural >= 0.30
        ]

    # ------------------------------------------------------------------ #
    # 5. Adjudicacao na zona cinzenta                                    #
    # ------------------------------------------------------------------ #
    cache_adj = _carregar_cache_extracao(cache_graphrag_path()) if client else {}

    # Zona cinzenta pela MARGEM entre o 1º e o 2º candidato (score neural):
    # consulta com margem abaixo de LIMIAR_MARGEM_ALTA é incerta — é a mesma
    # definição das faixas de confiança da fase [8], validada contra o gabarito.
    # O limiar absoluto ZONA_CINZENTA do MMH não separa nada na escala fundida.
    ordenado = pares.sort_values(["codigo_efisco", "score_neural"], ascending=[True, False])
    grupos = ordenado.groupby("codigo_efisco")["score_neural"]
    primeiro = grupos.first()
    segundo = grupos.apply(lambda s: s.iloc[1] if len(s) > 1 else s.iloc[0])
    margem = (primeiro - segundo).to_dict()
    alvos = [r for r in melhor.itertuples()
             if margem.get(r.codigo_efisco, 0.0) < LIMIAR_MARGEM_ALTA]
    n_zona = len(alvos)
    alvos.sort(key=lambda r: margem.get(r.codigo_efisco, 0.0))   # as mais incertas primeiro
    if adjudica and len(alvos) > LLM_MAX_ADJUDICACOES:
        log.info("[6] Zona cinzenta (margem < %.2f): %d consultas; teto %d — adjudicando as "
                 "de menor margem (máxima adjudicada: %.3f).", LIMIAR_MARGEM_ALTA, n_zona,
                 LLM_MAX_ADJUDICACOES,
                 margem.get(alvos[LLM_MAX_ADJUDICACOES - 1].codigo_efisco, 0.0))
    else:
        log.info("[6] Zona cinzenta (margem < %.2f): %d consultas.", LIMIAR_MARGEM_ALTA, n_zona)

    # Medidas da consulta satisfeitas por candidato (pista para a LLM em lista).
    bonus_med = pares["bonus_medidas"] if "bonus_medidas" in pares.columns else None
    top_k_por_consulta: dict[str, pd.DataFrame] = {}
    if adjudica and ADJUDICACAO == "lista":
        for cod, grupo in ordenado.groupby("codigo_efisco"):
            top_k_por_consulta[cod] = grupo.head(K_LISTA)

    ajustes: dict = {}          # idx do par -> (score_graphrag, justificativa)
    n_llm = n_voto = n_erro = 0
    a_adjudicar = alvos[:LLM_MAX_ADJUDICACOES] if adjudica else []
    restantes = alvos[len(a_adjudicar):] if adjudica else alvos

    def _contexto_tfidf(i_cons: int) -> list:
        vizinhos_idx = sorted(
            G_sim[i_cons], key=lambda v: -G_sim[i_cons][v].get("peso", 0)
        )[:5] if i_cons in G_sim else []
        ctx = []
        for v in vizinhos_idx:
            for cand in top3_por_consulta.get(v, []):
                j = cand["idx_catalogo"]
                ctx.append({"tipo": "vizinho_tfidf", "efisco": itens_e[v],
                            "catmat": itens_c[j] if j < len(itens_c) else "",
                            "score": cand["score"]})
        return ctx

    def _adjudicar_uma(r) -> tuple:
        """Devolve (lista de (idx_par, score, justificativa), erro) para uma consulta."""
        i_cons = r.idx_consulta
        texto_efisco = itens_e[i_cons]
        ctx_gr = gr.busca_local(gr_config, gr_artefatos, texto_efisco[:300]) if gr_ok else ""
        try:
            if ADJUDICACAO == "lista":
                grupo = top_k_por_consulta.get(r.codigo_efisco)
                if grupo is None or grupo.empty:
                    return [], None
                cands = []
                for rr in grupo.itertuples():
                    cands.append({
                        "codigo": rr.codigo_catmat,
                        "texto": itens_c[rr.idx_catalogo] if rr.idx_catalogo < len(itens_c) else "",
                        "score_neural": float(rr.score_neural),
                        "deducao": deducoes.get((rr.codigo_efisco, rr.codigo_catmat), {}).get("deducao", ""),
                        "medidas": float(bonus_med.loc[rr.Index]) if bonus_med is not None else None,
                    })
                ver = _adjudicar_lista_llm(texto_efisco, cands, ctx_gr, client, cache_adj, LLM_MODEL_PADRAO)
                saida = []
                for rr, nota in zip(grupo.itertuples(), ver["notas"]):
                    misto = (1.0 - PESO_ADJUDICACAO_LISTA) * float(rr.score_neural) + PESO_ADJUDICACAO_LISTA * nota
                    saida.append((rr.Index, misto, ver["justificativa"]))
                return saida, None
            par_dict = {
                "efisco": texto_efisco,
                "catmat": itens_c[r.idx_catalogo] if r.idx_catalogo < len(itens_c) else "",
                "score_neural": float(r.score_neural),
                "deducao": deducoes.get((r.codigo_efisco, r.codigo_catmat), {}).get("deducao", ""),
            }
            ver = _adjudicar_graphrag_llm(par_dict, _contexto_tfidf(i_cons), ctx_gr,
                                         client, cache_adj, LLM_MODEL_PADRAO)
            return [(r.Index, ver["score"], ver["justificativa"])], None
        except Exception as exc:
            return [], exc

    if a_adjudicar:
        from concurrent.futures import ThreadPoolExecutor
        log.info("[6] Adjudicando %d consultas em modo '%s' (%d threads)...",
                 len(a_adjudicar), ADJUDICACAO, LLM_WORKERS)
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            resultados = list(pool.map(_adjudicar_uma, a_adjudicar))
        for r, (saida, erro) in zip(a_adjudicar, resultados):
            if erro is not None:
                n_erro += 1
                if n_erro <= 3:
                    log.warning("[6] Adjudicacao falhou: %s", str(erro)[:140])
                restantes.append(r)
                continue
            if saida:
                n_llm += 1
                for idx, score, justif in saida:
                    ajustes[idx] = (score, justif)

    # Sem LLM (ou além do teto): votação pelos vizinhos TF-IDF, como antes.
    for r in restantes:
        ctx = _contexto_tfidf(r.idx_consulta)
        if ctx:
            s_viz = float(np.mean([c["score"] for c in ctx]))
            ajustes[r.Index] = (min(1.0, 0.6 * float(r.score_neural) + 0.4 * s_viz), "")
            n_voto += 1

    for idx, (score, justif) in ajustes.items():
        pares.at[idx, "score_graphrag"] = score
        pares.at[idx, "graphrag_justificativa"] = justif

    if client is not None:
        _salvar_cache_extracao(cache_adj, cache_graphrag_path())

    print("\n=== [6] GRAPHRAG (adjudicação) ===")
    print(f"  Indice Microsoft        : {'OK' if gr_ok else 'não usado'}")
    print(f"  Arestas grafo TF-IDF    : {G_sim.number_of_edges()}")
    print(f"  Zona cinzenta (margem)  : {n_zona} consultas")
    print(f"  Adjudicacao             : modo {ADJUDICACAO if adjudica else 'nenhuma'} | "
          f"LLM {n_llm} | votacao {n_voto}" + (f" | erros {n_erro}" if n_erro else ""))
    if ajustes:
        deltas = [abs(pares.at[i, "score_graphrag"] - pares.at[i, "score_neural"])
                  for i in ajustes]
        print(f"  Ajuste medio de score   : {np.mean(deltas):.4f}")

    estado.update({
        "pares": pares,
        "grafo_sim": G_sim,
        "stats_fase6": {
            "graphrag_microsoft": gr_ok,
            "n_arestas_sim": G_sim.number_of_edges(),
            "n_zona_cinzenta": n_zona,
            "criterio_zona": f"margem top1-top2 < {LIMIAR_MARGEM_ALTA}, as menores primeiro",
            "modo_adjudicacao": ADJUDICACAO if adjudica else "nenhuma",
            "k_lista": K_LISTA,
            "peso_adjudicacao_lista": PESO_ADJUDICACAO_LISTA,
            "adjudicacao_llm": n_llm,
            "adjudicacao_votacao": n_voto,
            "adjudicacao_erros": n_erro,
        },
    })
    return estado


# ---------------------------------------------------------------------------
# FASE [8] — CONFIANÇA (roda antes da [7]: o grafo só recebe o que foi decidido)
# ---------------------------------------------------------------------------

def fase_8_confianca(estado: dict) -> dict:
    """
    Fase [8]: combina as três camadas, RANQUEIA os candidatos de cada consulta
    e elege o top-1. Cada par sai com Alta/Média/Baixa e uma trilha auditável.
    """
    log.info("=" * 68)
    log.info("[8] CONFIANÇA — Combinação, ranqueamento e trilha")
    log.info("=" * 68)

    pares = estado["pares"]
    deducoes = estado.get("deducoes", {})

    if pares.empty:
        estado["stats_fase8"] = {}
        return estado

    s_ont, deducao_col, explic_col = [], [], []
    for r in pares.itertuples():
        d = deducoes.get((r.codigo_efisco, r.codigo_catmat), {})
        s_ont.append(float(d.get("score", 0.3)))
        deducao_col.append(d.get("deducao", "incompleto"))
        explic_col.append(d.get("explicacao", []))

    pares["score_ontologia"] = s_ont
    pares["deducao_ontologia"] = deducao_col
    pares["explicacao_ontologia"] = explic_col

    # Sem clip em 1: o score neural com os sinais da [5b] pode passar de 1 nos
    # primeiros colocados, e cortar ali achataria justamente o topo do ranking.
    pares["score_final"] = (
        PESOS_SCORE_FINAL["ontologia"] * pares["score_ontologia"]
        + PESOS_SCORE_FINAL["neural"]  * pares["score_neural"]
        + PESOS_SCORE_FINAL["graphrag"] * pares["score_graphrag"]
    ).clip(lower=0).round(4)

    # Veto: penalidade opcional (FATOR_VETO = 1,0 desliga; ver constante).
    if FATOR_VETO != 1.0:
        pares.loc[pares["deducao_ontologia"] == "veto", "score_final"] *= FATOR_VETO

    # Ranqueamento dentro de cada consulta — é isto que faltava antes.
    pares["rank"] = (
        pares.groupby("codigo_efisco")["score_final"]
        .rank(ascending=False, method="first").astype(int)
    )
    pares.sort_values(["codigo_efisco", "rank"], inplace=True)

    # Confiança pela MARGEM entre o 1º e o 2º candidato da consulta: é a
    # propriedade da decisão que o gabarito confirmou como preditiva (ver
    # LIMIAR_MARGEM_*). A faixa é da consulta e vale para todos os seus pares.
    grupos = pares.groupby("codigo_efisco")["score_final"]
    primeiro = grupos.first()
    segundo = grupos.apply(lambda s: s.iloc[1] if len(s) > 1 else 0.0)
    margem = (primeiro - segundo)
    pares["margem"] = pares["codigo_efisco"].map(margem).round(4)

    def _faixa(m):
        return ("Alta" if m >= LIMIAR_MARGEM_ALTA
                else ("Média" if m >= LIMIAR_MARGEM_MEDIA else "Baixa"))

    pares["confianca"] = pares["margem"].apply(_faixa)

    # Trilha de explicação (§4.1 "Governança")
    trilhas = []
    for r in pares.itertuples():
        partes = [
            f"score_final={r.score_final:.3f} -> {r.confianca} "
            f"(rank {r.rank}; margem 1o-2o da consulta {r.margem:.3f})",
            f"  Ontologia ({r.score_ontologia:.3f} x {PESOS_SCORE_FINAL['ontologia']}; "
            f"bonus [5b] {getattr(r, 'bonus_ontologia', 0.0):.1f}): {r.deducao_ontologia}",
        ]
        partes += [f"    {l}" for l in list(r.explicacao_ontologia)[:3]]
        partes.append(
            f"  Neural    ({r.score_neural:.3f} x {PESOS_SCORE_FINAL['neural']}): "
            f"fusao={getattr(r, 'score_fusao', float('nan')):.3f}, "
            f"cosine_e5={r.score_cosine:.3f}, cross={r.score_cross:.3f}"
        )
        if getattr(r, "score_sinais", 0.0):
            partes.append(
                f"    sinais (+{PESO_SINAIS} x {r.score_sinais:.3f}): "
                f"medidas={getattr(r, 'bonus_medidas', 0.0):.2f}, "
                f"graphrag_idf={getattr(r, 'bonus_graphrag_idf', 0.0):.2f}, "
                f"taxonomia={getattr(r, 'bonus_taxonomia', 0.0):.2f}"
            )
        partes.append(
            f"  GraphRAG  ({r.score_graphrag:.3f} x {PESOS_SCORE_FINAL['graphrag']})"
            + (f": {r.graphrag_justificativa}" if r.graphrag_justificativa else "")
        )
        trilhas.append("\n".join(partes))
    pares["trilha"] = trilhas

    top1 = pares[pares["rank"] == 1]
    por_conf = top1["confianca"].value_counts().to_dict()

    print("\n=== [8] CONFIANÇA ===")
    print(f"  Pares ranqueados        : {len(pares)}")
    print(f"  Consultas com decisão   : {len(top1)}")
    print(f"  Score final médio (top1): {top1['score_final'].mean():.3f}")
    for c in ("Alta", "Média", "Baixa"):
        n = por_conf.get(c, 0)
        print(f"  {c:6s}                 : {n:5d}  ({100*n/max(len(top1),1):.1f}%)")

    estado.update({
        "pares": pares,
        "top1": top1,
        "stats_fase8": {
            "n_pares": len(pares),
            "n_decisoes": len(top1),
            "score_medio_top1": float(top1["score_final"].mean()),
            "por_confianca": {k: int(v) for k, v in por_conf.items()},
        },
    })
    return estado


# ---------------------------------------------------------------------------
# AVALIAÇÃO — recall@k, MRR, precisão do top-1
# É a medição que a versão anterior não podia fazer: só existe quando há
# candidatos gerados e um ranking de verdade.
# ---------------------------------------------------------------------------

def avaliacao_retrieval(estado: dict) -> dict:
    """
    Mede a recuperação contra o gabarito.

    recall@k : fração de consultas cujo CATMAT correto aparece no top-k
    MRR      : média de 1/posição do primeiro acerto
    P@1      : fração de consultas cujo top-1 está correto
    """
    log.info("=" * 68)
    log.info("[*] AVALIAÇÃO — recall@k / MRR / precisão do top-1")
    log.info("=" * 68)

    pares, gold_map = estado["pares"], estado["gold_map"]
    if pares.empty:
        estado["stats_avaliacao"] = {}
        return estado

    ks = [1, 3, 5, 10, K_AVALIACAO]
    ks = sorted(set(k for k in ks if k <= max(K_RERANK, 1)))

    def _metricas(ranked: dict[str, list[str]]) -> dict:
        acertos = {k: 0 for k in ks}
        rr_total = 0.0
        n_aval = 0
        for cod_e, alvos in gold_map.items():
            if cod_e not in ranked or not alvos:
                continue
            n_aval += 1
            pos = next((i for i, c in enumerate(ranked[cod_e]) if c in alvos), None)
            if pos is not None:
                rr_total += 1.0 / (pos + 1)
                for k in ks:
                    if pos < k:
                        acertos[k] += 1
        return {"n_avaliados": n_aval, "mrr": rr_total / max(n_aval, 1),
                **{f"recall_at_{k}": acertos[k] / max(n_aval, 1) for k in ks}}

    def _ranking_por(coluna: str) -> dict[str, list[str]]:
        ranked: dict[str, list[str]] = defaultdict(list)
        ordenado = pares.sort_values(["codigo_efisco", coluna], ascending=[True, False])
        for r in ordenado.itertuples():
            ranked[r.codigo_efisco].append(r.codigo_catmat)
        return ranked

    # O ranking oficial é o da fase [8] (score_final).
    ranked_final: dict[str, list[str]] = defaultdict(list)
    for r in pares.sort_values("rank").itertuples():
        ranked_final[r.codigo_efisco].append(r.codigo_catmat)
    resultado = {
        **_metricas(ranked_final),
        "recall_blocking": estado["stats_fase3"]["recall_blocking"],
    }
    resultado["precisao_at_1"] = resultado.get("recall_at_1", 0.0)

    # Ablação: o mesmo top-K reordenado por cada camada isolada. É o que diz
    # quanto cada fase acrescentou — ou tirou — sem rodar o pipeline de novo.
    # A fase [6] só entra no score final (0,8 neural + 0,2 adjudicação): a coluna
    # `score_graphrag` isolada mistura pares adjudicados e não adjudicados numa
    # mesma consulta e não é um ranking coerente — por isso não há linha dela.
    camadas = [
        ("cosseno E5 (só o bi-encoder)", "score_cosine"),
        ("fusão E5+TF-IDF (estágio 1)", "score_fusao"),
        ("+ cross-encoder (estágio 2)", "score_neural_base"),
        ("+ sinais simbólicos (fase 5b)", "score_neural"),
        ("+ adjudicação (fase 6) = score final", "score_final"),
    ]
    ablacoes = {}
    for rotulo, coluna in camadas:
        if coluna in pares.columns:
            ablacoes[rotulo] = _metricas(_ranking_por(coluna))

    print("\n=== [*] AVALIAÇÃO DE RETRIEVAL ===")
    print(f"  Consultas avaliadas     : {resultado['n_avaliados']}")
    print(f"  Recall do blocking (teto): {100*resultado['recall_blocking']:.1f}%")
    print(f"  MRR                     : {resultado['mrr']:.4f}")
    for k in ks:
        print(f"  Recall@{k:<2d}               : {100*resultado[f'recall_at_{k}']:.1f}%")
    print(f"  Precisão@1              : {100*resultado['precisao_at_1']:.1f}%")
    print("\n  Ablação (mesmo top-K, reordenado por cada camada):")
    print(f"    {'camada':<36}{'MRR':>8}{'R@1':>8}{'R@3':>8}{'R@10':>8}")
    for rotulo, m in ablacoes.items():
        print(f"    {rotulo:<36}{m['mrr']:>8.4f}{100*m['recall_at_1']:>7.1f}%"
              f"{100*m['recall_at_3']:>7.1f}%{100*m.get('recall_at_10', 0):>7.1f}%")

    # As faixas de confiança valem o que a sua precisão medida vale: P@1 e R@3
    # do gabarito dentro de cada faixa. É o que permite dizer ao curador quanto
    # do que sai como "Alta" está certo.
    por_faixa = {}
    if "confianca" in pares.columns:
        top1 = pares[pares["rank"] == 1]
        for faixa in ("Alta", "Média", "Baixa"):
            sub = top1[top1["confianca"] == faixa]
            if sub.empty:
                continue
            cods = set(sub["codigo_efisco"])
            m = _metricas({c: l for c, l in ranked_final.items() if c in cods})
            por_faixa[faixa] = {"n": int(len(sub)), "fracao": round(len(sub) / len(top1), 4),
                                "precisao_at_1": round(m["recall_at_1"], 4),
                                "recall_at_3": round(m["recall_at_3"], 4)}
        print("\n  Confiança (margem 1o-2o) contra o gabarito:")
        print(f"    {'faixa':<8}{'n':>6}{'%':>8}{'P@1':>8}{'R@3':>8}")
        for faixa, m in por_faixa.items():
            print(f"    {faixa:<8}{m['n']:>6}{100*m['fracao']:>7.1f}%"
                  f"{100*m['precisao_at_1']:>7.1f}%{100*m['recall_at_3']:>7.1f}%")

    resultado["ablacoes"] = ablacoes
    resultado["por_confianca"] = por_faixa
    estado["stats_avaliacao"] = resultado
    return estado


# ---------------------------------------------------------------------------
# FASE [7] — GRAFO UNIFICADO
# Papel 1 da §4.2. Só entra :equivalenteA o que passou do limiar — o resto
# entra como :candidatoA, para não afirmar equivalência que ninguém decidiu.
# ---------------------------------------------------------------------------

def fase_7_grafo_unificado(estado: dict) -> dict:
    """Fase [7]: materializa o grafo unificado (itens + PDMs + classes)."""
    log.info("=" * 68)
    log.info("[7] GRAFO UNIFICADO — Materialização")
    log.info("=" * 68)

    consultas, catalogo = estado["consultas"], estado["catalogo"]
    pares = estado["pares"]

    Gu = nx.DiGraph(nome="Grafo Unificado CATMAT<->eFisco", data=HOJE)

    for _, row in consultas.iterrows():
        Gu.add_node(f"eFisco_{row['codigo_efisco']}", tipo="eFisco",
                    texto=str(row["item_efisco"])[:120],
                    pdm=row.get("pdm_ancoragem", ""))
    for _, row in catalogo.iterrows():
        Gu.add_node(f"CATMAT_{row['codigo_catmat']}", tipo="CATMAT",
                    texto=str(row["item_catmat"])[:120],
                    pdm=row.get("pdm", ""))

    n_equiv = n_cand = 0
    for r in pares.itertuples():
        no_e, no_c = f"eFisco_{r.codigo_efisco}", f"CATMAT_{r.codigo_catmat}"

        # Só afirma equivalência quando a decisão a sustenta: top-1 com margem
        # que não seja "Baixa" e sem veto da ontologia.
        if r.rank == 1 and r.confianca != "Baixa" and r.deducao_ontologia != "veto":
            relacao, n_equiv = "equivalenteA", n_equiv + 1
        else:
            relacao, n_cand = "candidatoA", n_cand + 1

        Gu.add_edge(no_e, no_c, relacao=relacao,
                    score=float(r.score_final),
                    rank=int(r.rank),
                    confianca=r.confianca,
                    deducao_ontologia=r.deducao_ontologia,
                    score_ontologia=float(r.score_ontologia),
                    score_neural=float(r.score_neural),
                    score_graphrag=float(r.score_graphrag))

    for _, row in consultas.iterrows():
        pdm = row.get("pdm_ancoragem", "")
        if pdm:
            Gu.add_node(f"PDM_{pdm}", tipo="PDM", label=pdm)
            Gu.add_edge(f"eFisco_{row['codigo_efisco']}", f"PDM_{pdm}",
                        relacao="mapeadoAoPDM", peso=1.0)
    for _, row in catalogo.iterrows():
        pdm = row.get("pdm", "")
        if pdm:
            Gu.add_node(f"PDM_{pdm}", tipo="PDM", label=pdm)
            Gu.add_edge(f"CATMAT_{row['codigo_catmat']}", f"PDM_{pdm}",
                        relacao="mapeadoAoPDM", peso=1.0)

    destino = dir_resultados() / "grafo_unificado.graphml"
    _xml_invalido = re.compile(r"[^\x09\x0A\x0D\x20-퟿-�]")

    Gx = Gu.copy()
    for n, d in Gx.nodes(data=True):
        for k, v in list(d.items()):
            Gx.nodes[n][k] = _xml_invalido.sub("", v if isinstance(v, str) else str(v)) \
                if not isinstance(v, (int, float, bool)) else v
    for u, v, d in Gx.edges(data=True):
        for k, val in list(d.items()):
            Gx[u][v][k] = _xml_invalido.sub("", val if isinstance(val, str) else str(val)) \
                if not isinstance(val, (int, float, bool)) else val
    nx.write_graphml(Gx, destino)

    print("\n=== [7] GRAFO UNIFICADO ===")
    print(f"  Nós                     : {Gu.number_of_nodes()}")
    print(f"  Arestas                 : {Gu.number_of_edges()}")
    print(f"  :equivalenteA afirmadas : {n_equiv}")
    print(f"  :candidatoA (não decid.): {n_cand}")
    print(f"  Exportado               : {destino.name}")

    estado.update({
        "grafo_unificado": Gu,
        "stats_fase7": {
            "n_nos": Gu.number_of_nodes(),
            "n_arestas": Gu.number_of_edges(),
            "n_equivalenteA": n_equiv,
            "n_candidatoA": n_cand,
        },
    })
    return estado


# ---------------------------------------------------------------------------
# FASE [8b] — EXPLICABILIDADE EM LINGUAGEM NATURAL
# ---------------------------------------------------------------------------

def fase_8b_explicabilidade_llm(estado: dict) -> dict:
    """Gera explicação textual para as decisões de menor confiança (curadoria)."""
    log.info("=" * 68)
    log.info("[8b] EXPLICABILIDADE — Justificativa em linguagem natural")
    log.info("=" * 68)

    top1 = estado.get("top1")
    if top1 is None or top1.empty:
        estado["stats_fase8b"] = {"n_explicacoes": 0}
        return estado

    client, motivo = _construir_cliente_openai()
    if client is None:
        log.warning("[8b] LLM indisponível (%s) — sem explicações.", motivo)
        estado["stats_fase8b"] = {"n_explicacoes": 0, "motivo_skip": motivo}
        return estado

    cache = _carregar_cache_extracao(cache_graphrag_path())
    alvos = top1[top1["confianca"] != "Alta"].head(60)

    explicacoes = {}
    for r in alvos.itertuples():
        chave = _hash_texto("explica||" + str(r.codigo_efisco) + str(r.codigo_catmat))
        if chave in cache:
            explicacoes[r.Index] = cache[chave].get("texto", "")
            continue
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_PADRAO, max_tokens=120,
                messages=[{"role": "user", "content":
                    "Explique em UMA frase, para "
                    f"{perfil_ativo().papel('auditor_curto')}, por que "
                    "o casamento abaixo ficou com confiança "
                    f"{r.confianca} (score {r.score_final:.2f}).\n\n{r.trilha[:700]}"}],
            )
            texto = (resp.choices[0].message.content or "").strip()
            cache[chave] = {"texto": texto}
            explicacoes[r.Index] = texto
        except Exception as exc:
            log.warning("[8b] Falhou: %s", str(exc)[:140])
            break

    _salvar_cache_extracao(cache, cache_graphrag_path())

    pares = estado["pares"]
    pares["explicacao_llm"] = ""
    for idx, texto in explicacoes.items():
        pares.at[idx, "explicacao_llm"] = texto

    print(f"\n=== [8b] EXPLICABILIDADE ===\n  Explicações geradas     : {len(explicacoes)}")
    estado["pares"] = pares
    estado["stats_fase8b"] = {"n_explicacoes": len(explicacoes)}
    return estado


# ---------------------------------------------------------------------------
# FASE [9] — ANÁLISE GLOBAL / SENSEMAKING
# Papel 3 da §4.2, na medida em que os dados permitem: comunidades e
# deduplicação (§3.3-4), OOV por PDM, anomalias para curadoria.
# ---------------------------------------------------------------------------

def fase_9_analise_global(estado: dict) -> dict:
    """Fase [9]: comunidades, deduplicação, OOV e anomalias."""
    log.info("=" * 68)
    log.info("[9] ANÁLISE GLOBAL — Comunidades, deduplicação, anomalias")
    log.info("=" * 68)

    consultas = estado["consultas"]
    pares, top1 = estado["pares"], estado.get("top1")
    G_sim = estado.get("grafo_sim", nx.Graph())
    oov = estado.get("oov_por_consulta", [[]] * len(consultas))

    # --- 9.1 Detecção de comunidade (§3.3-4) -------------------------------
    comunidades: list[set] = []
    if G_sim.number_of_edges() > 0:
        try:
            comunidades = nx.community.louvain_communities(G_sim, seed=42, weight="peso")
        except Exception as exc:
            log.warning("Louvain falhou (%s) — usando componentes conexas.", exc)
            comunidades = list(nx.connected_components(G_sim))

    comunidades = [c for c in comunidades if len(c) > 1]
    tam_comunidades = sorted((len(c) for c in comunidades), reverse=True)

    # --- 9.2 Deduplicação: consultas da mesma comunidade que casaram o mesmo
    #         CATMAT são candidatas a duplicata no e-Fisco.
    duplicatas: list[dict] = []
    if top1 is not None and not top1.empty:
        catmat_por_consulta = {r.idx_consulta: r.codigo_catmat for r in top1.itertuples()}
        for com in comunidades:
            por_catmat: dict[str, list[int]] = defaultdict(list)
            for i in com:
                cc = catmat_por_consulta.get(i)
                if cc:
                    por_catmat[cc].append(i)
            for cc, membros in por_catmat.items():
                if len(membros) > 1:
                    duplicatas.append({
                        "codigo_catmat": cc,
                        "n_itens_efisco": len(membros),
                        "codigos_efisco": [
                            consultas.iloc[m]["codigo_efisco"] for m in membros
                        ][:10],
                    })
    duplicatas.sort(key=lambda d: -d["n_itens_efisco"])

    # --- 9.3 OOV por PDM ---------------------------------------------------
    consultas["n_oov"] = [len(o) for o in oov]
    oov_por_pdm = (
        consultas.groupby(consultas["pdm_ancoragem"].replace("", "SEM_PDM"))["n_oov"]
        .mean().sort_values(ascending=False)
    )

    # --- 9.4 Anomalias: gabarito diz OK, o pipeline não achou --------------
    gold_map = estado["gold_map"]
    anomalias = []
    if top1 is not None and not top1.empty:
        for r in top1.itertuples():
            alvos = gold_map.get(r.codigo_efisco, set())
            if alvos and r.codigo_catmat not in alvos:
                anomalias.append({
                    "codigo_efisco": r.codigo_efisco,
                    "predito": r.codigo_catmat,
                    "esperado": sorted(alvos)[:3],
                    "score": float(r.score_final),
                    "confianca": r.confianca,
                })

    # --- 9.5 Por PDM -------------------------------------------------------
    if top1 is not None and not top1.empty:
        t = top1.merge(
            consultas[["codigo_efisco", "pdm_ancoragem"]], on="codigo_efisco", how="left"
        )
        por_pdm = t.groupby(t["pdm_ancoragem"].replace("", "SEM_PDM")).agg(
            n=("score_final", "size"),
            score_medio=("score_final", "mean"),
            n_alta=("confianca", lambda x: (x == "Alta").sum()),
        ).sort_values("n", ascending=False)
    else:
        por_pdm = pd.DataFrame()

    print("\n=== [9] ANÁLISE GLOBAL ===")
    print(f"  Comunidades (>1 item)   : {len(comunidades)}")
    if tam_comunidades:
        print(f"  Maiores comunidades     : {tam_comunidades[:8]}")
    print(f"  Grupos de duplicatas    : {len(duplicatas)}")
    if duplicatas:
        d = duplicatas[0]
        print(f"    ex.: {d['n_itens_efisco']} itens e-Fisco -> CATMAT {d['codigo_catmat']}")
    print(f"  Anomalias (top1 != gab.): {len(anomalias)}")
    print("\n  Top-5 PDMs por volume:")
    for pdm, row in por_pdm.head(5).iterrows():
        print(f"    {str(pdm)[:34]:34s} n={int(row['n']):4d}  "
              f"score={row['score_medio']:.3f}  alta={int(row['n_alta'])}")
    print("\n  PDMs com maior OOV médio:")
    for pdm, v in oov_por_pdm.head(5).items():
        print(f"    {str(pdm)[:34]:34s} OOV={v:.1f}")

    # --- 9.6 Gráficos ------------------------------------------------------
    try:
        _plotar_analise(estado, tam_comunidades)
    except Exception as exc:
        log.warning("Gráficos falharam: %s", exc)

    estado.update({
        "comunidades": comunidades,
        "duplicatas": duplicatas,
        "anomalias": anomalias,
        "stats_fase9": {
            "n_comunidades": len(comunidades),
            "tamanhos_comunidades": tam_comunidades[:20],
            "n_grupos_duplicatas": len(duplicatas),
            "n_anomalias": len(anomalias),
        },
    })
    return estado


def _plotar_analise(estado: dict, tam_comunidades: list) -> None:
    """Painel de 3 gráficos: distribuição de score, confiança e comunidades."""
    top1 = estado.get("top1")
    if top1 is None or top1.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_color("#555")

    axes[0].hist(top1["score_final"], bins=30, color="#2ca02c", edgecolor="white", alpha=.85)
    axes[0].set_title("Score final do top-1", color="white")
    axes[0].set_xlabel("score", color="white")
    axes[0].set_ylabel("consultas", color="white")

    conf = top1["confianca"].value_counts().reindex(["Alta", "Média", "Baixa"]).fillna(0)
    axes[1].bar(conf.index, conf.values, color=["#2ca02c", "#ff7f0e", "#d62728"])
    axes[1].set_title("Distribuição de confiança", color="white")
    axes[1].set_ylabel("consultas", color="white")

    if tam_comunidades:
        axes[2].hist(tam_comunidades, bins=min(30, len(tam_comunidades)),
                     color="#1f77b4", edgecolor="white", alpha=.85)
        axes[2].set_title("Tamanho das comunidades", color="white")
        axes[2].set_xlabel("itens por comunidade", color="white")
    else:
        axes[2].text(.5, .5, "sem comunidades", ha="center", color="white")

    for ax in axes:
        ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig(dir_resultados() / "analise_global.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# EXPORTAÇÃO
# ---------------------------------------------------------------------------

def exportar_resultados(estado: dict) -> None:
    """Grava CSV completo, YAML de apresentação e stats JSON."""
    pares = estado["pares"]
    top1 = estado.get("top1")

    cols = [
        "codigo_efisco", "codigo_catmat", "rank", "score_final", "confianca",
        "score_ontologia", "deducao_ontologia", "score_neural", "score_neural_base",
        "score_fusao", "score_cosine", "score_cross", "score_sinais",
        "bonus_medidas", "bonus_graphrag_idf", "bonus_taxonomia", "score_graphrag",
        "graphrag_justificativa", "trilha",
    ]
    cols = [c for c in cols if c in pares.columns]
    pares[cols].to_csv(dir_resultados() / "resultado_pipeline_completo.csv",
                       index=False, sep="|", encoding="utf-8-sig")

    stats = {
        "fase0": estado.get("stats_fase0", {}),
        "fase1": estado.get("stats_fase1", {}),
        "fase2": estado.get("stats_fase2", {}),
        "fase3": estado.get("stats_fase3", {}),
        "fase4": estado.get("stats_fase4", {}),
        "fase5": estado.get("stats_fase5", {}),
        "fase5b": estado.get("stats_fase5b", {}),
        "fase6": estado.get("stats_fase6", {}),
        "fase7": estado.get("stats_fase7", {}),
        "fase8": estado.get("stats_fase8", {}),
        "fase8b": estado.get("stats_fase8b", {}),
        "fase9": estado.get("stats_fase9", {}),
        "avaliacao": estado.get("stats_avaliacao", {}),
        "tempos_execucao": CRONO.tempos,
        "config": {
            "modelo_embedding": MODELO_EMBEDDING,
            "modelo_cross_encoder": MODELO_CROSS_ENCODER,
            "modelo_llm": LLM_MODEL_PADRAO,
            "k_rerank": K_RERANK,
            "pesos_score_final": PESOS_SCORE_FINAL,
            "fator_veto": FATOR_VETO,
            "pesos_fusao": PESOS_FUSAO,
            "texto_neural": TEXTO_NEURAL,
            "peso_sinais": PESO_SINAIS,
            "pesos_rel_sinais": PESOS_REL_SINAIS,
            "usar_llm": USAR_LLM,
            "usar_rede_semantica": USAR_REDE,
            "usar_cross_encoder": USAR_CROSS,
            "usar_sinais": USAR_SINAIS,
            "llm_max_adjudicacoes": LLM_MAX_ADJUDICACOES,
            "adjudicacao": ADJUDICACAO,
            "k_lista": K_LISTA,
            "peso_adjudicacao_lista": PESO_ADJUDICACAO_LISTA,
            "contexto_ms": USAR_CONTEXTO_MS,
            "limiares_margem": {"alta": LIMIAR_MARGEM_ALTA, "media": LIMIAR_MARGEM_MEDIA},
        },
        "perfil": perfil_ativo().resumo(),
    }

    def _limpar(o):
        if isinstance(o, dict):
            return {k: _limpar(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_limpar(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    with open(dir_resultados() / "stats_pipeline.json", "w", encoding="utf-8") as fh:
        json.dump(_limpar(stats), fh, ensure_ascii=False, indent=2)

    # YAML de apresentação: amostras por faixa de confiança
    if top1 is not None and not top1.empty:
        amostras = {}
        for faixa in ("Alta", "Média", "Baixa"):
            sub = top1[top1["confianca"] == faixa].head(15)
            amostras[faixa] = [
                {
                    "codigo_efisco": r.codigo_efisco,
                    "codigo_catmat": r.codigo_catmat,
                    "score": float(r.score_final),
                    "deducao": r.deducao_ontologia,
                    "trilha": r.trilha.split("\n"),
                }
                for r in sub.itertuples()
            ]
        saida = {
            "resumo": _limpar(stats["avaliacao"]),
            "amostras_por_confianca": amostras,
            "duplicatas": estado.get("duplicatas", [])[:20],
            "anomalias_para_curadoria": estado.get("anomalias", [])[:30],
        }
        with open(dir_resultados() / "resultado_pipeline.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(saida, fh, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# ORQUESTRAÇÃO
# ---------------------------------------------------------------------------

def testar_llm_conexao() -> bool:
    """Smoke test: valida chave, SDK e acesso ao modelo."""
    client, motivo = _construir_cliente_openai()
    if client is None:
        print(f"LLM indisponível: {motivo}")
        return False
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL_PADRAO, max_tokens=20,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content":
                       'Responda em JSON: {"ok": true} e nada mais.'}],
        )
        print(f"OK — {LLM_MODEL_PADRAO} respondeu: {r.choices[0].message.content}")
        print(f"tokens: {r.usage.prompt_tokens} + {r.usage.completion_tokens}")
        return True
    except Exception as exc:
        print(f"FALHOU: {type(exc).__name__}: {str(exc)[:300]}")
        return False


def executar_pipeline_completo() -> dict:
    """Executa as fases [0]–[9] em sequência, cronometrando cada uma."""
    ctx = contexto()
    print("\n" + "=" * 68)
    print(f"  PIPELINE NEURO-SIMBÓLICO — CATMAT <-> e-Fisco [{ctx.dataset.nome}]")
    print(f"  Domínio: {ctx.perfil.nome} | Rede semântica + OWL + Neural + GraphRAG")
    print(f"  LLM: {LLM_MODEL_PADRAO if USAR_LLM else 'DESLIGADA'} | "
          f"Rede semântica: {'ativa' if USAR_REDE else 'ABLAÇÃO'} | "
          f"Cross-encoder: {'ativo' if USAR_CROSS else 'desligado'} | "
          f"Sinais [5b]: {'ativos' if USAR_SINAIS else 'desligados'}")
    print("=" * 68)

    t_inicio = time.time()
    contexto().dataset.preparar_diretorios()

    with CRONO.medir("fase_0_fontes"):
        estado = fase_0_fontes()
    with CRONO.medir("fase_1_rede_semantica"):
        estado = fase_1_rede_semantica(estado)
    with CRONO.medir("fase_2_extracao"):
        estado = fase_2_extracao(estado)
    with CRONO.medir("fase_3_blocking"):
        estado = fase_3_blocking(estado)
    with CRONO.medir("fase_4_ontologia"):
        estado = fase_4_ontologia(estado)
    with CRONO.medir("fase_5_matcher_neural"):
        estado = fase_5_matcher_neural(estado)
    with CRONO.medir("fase_5b_sinais_simbolicos"):
        estado = fase_5b_sinais_simbolicos(estado)
    with CRONO.medir("fase_6_graphrag"):
        estado = fase_6_graphrag(estado)
    with CRONO.medir("fase_8_confianca"):
        estado = fase_8_confianca(estado)
    with CRONO.medir("avaliacao_retrieval"):
        estado = avaliacao_retrieval(estado)
    with CRONO.medir("fase_7_grafo_unificado"):
        estado = fase_7_grafo_unificado(estado)
    with CRONO.medir("fase_8b_explicabilidade"):
        estado = fase_8b_explicabilidade_llm(estado)
    with CRONO.medir("fase_9_analise_global"):
        estado = fase_9_analise_global(estado)

    exportar_resultados(estado)

    saida = dir_resultados()
    print("\n" + "=" * 68)
    print(f"  PIPELINE CONCLUÍDO em {time.time() - t_inicio:.1f}s")
    print(f"  Saída          : {saida}")
    for nome in ("resultado_pipeline_completo.csv", "stats_pipeline.json",
                 "grafo_unificado.graphml", "ontologia.owl",
                 "rede_semantica_skos.ttl"):
        print(f"    {nome}")
    if ctx.perfil.lacunas:
        print("  Lacunas do perfil de domínio (ver PERFIL no stats_pipeline.json):")
        for lacuna in ctx.perfil.lacunas:
            print(f"    - {lacuna}")
    print("=" * 68)
    return estado


def _cli() -> None:
    """CLI do pipeline. Escolhe o dataset ANTES de qualquer leitura de dados."""
    global USAR_LLM, USAR_REDE, USAR_CROSS, USAR_SINAIS, LLM_MODEL_PADRAO
    global TEXTO_NEURAL, LLM_MAX_ADJUDICACOES, K_RERANK, FATOR_VETO
    global ADJUDICACAO, K_LISTA, USAR_CONTEXTO_MS, PESO_ADJUDICACAO_LISTA

    import argparse

    from .config import ativar, listar_datasets, listar_perfis

    ap = argparse.ArgumentParser(
        description="Pipeline neuro-simbólico de casamento CATMAT <-> e-Fisco.")
    ap.add_argument("--dataset", default="",
                    help=f"dataset a processar. Disponíveis: {', '.join(listar_datasets())}")
    ap.add_argument("--perfil", default="",
                    help=f"força outro perfil de domínio ({', '.join(listar_perfis())})")
    ap.add_argument("--smoke", action="store_true",
                    help="testa a conexão com a API e sai")
    ap.add_argument("--sem-llm", action="store_true",
                    help="execução offline: regex do perfil no lugar da LLM")
    ap.add_argument("--sem-rede", action="store_true",
                    help="ablação: desliga a rede semântica da fase [1]")
    ap.add_argument("--cross", action="store_true",
                    help="liga o cross-encoder (estágio 2 da fase [5]); desligado por padrão "
                         "porque piora o ranking e custa ~30 min de CPU")
    ap.add_argument("--sem-cross", action="store_true",
                    help="(padrão) mantém o cross-encoder desligado; existe para compatibilidade")
    ap.add_argument("--sem-sinais", action="store_true",
                    help="ablação: desliga a fase [5b] (medida, GraphRAG-IDF, taxonomia)")
    ap.add_argument("--neural-texto", choices=("basico", "rede"), default=TEXTO_NEURAL,
                    help="texto da consulta na fusão: normalização básica (padrão) ou o "
                         "documento virtual expandido da rede semântica")
    ap.add_argument("--k-rerank", type=int, default=K_RERANK,
                    help=f"candidatos por consulta que seguem para as fases [5]-[8] "
                         f"(padrão: {K_RERANK})")
    ap.add_argument("--max-adjudicacoes", type=int, default=LLM_MAX_ADJUDICACOES,
                    help=f"teto de adjudicações LLM na fase [6] (padrão: {LLM_MAX_ADJUDICACOES})")
    ap.add_argument("--adjudicacao", choices=("lista", "par", "nenhuma"), default=ADJUDICACAO,
                    help="fase [6]: a LLM pontua os K melhores candidatos de uma vez (lista, "
                         "padrão), julga só o top-1 (par) ou não entra (nenhuma)")
    ap.add_argument("--k-lista", type=int, default=K_LISTA,
                    help=f"candidatos mostrados à LLM no modo lista (padrão: {K_LISTA})")
    ap.add_argument("--peso-adjudicacao", type=float, default=PESO_ADJUDICACAO_LISTA,
                    help="peso da nota da LLM no score_graphrag dos candidatos adjudicados em "
                         f"lista (padrão: {PESO_ADJUDICACAO_LISTA})")
    ap.add_argument("--contexto-ms", action="store_true",
                    help="constrói/consulta o índice Microsoft GraphRAG e passa o local_search "
                         "como contexto à adjudicação (caro: ~2 h de indexação)")
    ap.add_argument("--peso-rel", default="",
                    help="pesos relativos dos sinais da fase [5b], ex. "
                         "'medidas=0.6,graphrag_idf=1,taxonomia=0.3,ontologia=0.6'")
    ap.add_argument("--model", default="",
                    help="modelo da LLM; isola a saída em resultados/<dataset>/<modelo>/")
    ap.add_argument("--saida", default="",
                    help="subpasta de resultados/<dataset>/ para esta execução (não "
                         "sobrescreve as saídas publicadas do dataset)")
    ap.add_argument("--peso-ontologia", type=float,
                    help=f"peso da ontologia no score final (padrão "
                         f"{PESOS_SCORE_FINAL['ontologia']}); neural e graphrag são "
                         "rebalanceados na proporção original")
    ap.add_argument("--fator-veto", type=float,
                    help=f"multiplicador do score final de um par vetado (padrão {FATOR_VETO}; "
                         "1.0 desliga o veto)")
    args = ap.parse_args()

    if args.smoke:
        raise SystemExit(0 if testar_llm_conexao() else 1)

    ctx = ativar(args.dataset, perfil=args.perfil)

    if args.sem_llm:
        USAR_LLM = False
    if args.sem_rede:
        USAR_REDE = False
    if args.cross:
        USAR_CROSS = True
    if args.sem_cross:
        USAR_CROSS = False
    if args.sem_sinais:
        USAR_SINAIS = False
    TEXTO_NEURAL = args.neural_texto
    K_RERANK = args.k_rerank
    LLM_MAX_ADJUDICACOES = args.max_adjudicacoes
    ADJUDICACAO = args.adjudicacao
    K_LISTA = args.k_lista
    PESO_ADJUDICACAO_LISTA = args.peso_adjudicacao
    USAR_CONTEXTO_MS = args.contexto_ms
    for par in filter(None, (p.strip() for p in args.peso_rel.split(","))):
        nome, _, valor = par.partition("=")
        PESOS_REL_SINAIS[nome.strip()] = float(valor)
    if args.model:
        LLM_MODEL_PADRAO = args.model
        # Saída isolada por modelo, para comparar LLMs lado a lado sem que uma
        # execução sobrescreva a anterior.
        ctx.dataset.sub_resultados = args.model.replace("/", "-").replace(":", "-")
        print(f"[--model] LLM: {LLM_MODEL_PADRAO} | saída: {ctx.dataset.dir_resultados}")
    if args.saida:
        ctx.dataset.sub_resultados = re.sub(r"[^A-Za-z0-9_\-]+", "-", args.saida)
        print(f"[--saida] saída: {ctx.dataset.dir_resultados}")
    if args.peso_ontologia is not None:
        resto = 1.0 - args.peso_ontologia
        base = PESOS_SCORE_FINAL["neural"] + PESOS_SCORE_FINAL["graphrag"]
        PESOS_SCORE_FINAL["ontologia"] = args.peso_ontologia
        PESOS_SCORE_FINAL["neural"] = round(resto * PESOS_SCORE_FINAL["neural"] / base, 4)
        PESOS_SCORE_FINAL["graphrag"] = round(resto * PESOS_SCORE_FINAL["graphrag"] / base, 4)
        print(f"[--peso-ontologia] pesos do score final: {PESOS_SCORE_FINAL}")
    if args.fator_veto is not None:
        FATOR_VETO = args.fator_veto
        print(f"[--fator-veto] {FATOR_VETO}")

    executar_pipeline_completo()


if __name__ == "__main__":
    _cli()
