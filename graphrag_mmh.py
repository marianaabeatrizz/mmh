"""
graphrag_mmh.py
===============
GraphRAG do pipeline MMH em dois papéis:

(A) ADJUDICADOR — fase [6] do pipeline monolítico (`pipeline_completo_mmh.py`).
    Integração com o Microsoft GraphRAG (v3.x).

    Fluxo de indexação (roda uma vez, cached em graphrag_workspace/output/):
      1. Grava um .txt por item (e-Fisco + CATMAT) em graphrag_workspace/input/
      2. Configura GRAPHRAG_API_KEY no .env do workspace
      3. Executa `graphrag index` (extração de entidades, relações, comunidades, resumos)

    Fluxo de busca (por consulta na zona cinzenta):
      4. Carrega os artefatos parquet gerados pelo indexador
      5. Executa local_search(query=texto_efisco) via graphrag.api
      6. Retorna contexto textual para o LLM de adjudicação

(B) MÓDULO DA GRADE — `pré-proc x proc x pós-proc` (`avaliacao_modular_mmh.py`).
    Em vez de arbitrar pares já pontuados, o mesmo KG serve a dois eixos:
      - pré-processador: enriquece o texto da consulta ANTES da similaridade,
        com entidades vizinhas + rótulo da comunidade — `expandir_consultas()`;
      - pós-processador: reordena o top-K pela cobertura das entidades da
        consulta — construído sobre `construir_kg()`.
    Ver seção 7.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

log = logging.getLogger(__name__)

GR_WORKSPACE = Path(__file__).parent / "graphrag_workspace"
GR_INPUT_DIR  = GR_WORKSPACE / "input"
GR_OUTPUT_DIR = GR_WORKSPACE / "output"

# Arquivo sentinela: criado ao final de uma indexação bem-sucedida
_SENTINELA_INDEXADO = GR_OUTPUT_DIR / ".indexado_ok"


# ---------------------------------------------------------------------------
# 1. Preparação do corpus
# ---------------------------------------------------------------------------

def preparar_corpus(consultas: pd.DataFrame, catalogo: pd.DataFrame) -> int:
    """
    Grava um arquivo .txt por item em graphrag_workspace/input/.
    Retorna o número de arquivos gravados.
    Pula se os arquivos já existem (evita reindexar sem motivo).
    """
    GR_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0

    for _, row in consultas.iterrows():
        codigo = str(row.get("codigo_efisco", ""))
        texto  = str(row.get("item_efisco", "") or "")
        pdm    = str(row.get("pdm_ancoragem", "") or "")
        destino = GR_INPUT_DIR / f"efisco_{codigo}.txt"
        if not destino.exists():
            destino.write_text(
                f"Item e-Fisco {codigo}\nFamilia PDM: {pdm}\n{texto}",
                encoding="utf-8",
            )
        n += 1

    for _, row in catalogo.iterrows():
        codigo = str(row.get("codigo_catmat", ""))
        texto  = str(row.get("item_catmat", "") or "")
        pdm    = str(row.get("pdm", "") or "")
        destino = GR_INPUT_DIR / f"catmat_{codigo}.txt"
        if not destino.exists():
            destino.write_text(
                f"Item CATMAT {codigo}\nFamilia PDM: {pdm}\n{texto}",
                encoding="utf-8",
            )
        n += 1

    log.info("[GraphRAG] Corpus: %d arquivos em %s", n, GR_INPUT_DIR)
    return n


# ---------------------------------------------------------------------------
# 2. Configuração do ambiente
# ---------------------------------------------------------------------------

def configurar_env(api_key: str) -> None:
    """Grava .env no workspace com GRAPHRAG_API_KEY."""
    env_path = GR_WORKSPACE / ".env"
    env_path.write_text(f"GRAPHRAG_API_KEY={api_key}\n", encoding="utf-8")
    log.info("[GraphRAG] .env configurado em %s", env_path)


# ---------------------------------------------------------------------------
# 3. Indexação
# ---------------------------------------------------------------------------

def ja_indexado() -> bool:
    """True se o índice já foi gerado com sucesso."""
    return _SENTINELA_INDEXADO.exists()


def indexar(verbose: bool = True) -> None:
    """
    Executa `graphrag index --root <workspace>`.
    Bloqueante — pode levar 20-40 min na primeira execução.
    Grava um arquivo sentinela ao concluir.
    """
    log.info("[GraphRAG] Iniciando indexação (pode levar 20-40 min)...")
    cmd = ["graphrag", "index", "--root", str(GR_WORKSPACE)]
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        stderr = getattr(result, "stderr", "")
        raise RuntimeError(
            f"graphrag index falhou (código {result.returncode}):\n{stderr[:500]}"
        )
    _SENTINELA_INDEXADO.touch()
    log.info("[GraphRAG] Indexação concluída.")


# ---------------------------------------------------------------------------
# 4. Carregamento dos artefatos
# ---------------------------------------------------------------------------

def carregar_artefatos(config) -> dict[str, Any]:
    """
    Lê os parquets gerados pelo indexador via a API oficial do GraphRAG.
    Retorna dict com DataFrames prontos para graphrag.api.local_search.
    """
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider
    from graphrag.data_model.data_reader import DataReader

    storage  = create_storage(config.output_storage)
    provider = create_table_provider(config.table_provider, storage=storage)
    reader   = DataReader(provider)

    artefatos: dict[str, Any] = {}
    for nome in ["entities", "communities", "community_reports", "text_units", "relationships"]:
        try:
            artefatos[nome] = asyncio.run(getattr(reader, nome)())
            log.info("[GraphRAG] Carregado: %s (%d linhas)", nome, len(artefatos[nome]))
        except Exception as exc:
            log.warning("[GraphRAG] Não foi possível carregar %s: %s", nome, exc)
            artefatos[nome] = None

    return artefatos


# ---------------------------------------------------------------------------
# 5. Busca local (por item)
# ---------------------------------------------------------------------------

async def _local_search_async(config, artefatos: dict, query: str) -> str:
    """Executa graphrag.api.local_search de forma assíncrona."""
    import graphrag.api as api

    resultado, _ = await api.local_search(
        config=config,
        entities=artefatos["entities"],
        communities=artefatos["communities"],
        community_reports=artefatos["community_reports"],
        text_units=artefatos["text_units"],
        relationships=artefatos["relationships"],
        covariates=None,
        community_level=2,
        response_type="Single Paragraph",
        query=query,
    )
    return str(resultado)


def busca_local(config, artefatos: dict, query: str,
                max_chars: int = 600) -> str:
    """
    Wrapper síncrono para busca local via Microsoft GraphRAG.
    Retorna string com o contexto do grafo de conhecimento para a query.
    """
    try:
        resultado = asyncio.run(_local_search_async(config, artefatos, query))
        return resultado[:max_chars]
    except Exception as exc:
        log.warning("[GraphRAG] busca_local falhou: %s", str(exc)[:120])
        return ""


# ---------------------------------------------------------------------------
# 6. Inicialização completa (ponto de entrada para fase_6)
# ---------------------------------------------------------------------------

def inicializar(consultas: pd.DataFrame, catalogo: pd.DataFrame,
                api_key: str) -> tuple:
    """
    Ponto de entrada único para a fase [6]:
      - Prepara corpus
      - Configura env
      - Indexa (se necessário)
      - Carrega config + artefatos

    Returns: (config, artefatos)
    """
    from graphrag.config.load_config import load_config

    preparar_corpus(consultas, catalogo)
    configurar_env(api_key)

    if not ja_indexado():
        indexar(verbose=True)
    else:
        log.info("[GraphRAG] Índice já existente — pulando indexação.")

    config    = load_config(GR_WORKSPACE)
    artefatos = carregar_artefatos(config)
    return config, artefatos


# ---------------------------------------------------------------------------
# 7. PRÉ-PROCESSADOR — expansão de consultas por grafo
# ---------------------------------------------------------------------------
# Papel (B): o GraphRAG entra como PRÉ-PROCESSADOR da grade modular, não como
# adjudicador. A consulta e-Fisco é reescrita como "documento virtual": texto
# base + entidades recuperadas por busca local no grafo + rótulo da comunidade.
# Quem consome isso é o processador de similaridade seguinte (TF-IDF, fuzzy, E5).
#
# Dois modos:
#   offline (padrão) — KG construído aqui mesmo: entidades (PDM / atributos PDM /
#                      tokens), busca local por sobreposição de entidades e
#                      comunidades por Louvain. Custo zero de API.
#   ms_graphrag      — usa o índice oficial da Microsoft (local_search) e extrai
#                      do contexto retornado os termos a acrescentar. Requer
#                      OPENAI_API_KEY e o índice já construído (ver seções 1-6).
# ---------------------------------------------------------------------------

# Tokens sem valor discriminativo para virar entidade
_SW_ENTIDADE = {
    "PARA", "COMO", "TIPO", "COM", "SEM", "USO", "CADA", "DEVE",
    "ESTE", "ESSA", "PRODUTO", "ITEM", "UNIDADE", "CAIXA", "PACOTE",
    "DEVERA", "CONFORME", "ACORDO", "DADOS", "EMBALAGEM", "EMBALADO",
    "APRESENTACAO", "MATERIAL", "OUTROS", "DEMAIS",
}

# Entidade presente em mais de N itens do catálogo não discrimina nada
_MAX_ITENS_POR_ENTIDADE = 200

_CACHE_EXPANSAO_PATH = Path(__file__).parent / "cache_graphrag_expansao.json"


@dataclass
class GrafoConhecimento:
    """KG bipartito itens <-> entidades, com índices para busca local."""

    G: nx.Graph
    ents_consulta: dict[int, set[str]] = field(default_factory=dict)
    ents_catalogo: dict[int, set[str]] = field(default_factory=dict)
    indice_invertido: dict[str, list[int]] = field(default_factory=dict)
    comunidade: dict[int, int] = field(default_factory=dict)
    rotulo_comunidade: dict[int, str] = field(default_factory=dict)


def _entidades_item(texto: str, atributos: dict | None = None,
                    pdm: str = "") -> set[str]:
    """Entidades de um item: PDM + valores de atributos PDM + tokens relevantes."""
    entidades: set[str] = set()
    if pdm:
        entidades.add(f"PDM::{pdm.upper().strip()}")
    for chave, valor in (atributos or {}).items():
        if valor and chave not in ("tipo_produto", "texto_completo"):
            entidades.add(f"{chave}::{str(valor).upper().strip()[:40]}")
    tokens = [t for t in re.findall(r"[A-ZÀ-Ú]{4,}", (texto or "").upper())
              if t not in _SW_ENTIDADE]
    for tok in tokens[:10]:
        entidades.add(f"TOK::{tok}")
    return entidades


def _valor_entidade(ent: str) -> str:
    """'calibre::18G' -> '18G'; 'TOK::AGULHA' -> 'AGULHA'."""
    _, _, valor = ent.partition("::")
    return valor.strip()


def construir_kg(consultas: pd.DataFrame, catalogo: pd.DataFrame) -> GrafoConhecimento:
    """
    Constrói o KG bipartito (itens <-> entidades) e detecta comunidades.

    Nós: 'e:{i}' (consulta e-Fisco), 'c:{j}' (item CATMAT) e as entidades.
    As comunidades saem da projeção consulta <-> consulta (itens que
    compartilham >= 2 entidades), por Louvain — equivalente ao passo de
    detecção de comunidades do GraphRAG.
    """
    from preprocessamento_mmh import extrair_atributos_catmat

    G = nx.Graph()
    ents_consulta: dict[int, set[str]] = {}
    ents_catalogo: dict[int, set[str]] = {}
    indice_invertido: dict[str, list[int]] = defaultdict(list)

    col_pdm_e = "pdm_ancoragem" if "pdm_ancoragem" in consultas.columns else None

    for i, row in enumerate(consultas.itertuples(index=False)):
        pdm = (getattr(row, col_pdm_e, "") or "") if col_pdm_e else ""
        ents = _entidades_item(getattr(row, "item_efisco", "") or "", None, pdm)
        ents_consulta[i] = ents
        G.add_node(f"e:{i}", tipo="efisco", idx=i)
        for ent in ents:
            G.add_node(ent, tipo="entidade")
            G.add_edge(f"e:{i}", ent)

    for j, row in enumerate(catalogo.itertuples(index=False)):
        texto = getattr(row, "item_catmat", "") or ""
        atributos = getattr(row, "catmat_atributos", None)
        if not isinstance(atributos, dict):
            atributos = extrair_atributos_catmat(texto)
        pdm = (getattr(row, "pdm", "") or "") or atributos.get("tipo_produto", "")
        ents = _entidades_item(texto, atributos, pdm)
        ents_catalogo[j] = ents
        G.add_node(f"c:{j}", tipo="catmat", idx=j)
        for ent in ents:
            G.add_node(ent, tipo="entidade")
            G.add_edge(f"c:{j}", ent)
            indice_invertido[ent].append(j)

    kg = GrafoConhecimento(
        G=G,
        ents_consulta=ents_consulta,
        ents_catalogo=ents_catalogo,
        indice_invertido=dict(indice_invertido),
    )
    kg.comunidade, kg.rotulo_comunidade = _detectar_comunidades(kg, len(consultas))

    log.info("[GraphRAG/pré] KG: %d nós, %d arestas, %d entidades, %d comunidades",
             G.number_of_nodes(), G.number_of_edges(),
             sum(1 for _, d in G.nodes(data=True) if d.get("tipo") == "entidade"),
             len(set(kg.comunidade.values())))
    return kg


def _detectar_comunidades(kg: GrafoConhecimento, n_consultas: int) -> tuple[dict, dict]:
    """Projeta consulta <-> consulta, roda Louvain e rotula cada comunidade."""
    ent_para_consultas: dict[str, list[int]] = defaultdict(list)
    for i, ents in kg.ents_consulta.items():
        for ent in ents:
            ent_para_consultas[ent].append(i)

    G_proj = nx.Graph()
    G_proj.add_nodes_from(range(n_consultas))
    for idxs in ent_para_consultas.values():
        if len(idxs) > _MAX_ITENS_POR_ENTIDADE:
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
        comunidade = {m: cid for cid, membros in enumerate(coms) for m in membros}
    except Exception as exc:
        log.warning("[GraphRAG/pré] Louvain falhou (%s) — comunidade única.", exc)
        comunidade = {i: 0 for i in range(n_consultas)}

    # Rótulo da comunidade: entidade mais frequente entre seus membros.
    membros_por_com: dict[int, list[int]] = defaultdict(list)
    for i, com in comunidade.items():
        membros_por_com[com].append(i)

    rotulos: dict[int, str] = {}
    for com, membros in membros_por_com.items():
        if len(membros) < 2:
            continue
        contagem: Counter = Counter()
        for i in membros:
            for ent in kg.ents_consulta.get(i, ()):
                contagem[ent] += 1
        ordenadas = sorted(contagem.items(), key=lambda kv: (-kv[1], kv[0]))
        pdms = [e for e, _ in ordenadas if e.startswith("PDM::")]
        escolhida = pdms[0] if pdms else (ordenadas[0][0] if ordenadas else "")
        if escolhida:
            rotulos[com] = _valor_entidade(escolhida)

    return comunidade, rotulos


def _busca_local(kg: GrafoConhecimento, i: int, top_k: int) -> list[int]:
    """Itens do catálogo que mais compartilham entidades com a consulta i."""
    ents = kg.ents_consulta.get(i, set())
    relevantes = {e for e in ents if not e.startswith("PDM::")} or ents

    contagem: Counter = Counter()
    for ent in relevantes:
        idxs = kg.indice_invertido.get(ent)
        if not idxs or len(idxs) > _MAX_ITENS_POR_ENTIDADE:
            continue
        for j in idxs:
            contagem[j] += 1

    # Empates desempatados pelo índice, para o resultado não depender do hash seed
    ordenados = sorted(contagem.items(), key=lambda kv: (-kv[1], kv[0]))
    return [j for j, _ in ordenados[:top_k]]


def _termos_novos(candidatos: Counter, texto_base: str, max_termos: int) -> list[str]:
    """
    Termos ainda ausentes do texto base, do mais frequente ao menos.

    A deduplicação é por token: entidades distintas costumam repetir o mesmo
    valor ('ANESTESIA REGIONAL' e 'REGIONAL'), e repetir token infla a
    frequência dele no TF-IDF sem acrescentar informação.
    """
    base_tokens = set(texto_base.upper().split())
    escolhidos: list[str] = []
    tokens_escolhidos: set[str] = set()

    # Desempate pelo nome da entidade: `most_common` herda a ordem de inserção,
    # que vem de iteração sobre `set` — e essa varia com o hash seed do processo.
    for ent, _ in sorted(candidatos.items(), key=lambda kv: (-kv[1], kv[0])):
        valor = _valor_entidade(ent)
        if not valor or len(valor) < 3:
            continue
        tokens = [t for t in valor.split()
                  if t not in _SW_ENTIDADE
                  and t not in base_tokens
                  and t not in tokens_escolhidos]
        if not tokens:
            continue
        escolhidos.append(" ".join(tokens))
        tokens_escolhidos.update(tokens)
        if len(escolhidos) >= max_termos:
            break
    return escolhidos


def _carregar_cache_expansao() -> dict:
    if _CACHE_EXPANSAO_PATH.exists():
        try:
            return json.loads(_CACHE_EXPANSAO_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("[GraphRAG/pré] Cache de expansão ilegível — recomeçando.")
    return {}


def _salvar_cache_expansao(cache: dict) -> None:
    try:
        _CACHE_EXPANSAO_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8"
        )
    except OSError as exc:
        log.warning("[GraphRAG/pré] Falha ao salvar cache de expansão: %s", exc)


def _expandir_via_ms_graphrag(consultas: pd.DataFrame, catalogo: pd.DataFrame,
                              textos_base: list[str], api_key: str,
                              max_termos: int) -> list[str] | None:
    """
    Expansão usando o índice oficial da Microsoft: local_search por consulta,
    de onde se extraem os termos ausentes do texto base. Cacheado em disco
    (uma chamada LLM por consulta é caro). Retorna None se o índice não subir.
    """
    try:
        config, artefatos = inicializar(consultas, catalogo, api_key)
        if not all(v is not None for v in artefatos.values()):
            raise RuntimeError("artefatos incompletos")
    except Exception as exc:
        log.warning("[GraphRAG/pré] MS GraphRAG indisponível (%s) — modo offline.",
                    str(exc)[:120])
        return None

    cache = _carregar_cache_expansao()
    expandidos: list[str] = []
    n_api = 0

    for texto_base, texto_orig in zip(textos_base, consultas["item_efisco"].fillna("")):
        chave = hashlib.sha1(
            f"ms|{max_termos}|{texto_orig}".encode("utf-8")
        ).hexdigest()
        if chave in cache:
            termos = cache[chave]
        else:
            contexto = busca_local(config, artefatos, str(texto_orig)[:300])
            brutos = Counter(
                f"TOK::{t}" for t in re.findall(r"[A-ZÀ-Ú]{4,}", contexto.upper())
                if t not in _SW_ENTIDADE
            )
            termos = _termos_novos(brutos, texto_base, max_termos)
            cache[chave] = termos
            n_api += 1
        expandidos.append((texto_base + " " + " ".join(termos)).strip())

    if n_api:
        _salvar_cache_expansao(cache)
    log.info("[GraphRAG/pré] MS GraphRAG: %d consultas (%d chamadas novas).",
             len(expandidos), n_api)
    return expandidos


def expandir_consultas(
    consultas: pd.DataFrame,
    catalogo: pd.DataFrame,
    *,
    textos_base: list[str] | None = None,
    max_termos: int = 6,
    top_k_vizinhos: int = 6,
    incluir_comunidade: bool = True,
    usar_ms_graphrag: bool = False,
    api_key: str = "",
    kg: GrafoConhecimento | None = None,
) -> tuple[list[str], dict]:
    """
    Reescreve cada consulta como documento virtual expandido pelo grafo.

    Args:
        consultas: DataFrame de consultas e-Fisco (precisa de `item_efisco`).
        catalogo: DataFrame do catálogo CATMAT (precisa de `item_catmat`).
        textos_base: texto de partida por consulta (saída de um pré-processador
            anterior, p. ex. normalização). Padrão: `item_efisco` cru.
        max_termos: teto de termos acrescentados por consulta.
        top_k_vizinhos: itens do catálogo consultados na busca local.
        incluir_comunidade: acrescenta o rótulo da comunidade da consulta.
        usar_ms_graphrag: tenta o índice oficial da Microsoft antes do offline.
        api_key: OPENAI_API_KEY (só no modo `usar_ms_graphrag`).
        kg: KG já construído, para reaproveitar entre execuções.

    Returns:
        (textos_expandidos, estatisticas)
    """
    if textos_base is None:
        textos_base = consultas["item_efisco"].fillna("").astype(str).tolist()
    textos_base = [str(t or "") for t in textos_base]

    if usar_ms_graphrag:
        via_ms = _expandir_via_ms_graphrag(
            consultas, catalogo, textos_base, api_key or os.environ.get("OPENAI_API_KEY", ""),
            max_termos,
        )
        if via_ms is not None:
            n_add = [len(e.split()) - len(b.split())
                     for e, b in zip(via_ms, textos_base)]
            return via_ms, {
                "modo": "ms_graphrag",
                "n_consultas": len(via_ms),
                "media_termos_adicionados": round(sum(n_add) / max(len(n_add), 1), 2),
                "taxa_expandidas": round(sum(1 for n in n_add if n > 0)
                                         / max(len(n_add), 1), 4),
            }

    kg = kg or construir_kg(consultas, catalogo)

    expandidos: list[str] = []
    n_termos_add: list[int] = []
    n_com_comunidade = 0

    for i, texto_base in enumerate(textos_base):
        vizinhos = _busca_local(kg, i, top_k_vizinhos)

        # Entidades dos vizinhos que a consulta ainda não tem
        proprias = kg.ents_consulta.get(i, set())
        candidatos: Counter = Counter()
        for j in vizinhos:
            for ent in kg.ents_catalogo.get(j, ()):
                if ent not in proprias:
                    candidatos[ent] += 1

        termos = _termos_novos(candidatos, texto_base, max_termos)

        if incluir_comunidade:
            rotulo = kg.rotulo_comunidade.get(kg.comunidade.get(i, -1), "")
            if rotulo and rotulo.upper() not in texto_base.upper():
                termos.append(rotulo)
                n_com_comunidade += 1

        n_termos_add.append(len(termos))
        expandidos.append((texto_base + " " + " ".join(termos)).strip())

    stats = {
        "modo": "offline",
        "n_consultas": len(expandidos),
        "n_nos_kg": kg.G.number_of_nodes(),
        "n_arestas_kg": kg.G.number_of_edges(),
        "n_entidades": sum(1 for _, d in kg.G.nodes(data=True)
                           if d.get("tipo") == "entidade"),
        "n_comunidades": len(set(kg.comunidade.values())),
        "n_com_rotulo_comunidade": n_com_comunidade,
        "media_termos_adicionados": round(
            sum(n_termos_add) / max(len(n_termos_add), 1), 2),
        "taxa_expandidas": round(
            sum(1 for n in n_termos_add if n > 0) / max(len(n_termos_add), 1), 4),
    }
    log.info("[GraphRAG/pré] Expansão offline: +%.2f termos/consulta (%.0f%% expandidas).",
             stats["media_termos_adicionados"], 100 * stats["taxa_expandidas"])
    return expandidos, stats
