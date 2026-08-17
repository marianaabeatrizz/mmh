"""
graphrag_mmh.py
===============
Integração com o Microsoft GraphRAG (v3.x) para a fase [6] do pipeline MMH.

Fluxo de indexação (roda uma vez, cached em graphrag_workspace/output/):
  1. Grava um .txt por item (e-Fisco + CATMAT) em graphrag_workspace/input/
  2. Configura GRAPHRAG_API_KEY no .env do workspace
  3. Executa `graphrag index` (extração de entidades, relações, comunidades, resumos)

Fluxo de busca (por consulta na zona cinzenta):
  4. Carrega os artefatos parquet gerados pelo indexador
  5. Executa local_search(query=texto_efisco) via graphrag.api
  6. Retorna contexto textual para o LLM de adjudicação
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

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
