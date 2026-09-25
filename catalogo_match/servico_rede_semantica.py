"""
servico_rede_semantica.py
=========================
Serviço HTTP da rede semântica de domínio — §3.8 do documento técnico:

    "Serviço: Python (FastAPI) expondo normalizar(), expandir(), ancorar()."

O grafo léxico é construído uma única vez, na subida do processo (§3.7 separa
explicitamente CONSTRUÇÃO, batch e offline, de EXECUÇÃO, serviço e online).
Cada requisição só percorre o grafo já pronto.

Endpoints
---------
  GET  /saude                 estado do serviço e tamanho do grafo
  POST /normalizar            passo 1 — forma canônica dos tokens
  POST /expandir              passo 2 — spreading activation
  POST /desambiguar           passo 3 — escolha de sentido pelo contexto
  POST /ancorar               passo 4 — Classe/PDM do registro
  POST /documento-virtual     passo 5 — texto expandido pronto para o embedding
  POST /preparar              tudo de uma vez (o que o matcher consome)
  GET  /curadoria             fila priorizada por frequência x incerteza
  GET  /termo/{termo}         inspeção: sentidos e vizinhança de um termo

Subir:
    uvicorn servico_rede_semantica:app --host 127.0.0.1 --port 8000
    (ou: python servico_rede_semantica.py)

Documentação interativa: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import contexto, perfil_ativo
from .preprocessamento import normalizar_texto, tokenizar, remover_stopwords
from .rede_semantica import (
    construir_grafo,
    normalizar_termo,
    expandir_por_ativacao,
    desambiguar,
    sentidos_de,
    ancorar_pdm,
    ancorar_registro,
    fila_curadoria,
    versao_lexico,
    _id_no,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# O dataset servido vem de CM_DATASET (um serviço não tem linha de comando).
# Uma instância serve UM dataset; para servir dois, sobem-se dois processos com
# portas diferentes, que é o que mantém o grafo em memória previsível.
_CTX = contexto()

app = FastAPI(
    title=f"Rede Semântica de Domínio — CATMAT ↔ e-Fisco [{_CTX.dataset.nome}]",
    version=versao_lexico(),
    description=(
        "Serviço de normalização, expansão e ancoragem do léxico de "
        f"{_CTX.perfil.descricao}. Prepara o texto curto e sujo do e-Fisco "
        "antes de qualquer comparação — não decide o casamento."
    ),
)

# Estado do processo: o grafo é carregado uma vez e reusado.
GRAFO = None


@app.on_event("startup")
def _carregar_grafo() -> None:
    global GRAFO
    log.info("Construindo o grafo léxico de '%s' (uma vez, na subida)...",
             _CTX.dataset.nome)
    GRAFO = construir_grafo()
    log.info("Grafo pronto: %d nós | %d arestas",
             GRAFO.number_of_nodes(), GRAFO.number_of_edges())


def _g():
    if GRAFO is None:
        raise HTTPException(503, "Grafo ainda não carregado — tente em instantes.")
    return GRAFO


def _tokens(texto: str) -> list[str]:
    return remover_stopwords(tokenizar(normalizar_texto(texto, manter_maiusculas=True)))


# ---------------------------------------------------------------------------
# Modelos de entrada/saída
# ---------------------------------------------------------------------------

class TextoEntrada(BaseModel):
    texto: str = Field(..., description="Descrição livre do item (e-Fisco ou CATMAT).",
                       json_schema_extra={"example": "AGULHA ESPINHAL 22G 3,5POL QUICKLE ESTER DESC"})


class ExpansaoEntrada(BaseModel):
    texto: str = Field(..., description="Texto de onde partem as sementes de ativação.")
    limiar: float = Field(0.15, ge=0.0, le=1.0, description="Ativação mínima para propagar.")
    max_saltos: int = Field(2, ge=1, le=4, description="Profundidade da propagação.")
    top_n: int = Field(10, ge=1, le=50, description="Quantos termos ativados retornar.")


class DesambiguacaoEntrada(BaseModel):
    termo: str = Field(..., description="Termo possivelmente ambíguo.",
                       json_schema_extra={"example": "COMP"})
    contexto: list[str] = Field(default_factory=list,
                                description="Demais tokens do registro.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/saude", tags=["serviço"])
def saude() -> dict:
    """Estado do serviço e dimensão do léxico carregado."""
    if GRAFO is None:
        return {"status": "carregando", "versao_lexico": versao_lexico()}
    from collections import Counter
    tipos = Counter(d.get("tipo") for _, d in GRAFO.nodes(data=True))
    relacoes = Counter(d.get("relacao") for _, _, d in GRAFO.edges(data=True))
    return {
        "status": "pronto",
        "versao_lexico": versao_lexico(),
        "n_nos": GRAFO.number_of_nodes(),
        "n_arestas": GRAFO.number_of_edges(),
        "nos_por_tipo": dict(tipos),
        "arestas_por_relacao": dict(relacoes),
    }


@app.post("/normalizar", tags=["§3.7 passo 1"])
def endpoint_normalizar(entrada: TextoEntrada) -> dict:
    """
    Passo 1 — resolve abreviação / variante ortográfica para a forma canônica.
    A desambiguação usa os demais tokens do próprio registro como contexto.
    """
    G = _g()
    toks = _tokens(entrada.texto)
    norm = [normalizar_termo(G, t, contexto=toks) for t in toks]
    return {
        "texto_original": entrada.texto,
        "tokens": toks,
        "tokens_normalizados": norm,
        "texto_normalizado": " ".join(norm),
        "alteracoes": [
            {"de": a, "para": b} for a, b in zip(toks, norm) if a != b
        ],
    }


@app.post("/expandir", tags=["§3.7 passo 2"])
def endpoint_expandir(entrada: ExpansaoEntrada) -> dict:
    """
    Passo 2 — spreading activation restrito a sinônimo/hiperônimo/relacionadoA,
    com decaimento por salto e corte no limiar (evita explosão de contexto).
    """
    G = _g()
    toks = _tokens(entrada.texto)
    norm = [normalizar_termo(G, t, contexto=toks) for t in toks]
    seeds = [t for t in norm if len(t) > 3]
    ativacoes = expandir_por_ativacao(
        G, seeds, limiar=entrada.limiar, max_saltos=entrada.max_saltos
    )
    ordenado = sorted(ativacoes.items(), key=lambda kv: -kv[1])[: entrada.top_n]
    return {
        "sementes": seeds,
        "ativados": [{"termo": t, "ativacao": round(v, 4)} for t, v in ordenado],
    }


@app.post("/desambiguar", tags=["§3.7 passo 3"])
def endpoint_desambiguar(entrada: DesambiguacaoEntrada) -> dict:
    """
    Passo 3 — quando um termo liga a vários Conceitos, escolhe o de maior
    ativação acumulada dada a vizinhança do registro.
    """
    G = _g()
    candidatos = sentidos_de(G, entrada.termo)
    escolhido = desambiguar(G, entrada.termo, contexto=entrada.contexto or None)
    return {
        "termo": entrada.termo,
        "ambiguo": len(candidatos) > 1,
        "sentidos_possiveis": [
            {"conceito": G.nodes[nid]["label"], "peso_aresta": round(p, 3)}
            for nid, p in candidatos
        ],
        "sentido_escolhido": escolhido,
        "contexto_usado": entrada.contexto,
    }


@app.post("/ancorar", tags=["§3.7 passo 4"])
def endpoint_ancorar(entrada: TextoEntrada) -> dict:
    """
    Passo 4 — Classe/PDM do registro: chave de blocking e porta de entrada da
    ontologia. Decide por consenso dos tokens, não pelo primeiro que casar.
    """
    G = _g()
    toks = _tokens(entrada.texto)
    norm = [normalizar_termo(G, t, contexto=toks) for t in toks]
    return {
        "texto": entrada.texto,
        "pdm": ancorar_registro(G, norm) or None,
        "ancoras_por_token": [
            {"token": t, "pdm": ancorar_pdm(G, t)}
            for t in norm if ancorar_pdm(G, t)
        ],
    }


@app.post("/documento-virtual", tags=["§3.7 passo 5"])
def endpoint_documento_virtual(entrada: TextoEntrada) -> dict:
    """
    Passo 5 — forma canônica + termos ativados relevantes. É este texto que vai
    para o embedding, e é ele que resolve o abismo curto x longo no vetorial.
    """
    G = _g()
    toks = _tokens(entrada.texto)
    norm = [normalizar_termo(G, t, contexto=toks) for t in toks]
    seeds = [t for t in norm if len(t) > 3]
    ativacoes = expandir_por_ativacao(G, seeds, limiar=0.15, max_saltos=2)
    expandidos = [
        t for t, s in sorted(ativacoes.items(), key=lambda kv: -kv[1])
        if s > 0.3 and t not in norm
    ][:5]
    return {
        "documento_virtual": (" ".join(norm) + " " + " ".join(expandidos)).strip(),
        "termos_acrescentados": expandidos,
    }


@app.post("/preparar", tags=["pipeline"])
def endpoint_preparar(entrada: TextoEntrada) -> dict:
    """
    Os cinco passos numa chamada — é o que o matcher consome por item.
    Devolve também os tokens OOV, o sinal de deriva de domínio da §3.10.
    """
    G = _g()
    toks = _tokens(entrada.texto)
    norm = [normalizar_termo(G, t, contexto=toks) for t in toks]
    seeds = [t for t in norm if len(t) > 3]
    ativacoes = expandir_por_ativacao(G, seeds, limiar=0.15, max_saltos=2)
    expandidos = [
        t for t, s in sorted(ativacoes.items(), key=lambda kv: -kv[1])
        if s > 0.3 and t not in norm
    ][:5]
    oov = [o for o, n in zip(toks, norm)
           if o == n and _id_no("Conceito", o) not in G]
    return {
        "texto_original": entrada.texto,
        "texto_normalizado": " ".join(norm),
        "documento_virtual": (" ".join(norm) + " " + " ".join(expandidos)).strip(),
        "termos_expandidos": expandidos,
        "pdm": ancorar_registro(G, norm) or None,
        "tokens_oov": oov,
        "taxa_oov": round(len(oov) / max(len(toks), 1), 4),
    }


@app.get("/curadoria", tags=["§3.6-e"])
def endpoint_curadoria(limite: int = 50) -> dict:
    """Fila de curadoria priorizada por frequência x incerteza."""
    return {"itens": fila_curadoria(_g(), limite=limite)}


@app.get("/termo/{termo}", tags=["inspeção"])
def endpoint_termo(termo: str) -> dict:
    """Inspeciona um termo: sentidos, forma canônica e vizinhança no grafo."""
    G = _g()
    nid_t = _id_no("Termo", normalizar_texto(termo, manter_maiusculas=True))
    nid_c = _id_no("Conceito", normalizar_texto(termo, manter_maiusculas=True))

    vizinhos = []
    for nid in (nid_t, nid_c):
        if nid in G:
            for _, dest, d in G.out_edges(nid, data=True):
                vizinhos.append({
                    "relacao": d.get("relacao"),
                    "destino": G.nodes[dest].get("label"),
                    "peso": round(float(d.get("peso", 0)), 3),
                    "fonte": d.get("fonte"),
                    "validadoPor": d.get("validadoPor", ""),
                })

    return {
        "termo": termo,
        "existe_como_termo": nid_t in G,
        "existe_como_conceito": nid_c in G,
        "forma_canonica": normalizar_termo(G, termo),
        "sentidos": [
            {"conceito": G.nodes[n]["label"], "peso": round(p, 3)}
            for n, p in sentidos_de(G, termo)
        ],
        "vizinhanca": sorted(vizinhos, key=lambda v: -v["peso"])[:25],
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Serviço HTTP do léxico de domínio.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dataset", default="",
                    help="dataset a servir (ou use a variável CM_DATASET)")
    _args = ap.parse_args()

    if _args.dataset:
        # Reativa antes de subir o servidor; o grafo só é construído no startup.
        from .config import ativar
        _CTX = ativar(_args.dataset)

    uvicorn.run(app, host=_args.host, port=_args.port)
