"""
varredura.py — varre parâmetros de UMA combinação da grade.

A grade mede combinações de módulos com os parâmetros padrão. Quando um sinal
novo não rende o esperado, a pergunta seguinte é se o problema é o sinal ou a
escala em que ele entra — e isso se responde variando o peso (e os botões do
sinal) com o resto fixo. Foi assim que o peso 0,05 do sinal de medida foi
escolhido no MMH (ver RESULTADOS.md, "A forma de combinar importa").

Uso:
    python -m catalogo_match.varredura --dataset bigdata_profs \\
        --pre nada --proc e5 --pos taxonomia \\
        --param peso_medidas=0.02,0.05,0.1,0.2 --param taxonomia_peso_grupo=0,0.3

Cada `--param chave=v1,v2,...` é uma dimensão; o produto cartesiano é avaliado
e sai como tabela (e CSV em resultados/<dataset>/varreduras/). Os embeddings E5
são reaproveitados entre pontos; o que depende dos parâmetros (matriz de
taxonomia, KG) é recomputado.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from . import avaliacao_modular as am
from . import caracteristicas

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Chaves do contexto que são caras e NÃO dependem dos parâmetros varridos.
_PRESERVAR = ("e5::", "tfidf::", "cross::", "_corpus", "_kg_graphrag", "_idf_kg",
              "kg::matriz", "_medidas", "_medidas_resumo", "_taxonomia_rotulos")


def _valor(txt: str):
    try:
        return int(txt) if txt.strip().lstrip("-").isdigit() else float(txt)
    except ValueError:
        return txt


def main() -> None:
    from .config import ativar, listar_datasets

    ap = argparse.ArgumentParser(description="Varredura de parâmetros de uma combinação.")
    ap.add_argument("--dataset", default="", help=f"disponíveis: {', '.join(listar_datasets())}")
    ap.add_argument("--perfil", default="")
    ap.add_argument("--pre", default="nada")
    ap.add_argument("--proc", default="e5")
    ap.add_argument("--pos", default="nada",
                    help="um ou mais pós-processadores (csv): cada um é varrido")
    ap.add_argument("--param", action="append", default=[],
                    help="chave=v1,v2,... (repetível). Ex.: peso_medidas=0.02,0.05")
    ap.add_argument("--top-k", type=int, default=am.K_CANDIDATOS)
    ap.add_argument("--amostra", type=int)
    ap.add_argument("--metrica", default="recall_at_3")
    ap.add_argument("--particoes", type=int, default=1,
                    help="além do total, mede a métrica em N partições disjuntas das "
                         "consultas (i mod N). Peso escolhido numa partição e bom nas "
                         "outras é peso que generaliza; escolhido no total, é ajuste "
                         "ao próprio conjunto de teste.")
    args = ap.parse_args()

    ativar(args.dataset, perfil=args.perfil)
    corpus = am.carregar_corpus(amostra=args.amostra)

    dimensoes: dict[str, list] = {}
    for p in args.param:
        chave, _, valores = p.partition("=")
        dimensoes[chave.strip()] = [_valor(v) for v in valores.split(",") if v.strip()]
    pontos = [dict(zip(dimensoes, combo)) for combo in itertools.product(*dimensoes.values())] or [{}]
    ids_pos = [p.strip() for p in args.pos.split(",") if p.strip()]

    base_ctx = {"peso_pos": am.PESO_POS_PADRAO, "peso_medidas": am.PESO_MEDIDAS_PADRAO,
                "tolerancia_medidas": caracteristicas.TOLERANCIA_PADRAO,
                "k_cross": am.K_CROSS_RERANK, "api_key": os.environ.get("OPENAI_API_KEY", ""),
                "_corpus": corpus}
    textos = am._pre_com_cache(am.PRE_PROCESSADORES[args.pre], corpus, base_ctx)

    linhas = []
    preservado: dict = {}
    for ponto in pontos:
        ctx = {**base_ctx, **preservado, **ponto}
        t0 = time.time()
        scores, idx = am.PROCESSADORES[args.proc].fn(textos, args.top_k, ctx)
        for id_pos in ids_pos:
            scores_pos = am.POS_PROCESSADORES[id_pos].fn(corpus, textos, scores, idx, ctx)
            ordem = np.argsort(-scores_pos, axis=1, kind="stable")
            idx_final = np.take_along_axis(idx, ordem, axis=1)
            m = am.avaliar(corpus, idx_final)
            linha = {"pos": id_pos, **ponto, **m, "s": round(time.time() - t0, 1)}
            partes = ""
            for p in range(args.particoes if args.particoes > 1 else 0):
                sel = np.arange(len(corpus.consultas)) % args.particoes == p
                sub = am.Corpus(consultas=corpus.consultas[sel].reset_index(drop=True),
                                catalogo=corpus.catalogo, gold=corpus.gold)
                mp = am.avaliar(sub, idx_final[sel])
                linha[f"{args.metrica}_p{p}"] = mp[args.metrica]
                partes += f"  p{p} {mp[args.metrica]:.4f}"
            linhas.append(linha)
            print(f"  {id_pos:<32} {ponto}  ->  {args.metrica} {m[args.metrica]:.4f}  "
                  f"MRR {m['mrr']:.4f}  R@1 {m['recall_at_1']:.4f}  R@10 {m['recall_at_10']:.4f}"
                  f"{partes}")
        preservado = {k: v for k, v in ctx.items() if any(k.startswith(p) for p in _PRESERVAR)}

    tab = pd.DataFrame(linhas)
    destino = am.dir_resultados() / "varreduras"
    destino.mkdir(parents=True, exist_ok=True)
    sufixo_perfil = f"__perfil-{args.perfil}" if args.perfil else ""
    nome = (f"{args.pre}+{args.proc}+{'_'.join(ids_pos)}__{'_'.join(dimensoes) or 'padrao'}"
            f"{sufixo_perfil}.csv")
    tab.to_csv(destino / nome, index=False, encoding="utf-8-sig")
    print(f"\n=== melhor por {args.metrica} ===")
    print(tab.sort_values(args.metrica, ascending=False).head(10).to_string(index=False))
    print(f"\nCSV: {destino / nome}")


if __name__ == "__main__":
    main()
