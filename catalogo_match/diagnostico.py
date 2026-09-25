"""
diagnostico.py — onde o Recall@3 se perde, consulta a consulta.

A grade (`avaliacao_modular.py`) diz QUANTO cada combinação acerta. Este módulo
diz ONDE ela erra: em que posição o item correto ficou, para que tipo de
consulta, e o que mudou entre duas combinações. É a ferramenta que orientou as
decisões de RESULTADOS.md (o diagnóstico das 217 falhas do MMH) e que se
reaplica a qualquer dataset.

Para uma combinação (pré, proc, pós):

  1. posição do correto por consulta, em faixas: 1 | 2-3 | 4-10 | 11-K | >K
     (a última faixa é o que nenhum pós-processador alcança: é TETO, e só a
     recuperação — processador ou fusão — pode mexer nela)
  2. quebra das faixas por origem, grupo e-Fisco e presença de medida
  3. os casos de 4-10 (o alvo do R@3) com correto x intruso lado a lado

Para duas combinações (A -> B): quais consultas entraram e saíram do top-3,
com exemplos — é o que separa "ganhou 2 pp" de "trocou 30 acertos por 32".

Uso:
    python -m catalogo_match.diagnostico --dataset bigdata_profs \\
        --pre nada --proc e5 --pos nada
    python -m catalogo_match.diagnostico --dataset bigdata_profs \\
        --pre nada --proc e5 --pos nada --contra nada,e5_tfidf,medidas_taxonomia

Saídas em resultados/<dataset>/diagnostico/.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from . import avaliacao_modular as am
from . import caracteristicas

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

FAIXAS = ((1, 1, "1"), (2, 3, "2-3"), (4, 10, "4-10"))


def _faixa(pos: int | None, k: int) -> str:
    if pos is None:
        return f">{k}"
    for lo, hi, nome in FAIXAS:
        if lo <= pos <= hi:
            return nome
    return f"11-{k}"


def posicoes(corpus: am.Corpus, idx: np.ndarray) -> list[int | None]:
    """Posição (1 = topo) do primeiro acerto por consulta; None se fora do top-K."""
    cods_cat = corpus.catalogo["codigo_catmat"].tolist()
    saida: list[int | None] = []
    for i, cod_e in enumerate(corpus.consultas["codigo_efisco"]):
        alvos = corpus.gold.get(cod_e, set())
        pos = next((p + 1 for p, j in enumerate(idx[i]) if cods_cat[j] in alvos), None)
        saida.append(pos)
    return saida


def executar_combinacao(corpus: am.Corpus, id_pre: str, id_proc: str, id_pos: str,
                        ctx: dict, top_k: int) -> tuple[np.ndarray, np.ndarray, am.Textos]:
    """Roda uma célula da grade e devolve (scores, idx) já reordenados pelo pós."""
    textos = am._pre_com_cache(am.PRE_PROCESSADORES[id_pre], corpus, ctx)
    scores, idx = am.PROCESSADORES[id_proc].fn(textos, top_k, ctx)
    scores_pos = am.POS_PROCESSADORES[id_pos].fn(corpus, textos, scores, idx, ctx)
    ordem = np.argsort(-scores_pos, axis=1, kind="stable")
    return (np.take_along_axis(scores_pos, ordem, axis=1),
            np.take_along_axis(idx, ordem, axis=1), textos)


def tabela_consultas(corpus: am.Corpus, idx: np.ndarray, pos: list[int | None],
                     top_k: int) -> pd.DataFrame:
    """Uma linha por consulta: posição do correto, faixa, metadados e top-1."""
    perfil = am.perfil_ativo()
    cons, cat = corpus.consultas, corpus.catalogo
    cods_cat = cat["codigo_catmat"].tolist()
    med_q = caracteristicas.indice(cons["item_efisco"].fillna("").astype(str).tolist(), perfil)
    linhas = []
    for i in range(len(cons)):
        cod_e = cons["codigo_efisco"].iloc[i]
        alvos = corpus.gold.get(cod_e, set())
        j_top = int(idx[i, 0])
        j_ok = next((int(j) for j in idx[i] if cods_cat[j] in alvos), None)
        linhas.append({
            "codigo_efisco": cod_e,
            "posicao": pos[i],
            "faixa": _faixa(pos[i], top_k),
            "origem": cons["origem"].iloc[i] if "origem" in cons.columns else "",
            "grupo_efisco": cons["grupo_efisco"].iloc[i] if "grupo_efisco" in cons.columns else "",
            "classe_efisco": cons["classe_efisco"].iloc[i],
            "tem_medida": bool(med_q[i]),
            "medidas_consulta": str(med_q[i]) if med_q[i] else "",
            "item_efisco": cons["item_efisco"].iloc[i],
            "top1_catmat": cat["item_catmat"].iloc[j_top],
            "top1_classe": cat["classe_catmat"].iloc[j_top],
            "correto_catmat": cat["item_catmat"].iloc[j_ok] if j_ok is not None else "",
            "correto_classe": cat["classe_catmat"].iloc[j_ok] if j_ok is not None else "",
            "mesmo_pdm_top1_correto": (
                cat["pdm"].iloc[j_top] == cat["pdm"].iloc[j_ok] if j_ok is not None else None),
        })
    return pd.DataFrame(linhas)


def resumo_faixas(tab: pd.DataFrame, top_k: int, por: str | None = None) -> pd.DataFrame:
    ordem = ["1", "2-3", "4-10", f"11-{top_k}", f">{top_k}"]
    if por is None:
        cont = tab["faixa"].value_counts().reindex(ordem, fill_value=0)
        df = cont.to_frame("n")
        df["%"] = (100 * df["n"] / max(1, len(tab))).round(1)
        return df
    ct = pd.crosstab(tab[por], tab["faixa"]).reindex(columns=ordem, fill_value=0)
    ct["n"] = ct.sum(axis=1)
    ct["R@3"] = ((ct["1"] + ct["2-3"]) / ct["n"]).round(3)
    ct["teto(top-K)"] = (1 - ct[f">{top_k}"] / ct["n"]).round(3)
    return ct.sort_values("n", ascending=False)


def comparar(tab_a: pd.DataFrame, tab_b: pd.DataFrame) -> dict:
    """Quem entrou e quem saiu do top-3 entre A e B."""
    a3 = tab_a["posicao"].apply(lambda p: p is not None and p <= 3)
    b3 = tab_b["posicao"].apply(lambda p: p is not None and p <= 3)
    entraram = tab_b[~a3 & b3]
    sairam = tab_b[a3 & ~b3]
    return {"entraram": entraram, "sairam": sairam,
            "r3_a": round(a3.mean(), 4), "r3_b": round(b3.mean(), 4)}


def _imprimir_exemplos(tab: pd.DataFrame, titulo: str, n: int = 6) -> None:
    print(f"\n--- {titulo} ({len(tab)}) ---")
    for _, r in tab.head(n).iterrows():
        print(f"\n  [{r['faixa']:>5}] {r['origem']} | {r['grupo_efisco'][:50]}")
        print(f"  CONSULTA : {r['item_efisco'][:150]}")
        print(f"  TOP-1    : {r['top1_catmat'][:150]}")
        if r["correto_catmat"]:
            print(f"  CORRETO  : {r['correto_catmat'][:150]}")
        else:
            print(f"  CORRETO  : (fora do top-K)")
        if r["medidas_consulta"]:
            print(f"  medidas  : {r['medidas_consulta']}")


def main() -> None:
    from .config import ativar, listar_datasets

    ap = argparse.ArgumentParser(description="Diagnóstico de falhas de uma combinação da grade.")
    ap.add_argument("--dataset", default="", help=f"disponíveis: {', '.join(listar_datasets())}")
    ap.add_argument("--perfil", default="")
    ap.add_argument("--pre", default="nada")
    ap.add_argument("--proc", default="e5")
    ap.add_argument("--pos", default="nada")
    ap.add_argument("--contra", default="",
                    help="segunda combinação pre,proc,pos para comparar (quem entrou/saiu do top-3)")
    ap.add_argument("--top-k", type=int, default=am.K_CANDIDATOS)
    ap.add_argument("--amostra", type=int)
    ap.add_argument("--exemplos", type=int, default=6)
    ap.add_argument("--peso-medidas", type=float, default=am.PESO_MEDIDAS_PADRAO)
    args = ap.parse_args()

    ativar(args.dataset, perfil=args.perfil)
    corpus = am.carregar_corpus(amostra=args.amostra)
    ctx = {"peso_pos": am.PESO_POS_PADRAO, "peso_medidas": args.peso_medidas,
           "tolerancia_medidas": caracteristicas.TOLERANCIA_PADRAO,
           "k_cross": am.K_CROSS_RERANK, "api_key": os.environ.get("OPENAI_API_KEY", "")}

    destino = am.dir_resultados() / "diagnostico"
    destino.mkdir(parents=True, exist_ok=True)
    combo = f"{args.pre}+{args.proc}+{args.pos}"

    _, idx, _ = executar_combinacao(corpus, args.pre, args.proc, args.pos, ctx, args.top_k)
    pos = posicoes(corpus, idx)
    tab = tabela_consultas(corpus, idx, pos, args.top_k)
    tab.to_csv(destino / f"consultas_{combo}.csv", index=False, encoding="utf-8-sig", sep="|")

    metricas = am.avaliar(corpus, idx)
    print(f"\n=== {combo} | {corpus.dataset} | {len(corpus.consultas)} consultas x "
          f"{len(corpus.catalogo)} itens ===")
    print(f"MRR {metricas['mrr']:.4f}  R@1 {metricas['recall_at_1']:.4f}  "
          f"R@3 {metricas['recall_at_3']:.4f}  R@10 {metricas['recall_at_10']:.4f}")

    print("\n--- posição do correto ---")
    print(resumo_faixas(tab, args.top_k).to_string())
    for por in ("origem", "tem_medida", "grupo_efisco"):
        if por in tab.columns and tab[por].nunique() > 1:
            print(f"\n--- por {por} (top 12) ---")
            print(resumo_faixas(tab, args.top_k, por).head(12).to_string())

    falhas_r3 = tab[tab["faixa"] == "4-10"]
    if len(falhas_r3):
        mesmo_pdm = falhas_r3["mesmo_pdm_top1_correto"].mean()
        print(f"\n--- falhas de R@3 recuperáveis (correto em 4-10): {len(falhas_r3)} "
              f"({100 * len(falhas_r3) / len(tab):.1f}%) | top-1 com o MESMO PDM do correto: "
              f"{100 * mesmo_pdm:.0f}% ---")
    _imprimir_exemplos(falhas_r3, "exemplos: correto em 4-10", args.exemplos)
    _imprimir_exemplos(tab[tab["faixa"] == f">{args.top_k}"],
                       f"exemplos: correto fora do top-{args.top_k} (teto)", args.exemplos)

    if args.contra:
        pre_b, proc_b, pos_b = [p.strip() for p in args.contra.split(",")]
        combo_b = f"{pre_b}+{proc_b}+{pos_b}"
        _, idx_b, _ = executar_combinacao(corpus, pre_b, proc_b, pos_b, ctx, args.top_k)
        tab_b = tabela_consultas(corpus, idx_b, posicoes(corpus, idx_b), args.top_k)
        tab_b.to_csv(destino / f"consultas_{combo_b}.csv", index=False,
                     encoding="utf-8-sig", sep="|")
        cmp = comparar(tab, tab_b)
        print(f"\n=== {combo}  ->  {combo_b} ===")
        print(f"R@3 {cmp['r3_a']:.4f} -> {cmp['r3_b']:.4f}  | entraram no top-3: "
              f"{len(cmp['entraram'])} | saíram: {len(cmp['sairam'])}")
        for chave in ("entraram", "sairam"):
            df = cmp[chave]
            if "grupo_efisco" in df.columns and len(df):
                print(f"  {chave} por grupo: "
                      + ", ".join(f"{g[:28]}={n}" for g, n in
                                  Counter(df["grupo_efisco"]).most_common(6)))
        _imprimir_exemplos(cmp["entraram"], f"entraram no top-3 com {combo_b}", args.exemplos)
        _imprimir_exemplos(cmp["sairam"], f"saíram do top-3 com {combo_b}", args.exemplos)
        pd.concat([cmp["entraram"].assign(movimento="entrou"),
                   cmp["sairam"].assign(movimento="saiu")]).to_csv(
            destino / f"movimentos_{combo}__{combo_b}.csv", index=False,
            encoding="utf-8-sig", sep="|")

    print(f"\nArquivos em {destino}")


if __name__ == "__main__":
    main()
