"""
taxonomia.py — as duas taxonomias como ontologia leve de alinhamento.

Os dois catálogos já são hierarquias: o e-Fisco organiza os itens em
grupo > classe, e o CATMAT em grupo > classe > PDM. São duas ontologias de
classificação de materiais, com rótulos diferentes para o mesmo recorte do
mundo ("MATERIAIS DE USO TECNICO HOSPITALAR" de um lado, "INSTRUMENTOS,
EQUIPAMENTOS E SUPRIMENTOS MÉDICOS E CIRÚRGICOS" do outro). Alinhar as duas é
ontology matching no sentido clássico — e é conhecimento que o texto do item
não carrega, porque a classe é metadado, não descrição.

Este módulo produz um sinal de COMPATIBILIDADE entre a classe da consulta e a
classe do candidato, em [0, 1], por dois caminhos independentes:

  histórico  — P(classe_catmat | classe_efisco) estimada nos vínculos já
               conhecidos, em LEAVE-ONE-OUT: a consulta que está sendo
               avaliada nunca contribui para a própria estimativa. É a mesma
               fonte da mineração de jargão da rede semântica (§3.6-b),
               aplicada ao nível da classe e não do token; a diferença é que
               aqui o vazamento é fechado explicitamente.
  rótulo     — similaridade E5 entre os rótulos das classes dos dois lados,
               sem nenhum vínculo conhecido. É o alinhamento que se tem quando
               o histórico não existe (catálogo novo), e serve de controle: o
               quanto do ganho do histórico é a informação dos vínculos e o
               quanto é só o texto do rótulo.

Ambos caem para o nível de GRUPO quando a classe é desconhecida (backoff), e
ambos se calam (0,0) quando não há nada a dizer — sinal esparso, como o de
medida (`caracteristicas.py`), pelo mesmo motivo: sinal denso e médio
reordena o que já estava certo.

Uso na grade (`avaliacao_modular.py`): pós-processadores `taxonomia` e
`taxonomia_rotulo`, sozinhos e somados a `medidas` e `graphrag`.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import contexto

log = logging.getLogger(__name__)

# Peso do nível de grupo quando a classe é conhecida dos dois lados: a classe
# já diz quase tudo; o grupo só suaviza. Quando a classe da consulta nunca foi
# vista, o grupo é tudo que há e entra inteiro.
PESO_GRUPO_COM_CLASSE = 0.3

# Mínimo de vínculos observados para a classe da consulta ter voto. Com um único
# vínculo, P(cc|ce) é 0 ou 1 e vira ruído; com três já separa o comum do
# acidental.
MIN_OBS_CLASSE = 2


def _col(df: pd.DataFrame, nome: str) -> list[str]:
    """Coluna como lista de strings normalizadas, ou lista vazia se não existe."""
    if nome not in df.columns:
        return []
    return [str(v or "").strip().upper() for v in df[nome].fillna("")]


# ---------------------------------------------------------------------------
# 1. Alinhamento histórico (leave-one-out)
# ---------------------------------------------------------------------------

@dataclass
class AlinhamentoHistorico:
    """
    Contagens dos vínculos (classe_efisco, classe_catmat) e (grupo_efisco,
    grupo_catmat), com o que cada consulta contribuiu — para poder subtraí-lo.
    """

    classe_e: list[str]
    classe_c: list[str]
    grupo_e: list[str]
    grupo_c: list[str]
    pares_classe: Counter = field(default_factory=Counter)
    total_classe: Counter = field(default_factory=Counter)
    pares_grupo: Counter = field(default_factory=Counter)
    total_grupo: Counter = field(default_factory=Counter)
    # Por consulta: as classes/grupos CATMAT dos seus vínculos (para o LOO).
    contrib: dict[int, list[tuple[str, str]]] = field(default_factory=dict)
    # Botões (ver constantes no topo): ajustáveis por execução para a varredura.
    peso_grupo: float = PESO_GRUPO_COM_CLASSE
    min_obs: int = MIN_OBS_CLASSE

    def _prob(self, pares: Counter, total: Counter, chave_e: str, chave_c: str,
              descontar: list[str], minimo: int) -> Optional[float]:
        n_tot = total[chave_e] - len(descontar)
        if n_tot < minimo or not chave_e or not chave_c:
            return None
        n_par = pares[(chave_e, chave_c)] - sum(1 for c in descontar if c == chave_c)
        return max(0.0, n_par) / n_tot

    def compatibilidade(self, i: int, j: int, minimo: Optional[int] = None) -> float:
        """P(classe_c[j] | classe_e[i]) em LOO, com backoff para grupo."""
        minimo = self.min_obs if minimo is None else minimo
        ce, cc = self.classe_e[i], self.classe_c[j]
        ge, gc = self.grupo_e[i], self.grupo_c[j]
        proprios = self.contrib.get(i, [])
        p_classe = self._prob(self.pares_classe, self.total_classe, ce, cc,
                              [c for c, _ in proprios], minimo)
        p_grupo = self._prob(self.pares_grupo, self.total_grupo, ge, gc,
                             [g for _, g in proprios], minimo)
        if p_classe is None and p_grupo is None:
            return 0.0
        if p_classe is None:
            return p_grupo
        if p_grupo is None or self.peso_grupo <= 0:
            return p_classe
        return (1 - self.peso_grupo) * p_classe + self.peso_grupo * p_grupo

    def resumo(self) -> dict:
        return {
            "n_classes_efisco": len(self.total_classe),
            "n_classes_catmat": len({c for _, c in self.pares_classe}),
            "n_pares_classe_distintos": len(self.pares_classe),
            "n_grupos_efisco": len(self.total_grupo),
            "n_pares_grupo_distintos": len(self.pares_grupo),
            "consultas_com_classe": sum(1 for c in self.classe_e if c),
            "catalogo_com_classe": sum(1 for c in self.classe_c if c),
        }


def alinhamento_historico(consultas: pd.DataFrame, catalogo: pd.DataFrame,
                          gold: dict[str, set]) -> AlinhamentoHistorico:
    """
    Conta os vínculos conhecidos por (classe, classe) e (grupo, grupo).

    `gold` mapeia codigo_efisco -> {codigo_catmat}. Só entram vínculos cujo item
    CATMAT está no catálogo avaliado — o que está fora não é candidato e não
    ensina nada sobre o ranking.
    """
    classe_e, classe_c = _col(consultas, "classe_efisco"), _col(catalogo, "classe_catmat")
    grupo_e, grupo_c = _col(consultas, "grupo_efisco"), _col(catalogo, "grupo_catmat")
    n_q, n_c = len(consultas), len(catalogo)
    classe_e = classe_e or [""] * n_q
    classe_c = classe_c or [""] * n_c
    grupo_e = grupo_e or [""] * n_q
    grupo_c = grupo_c or [""] * n_c

    idx_catmat = {str(c).strip(): j for j, c in enumerate(catalogo["codigo_catmat"])}
    al = AlinhamentoHistorico(classe_e, classe_c, grupo_e, grupo_c)

    for i, cod_e in enumerate(consultas["codigo_efisco"]):
        contribs: list[tuple[str, str]] = []
        for cod_c in gold.get(str(cod_e).strip(), ()):
            j = idx_catmat.get(str(cod_c).strip())
            if j is None:
                continue
            ce, cc, ge, gc = classe_e[i], classe_c[j], grupo_e[i], grupo_c[j]
            if ce and cc:
                al.pares_classe[(ce, cc)] += 1
                al.total_classe[ce] += 1
            if ge and gc:
                al.pares_grupo[(ge, gc)] += 1
                al.total_grupo[ge] += 1
            contribs.append((cc, gc))
        al.contrib[i] = contribs

    r = al.resumo()
    log.info("[taxonomia] histórico: %d classes e-Fisco x %d classes CATMAT, "
             "%d pares de classe distintos, %d de grupo.",
             r["n_classes_efisco"], r["n_classes_catmat"],
             r["n_pares_classe_distintos"], r["n_pares_grupo_distintos"])
    return al


# ---------------------------------------------------------------------------
# 2. Alinhamento por rótulo (sem histórico)
# ---------------------------------------------------------------------------

@dataclass
class AlinhamentoRotulos:
    """Similaridade normalizada entre rótulos de classe (e de grupo) dos dois lados."""

    classe_e: list[str]
    classe_c: list[str]
    grupo_e: list[str]
    grupo_c: list[str]
    sim_classe: dict[tuple[str, str], float]
    sim_grupo: dict[tuple[str, str], float]

    def compatibilidade(self, i: int, j: int) -> float:
        ce, cc, ge, gc = self.classe_e[i], self.classe_c[j], self.grupo_e[i], self.grupo_c[j]
        s_c = self.sim_classe.get((ce, cc)) if ce and cc else None
        s_g = self.sim_grupo.get((ge, gc)) if ge and gc else None
        if s_c is None and s_g is None:
            return 0.0
        if s_c is None:
            return s_g
        if s_g is None:
            return s_c
        return (1 - PESO_GRUPO_COM_CLASSE) * s_c + PESO_GRUPO_COM_CLASSE * s_g


def _similaridade_rotulos(rotulos_e: list[str], rotulos_c: list[str],
                          modelo_nome: str) -> dict[tuple[str, str], float]:
    """
    Cosseno E5 entre cada rótulo e-Fisco e cada rótulo CATMAT, reescalado por
    linha para [0, 1]: o cosseno bruto do E5 fica em ~0,75-0,95 para qualquer
    par, e o que informa é a POSIÇÃO relativa do candidato entre as classes.
    """
    if not rotulos_e or not rotulos_c:
        return {}
    ds = contexto().dataset
    ds.dir_cache_embeddings.mkdir(parents=True, exist_ok=True)
    assinatura = hashlib.sha1(
        (modelo_nome + "|" + "\n".join(rotulos_e) + "|||" + "\n".join(rotulos_c)).encode("utf-8")
    ).hexdigest()[:16]
    arquivo = ds.dir_cache_embeddings / f"taxonomia_rotulos_{assinatura}.npz"
    if arquivo.exists():
        dados = np.load(arquivo)
        emb_e, emb_c = dados["e"], dados["c"]
    else:
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer(modelo_nome)
        emb_e = modelo.encode([f"query: {r}" for r in rotulos_e], convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        emb_c = modelo.encode([f"passage: {r}" for r in rotulos_c], convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        np.savez_compressed(arquivo, e=emb_e, c=emb_c)
    sims = emb_e @ emb_c.T
    lo = sims.min(axis=1, keepdims=True)
    hi = sims.max(axis=1, keepdims=True)
    norm = (sims - lo) / np.maximum(hi - lo, 1e-9)
    return {(re_, rc): float(norm[a, b])
            for a, re_ in enumerate(rotulos_e) for b, rc in enumerate(rotulos_c)}


def alinhamento_rotulos(consultas: pd.DataFrame, catalogo: pd.DataFrame,
                        modelo_nome: str = "intfloat/multilingual-e5-base") -> AlinhamentoRotulos:
    classe_e, classe_c = _col(consultas, "classe_efisco"), _col(catalogo, "classe_catmat")
    grupo_e, grupo_c = _col(consultas, "grupo_efisco"), _col(catalogo, "grupo_catmat")
    n_q, n_c = len(consultas), len(catalogo)
    classe_e = classe_e or [""] * n_q
    classe_c = classe_c or [""] * n_c
    grupo_e = grupo_e or [""] * n_q
    grupo_c = grupo_c or [""] * n_c

    rot_ce = sorted({c for c in classe_e if c})
    rot_cc = sorted({c for c in classe_c if c})
    rot_ge = sorted({g for g in grupo_e if g})
    rot_gc = sorted({g for g in grupo_c if g})
    sim_classe = _similaridade_rotulos(rot_ce, rot_cc, modelo_nome)
    sim_grupo = _similaridade_rotulos(rot_ge, rot_gc, modelo_nome)
    log.info("[taxonomia] rótulos: %d x %d classes, %d x %d grupos alinhados por E5.",
             len(rot_ce), len(rot_cc), len(rot_ge), len(rot_gc))
    return AlinhamentoRotulos(classe_e, classe_c, grupo_e, grupo_c, sim_classe, sim_grupo)


# ---------------------------------------------------------------------------
# 3. Matrizes: consulta x catálogo inteiro, e consulta x top-K
# ---------------------------------------------------------------------------

def matriz_compatibilidade(alinhamento) -> np.ndarray:
    """
    Compatibilidade de CADA consulta com CADA item do catálogo (n x m).

    É o que permite usar a taxonomia na RECUPERAÇÃO, e não só na reordenação
    do top-K: um item da classe certa que o bi-encoder deixou na 40ª posição só
    entra no top-3 se o sinal for aplicado antes do corte. Vetorizado por
    classe: o candidato j só depende da sua classe/grupo, então a linha i é um
    lookup em dois vetores pequenos (um por classe CATMAT, um por grupo).
    """
    classe_c, grupo_c = alinhamento.classe_c, alinhamento.grupo_c
    n, m = len(alinhamento.classe_e), len(classe_c)
    classes = sorted(set(classe_c))
    grupos = sorted(set(grupo_c))
    idx_classe = np.array([classes.index(c) for c in classe_c])
    idx_grupo = np.array([grupos.index(g) for g in grupo_c])

    # Um representante por (classe, grupo) do catálogo basta para calcular a
    # compatibilidade: todos os itens com o mesmo par têm o mesmo valor.
    representantes: dict[tuple[int, int], int] = {}
    for j in range(m):
        representantes.setdefault((int(idx_classe[j]), int(idx_grupo[j])), j)

    matriz = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        valores = {par: alinhamento.compatibilidade(i, j) for par, j in representantes.items()}
        if not any(valores.values()):
            continue
        for par, v in valores.items():
            if v:
                sel = (idx_classe == par[0]) & (idx_grupo == par[1])
                matriz[i, sel] = v
    return matriz


def bonus_top_k(alinhamento, idx: np.ndarray, matriz: Optional[np.ndarray] = None) -> np.ndarray:
    """Compatibilidade consulta x candidato para cada célula do top-K."""
    if matriz is not None:
        return np.take_along_axis(matriz, idx, axis=1)
    bonus = np.zeros(idx.shape, dtype=np.float32)
    for i in range(idx.shape[0]):
        for k, j in enumerate(idx[i]):
            bonus[i, k] = alinhamento.compatibilidade(i, int(j))
    return bonus


def diagnostico_cobertura(al: AlinhamentoHistorico, gold: dict[str, set],
                          consultas: pd.DataFrame, catalogo: pd.DataFrame) -> dict:
    """
    Quantas consultas têm o par correto com compatibilidade > 0 em LOO — o
    teto do que o sinal pode ajudar — e quantas têm classe nunca vista.
    """
    idx_catmat = {str(c).strip(): j for j, c in enumerate(catalogo["codigo_catmat"])}
    n = n_pos = n_sem_classe = 0
    for i, cod_e in enumerate(consultas["codigo_efisco"]):
        alvos = [idx_catmat.get(str(c).strip()) for c in gold.get(str(cod_e).strip(), ())]
        alvos = [j for j in alvos if j is not None]
        if not alvos:
            continue
        n += 1
        if al.total_classe[al.classe_e[i]] - len(al.contrib.get(i, [])) < MIN_OBS_CLASSE:
            n_sem_classe += 1
        if any(al.compatibilidade(i, j) > 0 for j in alvos):
            n_pos += 1
    return {"consultas": n, "correto_com_compatibilidade": n_pos,
            "classe_sem_historico_loo": n_sem_classe,
            "fracao_correto_compativel": round(n_pos / max(n, 1), 4)}
