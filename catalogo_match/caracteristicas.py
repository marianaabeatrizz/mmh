"""
caracteristicas.py — separacao de caracteristicas para o casamento de itens.

O texto do e-Fisco descreve o item numa frase corrida ("OVO DE GALINHA, BRANCO,
MEDIO, ISENTO DE SUJIDADE"), enquanto o rotulo do CATMAT ja vem separado por
caracteristica ("COR: BRANCA , TAMANHO: MEDIO"). Este modulo extrai do texto
livre a parte que da para comparar com seguranca: **medida tipada**, ou seja
valor numerico + unidade, convertido para a unidade base da sua categoria.

Por que so a medida
-------------------
Tres variantes de comparacao de caracteristica CATEGORICA foram medidas nas 217
falhas de R@3 do MMH (correto no top-10, fora do top-3) e todas perderam para o
acaso:

    cobertura de todos os atributos do candidato     26% x 28% (correto x intruso)
    so as chaves em que candidatos divergem          23% x 27%
    idem, com peso por variabilidade na familia      17% x 30%
    cobertura ancorada na consulta, com IDF           18% x 25%

O motivo e sempre o mesmo: a distribuicao de caracteristicas e dominada por
vocabulario de familia -- DESCARTAVEL, ESTERIL, EMBALAGEM INDIVIDUAL --, que
todo candidato do bloco satisfaz. Pesar por IDF nos candidatos nao resolve,
porque os intrusos que chegaram ao top-3 sao lexicalmente parecidos com a
consulta; e por isso que chegaram. O conjunto de candidatos e adversarial por
construcao.

A medida tipada, ao contrario, separa bem: 46 acertos contra 9 erros nas mesmas
217 falhas (razao 5:1). Ela e o que o bi-encoder e estruturalmente cego para --
"3 1/2 polegadas" e "90 MM" sao o mesmo comprimento e nenhum modelo de
similaridade textual sabe disso.

Consequencia de desenho: o sinal e ESPARSO E CONFIANTE. Ele opina quando ha
categoria de medida declarada dos dois lados e se cala no resto, em vez de
produzir um numero para todo par. Sinal denso e medio foi o que falhou.

Dirigido pelo perfil
--------------------
Nada aqui e especifico de material hospitalar. As unidades, suas grafias de
superficie e os fatores de conversao vem de `unidades:` no perfil de dominio;
uma unidade sem `aliases` declarados nao e extraida de texto livre. Os campos
opcionais que o extrator le:

    aliases        grafias no texto livre ("MM"; '"', "POL"; "G", "GA", "GAUGE")
    prefixo        a unidade tambem aparece ANTES do numero ("G16", e nao "16G")
    fracao         aceita fracao mista ("3 1/2\\"" -> 3,5)
    faixa          [min, max] plausivel; fora disso o casamento e descartado
    inteiro        o valor precisa ser inteiro (calibre 16, nao 16,3)

A comparacao e feita por `base` (a unidade canonica da categoria), e nao por
`categoria`: gauge e french sao ambos calibre, mas 7 FR nao e 7 G.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Iterable, Optional

from .config import PerfilDominio, perfil_ativo
from .preprocessamento import normalizar_texto

log = logging.getLogger(__name__)

# Tolerancia relativa no casamento de valores. O catalogo escreve "CERCA DE 3\"
# - 80 MM" (3 polegadas sao 76,2 mm, rotuladas como 80), entao comparar
# comprimento exige folga. 8% cobre esse arredondamento e ainda separa 50 de 90.
TOLERANCIA_PADRAO = 0.08

_NUM = r"\d+(?:[.,]\d+)?"


def _f(txt: str) -> float:
    return float(str(txt).replace(",", "."))


# ---------------------------------------------------------------------------
# Padroes de extracao, montados do perfil
# ---------------------------------------------------------------------------

# Prioridade de leitura. A fracao vem antes do sufixo porque `3 1/4"` tem de ser
# lido como 3,25 polegadas e nao, tambem, como 4 polegadas: o trecho casado e
# consumido, e o padrao seguinte nao o reencontra.
_PRIORIDADE = {"fracao": 0, "fracao_simples": 1, "prefixo": 2, "sufixo": 3}


class _Padrao:
    """Uma forma de superficie de uma unidade, com o que fazer ao casar."""

    __slots__ = ("regex", "base", "fator", "faixa", "inteiro", "tipo", "ordem", "alias")

    def __init__(self, regex, base, fator, faixa, inteiro, tipo, ordem, alias=""):
        self.regex = regex
        self.base = base
        self.fator = fator
        self.faixa = faixa
        self.inteiro = inteiro
        self.tipo = tipo          # "sufixo" | "prefixo" | "fracao" | "fracao_simples"
        self.ordem = ordem        # (prioridade do tipo, -tamanho do alias)
        self.alias = alias        # grafia de superficie que este padrao le

    def _valor(self, m) -> Optional[float]:
        if self.tipo == "fracao":
            v = _f(m.group(1)) + _f(m.group(2)) / _f(m.group(3))
        elif self.tipo == "fracao_simples":
            v = _f(m.group(1)) / _f(m.group(2))
        else:
            v = _f(m.group(1))
        v *= self.fator
        if self.inteiro and v != int(v):
            return None
        if self.faixa and not (self.faixa[0] <= v <= self.faixa[1]):
            return None
        return round(v, 1)


_CACHE_PADROES: dict[int, tuple] = {}


def padroes(perfil: Optional[PerfilDominio] = None) -> tuple:
    """Padroes de extracao do perfil, memorizados por tabela de unidades."""
    perfil = perfil or perfil_ativo()
    chave = id(perfil.unidades)
    if chave in _CACHE_PADROES:
        return _CACHE_PADROES[chave]

    montados: list[_Padrao] = []
    for nome, spec in (perfil.unidades or {}).items():
        base = spec.get("base", nome)
        fator = float(spec.get("fator_para_base", 1.0))
        faixa = spec.get("faixa")
        faixa = (float(faixa[0]), float(faixa[1])) if faixa else None
        inteiro = bool(spec.get("inteiro"))
        for alias in (spec.get("aliases") or ()):
            alias = str(alias)
            a = re.escape(alias)
            # `\b` nao serve ao lado de aspas: as fronteiras sao explicitas.
            fim = r"(?![A-Z0-9])" if alias[-1].isalnum() else ""
            ini_alias = r"(?<![A-Z0-9])" if alias[0].isalnum() else ""

            def novo(rx, tipo):
                montados.append(_Padrao(
                    re.compile(rx, re.IGNORECASE), base, fator, faixa, inteiro,
                    tipo, (_PRIORIDADE[tipo], -len(alias)), alias))

            novo(rf"(?<![A-Z]){{0}}({_NUM})\s*{a}{fim}".format(""), "sufixo")
            if spec.get("prefixo"):
                # Fronteira a esquerda obrigatoria: sem ela o alias "GA" casa
                # dentro de "SERINGA 10 ML" e inventa um calibre 10.
                novo(rf"{ini_alias}{a}\s*({_NUM})(?![A-Z0-9])", "prefixo")
            if spec.get("fracao"):
                novo(rf"(\d+)\s+(\d+)\s*/\s*(\d+)\s*{a}{fim}", "fracao")
                novo(rf"(\d+)\s*/\s*(\d+)\s*{a}{fim}", "fracao_simples")

    # Desempate entre padroes do MESMO tipo e alias: o mais restrito (com
    # `faixa`/`inteiro`) le antes do irrestrito. E o que permite "G" ser gauge
    # E grama no mesmo perfil: "16 G" cai na faixa 5-34 e e lido como calibre;
    # "500 G" e rejeitado pelo calibre e sobra para a massa. Sem isto a ordem
    # de declaracao no YAML decidia, e a massa (declarada no perfil base, antes)
    # engolia todos os calibres.
    saida = tuple(sorted(
        montados,
        key=lambda p: (*p.ordem, 0 if (p.faixa or p.inteiro) else 1),
    ))
    _CACHE_PADROES[chave] = saida
    if not saida:
        log.info("Perfil '%s' nao declara `aliases` em nenhuma unidade: a "
                 "separacao por medida fica inativa.", perfil.nome)
    return saida


# ---------------------------------------------------------------------------
# Extracao e comparacao
# ---------------------------------------------------------------------------

_RE_ESPACOS = re.compile(r"\s{2,}")


def _normalizar(texto: str) -> str:
    """
    Caixa alta e sem acento, PRESERVANDO a aspas da polegada e a barra da
    fracao. `normalizar_texto` do pipeline remove aspas, e com elas some a
    medida `3 1/2"` -- que e uma das que mais discriminam no corpus.
    """
    import unicodedata

    t = unicodedata.normalize("NFD", str(texto or ""))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").upper()
    # O "%" fica: concentracao ("SOLUCAO 0,9%") e medida tipada em farmacia e
    # limpeza, e um perfil pode declara-lo como alias de unidade.
    t = re.sub(r"[^\w\s,\.\-/\"%]", " ", t)
    return _RE_ESPACOS.sub(" ", t).strip()


_CACHE_DIMENSOES: dict[int, Optional["re.Pattern"]] = {}


def _re_dimensoes(pads: tuple) -> Optional["re.Pattern"]:
    """
    Cadeia de dimensoes com unidade so no fim: "210X86X162 CM", "5 X 10CM",
    "1,20 X 0,80 M". Os aliases vem dos padroes de sufixo do perfil.
    """
    chave = id(pads)
    if chave in _CACHE_DIMENSOES:
        return _CACHE_DIMENSOES[chave]
    # So aliases alfanumericos: a polegada como aspas ('3 1/2" X 5"') ja e lida
    # pelo padrao de fracao, e "%" nao descreve dimensao.
    alnum = sorted({p.alias.upper() for p in pads
                    if p.tipo == "sufixo" and p.alias.isalnum()},
                   key=len, reverse=True)
    if not alnum:
        _CACHE_DIMENSOES[chave] = None
        return None
    alternativas = "|".join(re.escape(a) for a in alnum)
    rx = re.compile(
        rf"(?<![A-Z0-9])({_NUM})((?:\s*X\s*{_NUM}){{1,2}})\s*({alternativas})(?![A-Z0-9])",
        re.IGNORECASE,
    )
    _CACHE_DIMENSOES[chave] = rx
    return rx


def _expandir_dimensoes(t: str, pads: tuple) -> str:
    """
    "210,00X86,00X162,00CM" -> "210,00 CM X 86,00 CM X 162,00 CM".

    Sem isto, a unidade so era lida no ultimo numero da cadeia — e nem nele
    direito: com o "X" colado, a fronteira `(?<![A-Z])` cortava "162" em "62" e
    a cama de 162 cm virava uma de 620 mm. Mobiliario, embalagem e tecido
    descrevem quase toda medida assim; material hospitalar, quase nunca.
    """
    rx = _re_dimensoes(pads)
    if rx is None:
        return t

    def _reescrever(m) -> str:
        unidade = m.group(3)
        numeros = [m.group(1)] + re.findall(_NUM, m.group(2))
        return " X ".join(f"{n} {unidade}" for n in numeros)

    return rx.sub(_reescrever, t)


def extrair(texto: str, perfil: Optional[PerfilDominio] = None) -> dict[str, set]:
    """
    Texto livre -> {unidade_base: {valores convertidos}}.

    Recebe o texto CRU de proposito (ver `_normalizar`). Cada trecho casado e
    consumido, na ordem de prioridade dos padroes, para que a mesma sequencia de
    digitos nao seja lida duas vezes com leituras incompativeis.
    """
    pads = padroes(perfil)
    t = _expandir_dimensoes(_normalizar(texto), pads)
    achadas: dict[str, set] = defaultdict(set)
    for p in pads:
        consumir: list[tuple[int, int]] = []
        for m in p.regex.finditer(t):
            v = p._valor(m)
            if v is None:
                continue
            achadas[p.base].add(v)
            consumir.append(m.span())
        # Mascara o que foi lido, preservando o tamanho do texto.
        for a, b in consumir:
            t = t[:a] + (" " * (b - a)) + t[b:]
    return dict(achadas)


def _casa(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def concordancia(ma: dict, mb: dict, tol: float = TOLERANCIA_PADRAO) -> float:
    """
    Acordo entre duas extracoes, em [-1, 1].

    So entram as unidades base que os DOIS lados declaram: acordo soma, conflito
    subtrai, e silencio de um dos lados nao conta. A assimetria e deliberada --
    ausencia nao e evidencia, divergencia e.

    Devolve 0,0 quando nao ha base em comum, que e o valor neutro: o sinal se
    cala em vez de chutar.
    """
    comuns = set(ma) & set(mb)
    if not comuns:
        return 0.0
    acordo = conflito = 0
    for base in comuns:
        if any(_casa(x, y, tol) for x in ma[base] for y in mb[base]):
            acordo += 1
        else:
            conflito += 1
    return (acordo - conflito) / len(comuns)


def compativel(consulta: dict, candidato: dict, tol: float = TOLERANCIA_PADRAO) -> bool:
    """
    True quando o candidato satisfaz TODAS as medidas da consulta.

    E o predicado do indice de recuperacao: mais rigido que `concordancia`, para
    gerar um conjunto pequeno e preciso (mediana de 3 itens no MMH).
    """
    if not consulta:
        return False
    for base, vals in consulta.items():
        if base not in candidato:
            return False
        if not any(_casa(x, y, tol) for x in vals for y in candidato[base]):
            return False
    return True


def indice(textos: list[str], perfil: Optional[PerfilDominio] = None) -> list[dict]:
    """Extrai as medidas de uma lista de textos, na mesma ordem."""
    return [extrair(t, perfil) for t in textos]


def resumo(medidas: list[dict]) -> dict:
    """Quantos textos tem medida e quantas bases aparecem (para log e YAML)."""
    por_base: dict[str, int] = defaultdict(int)
    for m in medidas:
        for base in m:
            por_base[base] += 1
    return {
        "com_medida": sum(1 for m in medidas if m),
        "total": len(medidas),
        "por_base": dict(sorted(por_base.items(), key=lambda kv: -kv[1])),
    }


def satisfacao(consulta: dict, candidato: dict,
               tol: float = TOLERANCIA_PADRAO) -> float:
    """
    Fracao das unidades base da CONSULTA que o candidato satisfaz, em [0,1].

    Diferente de `concordancia`, que so olha as bases comuns e pune divergencia,
    aqui o denominador e o que a consulta pede: candidato silencioso sobre uma
    medida pedida simplesmente nao pontua nela. E a forma que a medicao aprovou
    -- premiar acordo funciona, punir divergencia nao (ver `concordancia`).
    """
    if not consulta:
        return 0.0
    ok = 0
    for base, vals in consulta.items():
        vc = candidato.get(base)
        if vc and any(_casa(x, y, tol) for x in vals for y in vc):
            ok += 1
    return ok / len(consulta)


# ---------------------------------------------------------------------------
# Inspecao
# ---------------------------------------------------------------------------

def _cli() -> None:
    """
    Mostra o que o extrator ve no dataset ativo.

    E a ferramenta para levar isto a um dominio novo: se as medidas do corpus
    nao aparecem aqui, falta declarar `aliases` na unidade correspondente do
    perfil, e o sinal de medida esta inativo sem avisar.
    """
    import argparse

    from .config import ativar, listar_datasets

    ap = argparse.ArgumentParser(
        description="Inspeciona a extracao de medidas do dataset ativo.")
    ap.add_argument("--dataset", default="",
                    help=f"dataset. Disponiveis: {', '.join(listar_datasets())}")
    ap.add_argument("--perfil", default="", help="forca outro perfil de dominio")
    ap.add_argument("--amostra", type=int, default=8,
                    help="quantos textos mostrar em detalhe (padrao: 8)")
    ap.add_argument("--texto", default="",
                    help="extrai de um texto dado e sai, sem ler o dataset")
    args = ap.parse_args()

    ativar(args.dataset, perfil=args.perfil)
    perfil = perfil_ativo()

    print(f"perfil: {perfil.nome}  |  padroes montados: {len(padroes(perfil))}")
    com_alias = {n: s.get("aliases") for n, s in (perfil.unidades or {}).items()
                 if s.get("aliases")}
    print(f"unidades extraiveis ({len(com_alias)}):")
    for nome, aliases in com_alias.items():
        spec = perfil.unidades[nome]
        extras = [k for k in ("prefixo", "fracao", "inteiro") if spec.get(k)]
        faixa = f" faixa={spec['faixa']}" if spec.get("faixa") else ""
        print(f"  {nome:<10} base={spec.get('base', nome):<8} "
              f"aliases={list(aliases)}{faixa}"
              + (f" [{', '.join(extras)}]" if extras else ""))
    sem_alias = [n for n, s in (perfil.unidades or {}).items() if not s.get("aliases")]
    if sem_alias:
        print(f"sem aliases (nao extraidas de texto livre): {', '.join(sem_alias)}")

    if args.texto:
        print(f"\n  {args.texto}\n  -> {extrair(args.texto, perfil)}")
        return

    from .preprocessamento import carregar_dados

    df = carregar_dados()
    for coluna, rotulo in (("item_efisco", "consultas"), ("item_catmat", "catalogo")):
        textos = df[coluna].fillna("").astype(str).tolist()
        medidas = indice(textos, perfil)
        r = resumo(medidas)
        print(f"\n{rotulo}: {r['com_medida']}/{r['total']} com medida "
              f"({100 * r['com_medida'] / max(1, r['total']):.0f}%)")
        print(f"  por base: {r['por_base']}")
        mostrados = 0
        for t, m in zip(textos, medidas):
            if not m or mostrados >= args.amostra:
                continue
            print(f"  {t[:96]}")
            print(f"    -> {m}")
            mostrados += 1


if __name__ == "__main__":
    _cli()
