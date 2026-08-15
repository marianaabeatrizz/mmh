"""
rede_semantica_mmh.py
Grafo lexico de dominio para CATMAT <-> e-Fisco.

Baseado em: "Rede Semantica e Ontologia para o casamento CATMAT <-> e-Fisco"
Secao 3.5 — modelo de dados (esquema do grafo)
Secao 3.6 — pipeline de construcao

Nos (5 tipos conforme o doc):
    PDM          — classe/padrao descritivo do material; no de ancoragem
    Conceito     — forma canonica / sentido do dominio (synset de dominio)
    Termo        — forma de superficie observada (jargao, abreviacao, variante)
    ValorDeAtributo — valor tipado de caracteristica tecnica (dimensao, cor, material)
    Unidade      — unidade de medida com fator de conversao

Arestas (com peso e fonte, conforme o doc):
    formaCanonicaDe    Termo -> Conceito
    abreviacaoDe       Termo -> Conceito  (subrelacao de formaCanonicaDe)
    varianteOrtograficaDe  Termo -> Conceito
    sinonimoDe         Conceito <-> Conceito
    hiperonimoDe       Conceito -> Conceito (hierarquia)
    relacionadoA       Conceito <-> Conceito (associacao fraca)
    mapeiaParaPDM      Conceito/ValorDeAtributo -> PDM
    temUnidade         ValorDeAtributo -> Unidade
    convertePara       Unidade -> Unidade

Pipeline de construcao (secao 3.6):
    (a) Semente estrutural (verteba)  — rotulos CATMAT/PDM extraidos de item_catmat
    (b) Mineracao no historico (carne) — pares efisco<->catmat para jargao real
    (c) Semente lexical curada (tempero) — abreviacoes/siglas do dominio de compras
    (f) Unidades e numeros            — tabela com fatores de conversao
"""

import re
import ast
import unicodedata
import logging
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Reutiliza funcoes de preprocessamento ja criadas
from preprocessamento_mmh import (
    carregar_dados,
    extrair_atributos_catmat,
    normalizar_texto,
    tokenizar,
    remover_stopwords,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
HOJE = str(date.today())

# Versao do lexico (sec. 3.6, "grafo lexico de dominio versionado"): entra no
# grafo, no GraphML e na vista SKOS, para que cada release seja rastreavel.
VERSAO_LEXICO = "2.0"

# Namespace da vista SKOS/RDF (sec. 3.5) — mesma base usada pela ontologia OWL.
IRI_BASE = "http://mmh.sad.pe.gov.br/lexico"

# ---------------------------------------------------------------------------
# Constantes do esquema (sec. 3.5)
# ---------------------------------------------------------------------------

TIPOS_NO = {"PDM", "Conceito", "Termo", "ValorDeAtributo", "Unidade"}

TIPOS_ARESTA = {
    "formaCanonicaDe", "abreviacaoDe", "varianteOrtograficaDe",
    "sinonimoDe", "hiperonimoDe", "relacionadoA",
    "mapeiaParaPDM", "temUnidade", "convertePara",
}

# Paleta de cores por tipo de no para visualizacao
COR_NO = {
    "PDM":             "#1f77b4",   # azul
    "Conceito":        "#2ca02c",   # verde
    "Termo":           "#ff7f0e",   # laranja
    "ValorDeAtributo": "#9467bd",   # roxo
    "Unidade":         "#8c564b",   # marrom
}

# Estilo de aresta por relacao
ESTILO_ARESTA = {
    "formaCanonicaDe":        {"color": "#ff7f0e", "style": "solid",  "width": 1.5},
    "abreviacaoDe":           {"color": "#ff7f0e", "style": "dashed", "width": 1.2},
    "varianteOrtograficaDe":  {"color": "#d62728", "style": "dotted", "width": 1.0},
    "sinonimoDe":             {"color": "#2ca02c", "style": "solid",  "width": 1.5},
    "hiperonimoDe":           {"color": "#17becf", "style": "solid",  "width": 1.2},
    "relacionadoA":           {"color": "#bcbd22", "style": "dashed", "width": 0.8},
    "mapeiaParaPDM":          {"color": "#1f77b4", "style": "solid",  "width": 2.0},
    "temUnidade":             {"color": "#9467bd", "style": "solid",  "width": 1.0},
    "convertePara":           {"color": "#8c564b", "style": "dotted", "width": 1.0},
}

# ---------------------------------------------------------------------------
# Abreviacoes curadas do dominio de compras (secao 3.6-c)
# ---------------------------------------------------------------------------

ABREVIACOES_CURADAS = {
    # conectores e preposicoes
    "C/":    "COM",
    "P/":    "PARA",
    "S/":    "SEM",
    # unidades de embalagem
    "CX":    "CAIXA",
    "PCT":   "PACOTE",
    "FR":    "FRASCO",
    "AMP":   "AMPOLA",
    "ENV":   "ENVELOPE",
    "RL":    "ROLO",
    "PC":    "PECA",
    "PR":    "PAR",
    "KIT":   "KIT",
    "UND":   "UNIDADE",
    "UN":    "UNIDADE",
    # adjetivos comuns
    "DESC":  "DESCARTAVEL",
    "ESTER": "ESTERIL",
    "INOX":  "ACO INOXIDAVEL",
    "PVC":   "CLORETO DE POLIVINILA",
    # gauge / calibre
    "GA":    "GAUGE",
    # referencias tecnicas residuais
    "LUER":  "LUER LOCK",
}

# Variantes ortograficas observadas nos dados (secao 3.6-b/c)
VARIANTES_ORTOGRAFICAS = {
    "QUICKLE":    "QUINCKE",
    "QUINKER":    "QUINCKE",
    "TOUHY":      "TUOHY",
    "ESTERILIZADOR": "ESTERILIZACAO",
    "BIOPCIA":    "BIOPSIA",
    "ESPESSURA":  "ESPESSURA",
    "ANASTESIA":  "ANESTESIA",
}

# Sinonimos de dominio curados
SINONIMOS_CURADOS = {
    "AGULHA PARA PUNCAO": "AGULHA PUNCAO",
    "BISEL":              "BISEL CORTANTE",
    "CANHAO":             "CONECTOR",
    "TUBO":               "CANULA",
    "SERINGA":            "DISPOSITIVO INFUSAO",
}

# Hierarquias (hiperonimo -> lista de hiponimos)
HIERARQUIA_CURADA = {
    "AGULHA": [
        "AGULHA PUNCAO OSSEA",
        "AGULHA BIOPSIA",
        "AGULHA ANESTESIA RAQUIDIANA",
        "DISPOSITIVO P ANESTESIA REGIONAL",
        "AGULHA PERIDURAL",
    ],
    "DISPOSITIVO MEDICO": [
        "AGULHA",
        "FIO DE SUTURA AGULHADO",
        "DISPOSITIVO PORTÁTIL PARA TERAPIA RESPIRATORIA",
        "TUBO HOSPITALAR",
    ],
    "AGULHA ANESTESIA": [
        "AGULHA ANESTESIA RAQUIDIANA",
        "DISPOSITIVO P ANESTESIA REGIONAL",
    ],
}

# ---------------------------------------------------------------------------
# Tabela de unidades (secao 3.6-f)
# ---------------------------------------------------------------------------

UNIDADES = {
    # comprimento
    "MM":   {"categoria": "comprimento", "base": "MM"},
    "CM":   {"categoria": "comprimento", "base": "MM", "fator_para_base": 10.0},
    "M":    {"categoria": "comprimento", "base": "MM", "fator_para_base": 1000.0},
    # volume
    "ML":   {"categoria": "volume", "base": "ML"},
    "L":    {"categoria": "volume", "base": "ML", "fator_para_base": 1000.0},
    # massa
    "G":    {"categoria": "massa", "base": "G"},
    "KG":   {"categoria": "massa", "base": "G", "fator_para_base": 1000.0},
    "MG":   {"categoria": "massa", "base": "G", "fator_para_base": 0.001},
    # contagem
    "UNIDADE": {"categoria": "contagem", "base": "UNIDADE"},
    "DUZIA":   {"categoria": "contagem", "base": "UNIDADE", "fator_para_base": 12.0},
    "CENTO":   {"categoria": "contagem", "base": "UNIDADE", "fator_para_base": 100.0},
    # calibre
    "G_GAUGE": {"categoria": "calibre",  "base": "G_GAUGE"},   # gauge medico
    "FR":      {"categoria": "calibre",  "base": "FR"},         # french/charriere
    # embalagem
    "CAIXA":   {"categoria": "embalagem", "base": "CAIXA"},
    "PACOTE":  {"categoria": "embalagem", "base": "PACOTE"},
    "ROLO":    {"categoria": "embalagem", "base": "ROLO"},
}

# ---------------------------------------------------------------------------
# Construcao do grafo
# ---------------------------------------------------------------------------

def _id_no(tipo: str, texto: str) -> str:
    """Gera ID canonico de no: TIPO::TEXTO_NORMALIZADO."""
    texto_norm = unicodedata.normalize("NFD", texto)
    texto_norm = "".join(c for c in texto_norm if unicodedata.category(c) != "Mn")
    texto_norm = re.sub(r"\s+", "_", texto_norm.upper().strip())
    return f"{tipo}::{texto_norm}"


def _add_no(G: nx.DiGraph, tipo: str, label: str, **props) -> str:
    nid = _id_no(tipo, label)
    if nid not in G:
        G.add_node(nid, tipo=tipo, label=label, **props)
    return nid


def _add_aresta(
    G: nx.DiGraph,
    origem: str,
    destino: str,
    relacao: str,
    peso: float = 1.0,
    fonte: str = "curado",
    frequencia: int = 1,
    validado_por: str = "",
):
    """Adiciona (ou reforca) uma aresta tipada.

    Propriedades gravadas conforme sec. 3.5: peso (0-1, confianca), fonte,
    frequencia, validadoPor (quem curou; vazio = nao validado) e data.
    """
    key = (origem, destino, relacao)
    if G.has_edge(origem, destino, key=relacao):
        G[origem][destino][relacao]["frequencia"] += 1
        G[origem][destino][relacao]["peso"] = min(
            1.0, G[origem][destino][relacao]["peso"] + 0.05
        )
        # Uma validacao humana posterior sobrescreve o campo vazio.
        if validado_por:
            G[origem][destino][relacao]["validadoPor"] = validado_por
    else:
        G.add_edge(
            origem, destino, key=relacao,
            relacao=relacao, peso=peso, fonte=fonte,
            frequencia=frequencia, validadoPor=validado_por, data=HOJE,
        )


# ---- (a) Semente estrutural: verteba do CATMAT/PDM -------------------------

def _construir_verteba(G: nx.DiGraph, df: pd.DataFrame) -> None:
    """
    Secao 3.6-a: ingerir rotulos CATMAT/PDM.
    Cada tipo_produto vira um no PDM.
    Atributos viram Conceito/ValorDeAtributo ligados ao PDM via mapeiaParaPDM.
    """
    log.info("(a) Construindo verteba estrutural a partir de %d registros CATMAT...", len(df))

    # Acumula frequencia de cada PDM
    freq_pdm: Counter = Counter()
    freq_conceito: Counter = Counter()

    for _, row in df.iterrows():
        atribs = extrair_atributos_catmat(row["item_catmat"])
        if not atribs:
            continue

        tipo_pdm = atribs.get("tipo_produto", "")
        if not tipo_pdm:
            continue

        freq_pdm[tipo_pdm] += 1

        nid_pdm = _add_no(G, "PDM", tipo_pdm, classe=row.get("classe_catmat", ""))

        for chave, valor in atribs.items():
            if chave in ("tipo_produto", "texto_completo") or not valor:
                continue

            valor_norm = normalizar_texto(valor, manter_maiusculas=True)

            # Detecta se o valor contem unidade de medida
            if _e_valor_numerico(valor_norm):
                nid_val = _add_no(G, "ValorDeAtributo", valor_norm,
                                  caracteristica=chave)
                _add_aresta(G, nid_val, nid_pdm, "mapeiaParaPDM",
                            fonte="historico", frequencia=1)
                _associar_unidade(G, nid_val, valor_norm)
            else:
                freq_conceito[valor_norm] += 1
                nid_conc = _add_no(G, "Conceito", valor_norm,
                                   caracteristica=chave)
                _add_aresta(G, nid_conc, nid_pdm, "mapeiaParaPDM",
                            fonte="historico", frequencia=1)

    # Atualiza pesos de frequencia nos nos PDM
    for pdm, freq in freq_pdm.items():
        nid = _id_no("PDM", pdm)
        if nid in G:
            G.nodes[nid]["frequencia"] = freq

    log.info("  -> %d nos PDM criados", sum(1 for n, d in G.nodes(data=True) if d.get("tipo") == "PDM"))
    log.info("  -> %d nos Conceito criados", sum(1 for n, d in G.nodes(data=True) if d.get("tipo") == "Conceito"))


# ---- (b) Mineracao no historico: jargao do e-Fisco -------------------------

def _construir_jargao_historico(G: nx.DiGraph, df: pd.DataFrame) -> None:
    """
    Secao 3.6-b: a partir de pares conhecidos (efisco <-> catmat vinculados),
    minera abreviacoes e termos relacionados por alinhamento de tokens.
    """
    log.info("(b) Minerando jargao do historico de vinculos...")

    # Conta co-ocorrencias e frequencias marginais
    cooc: Counter = Counter()        # (te, tc) -> n vezes que co-ocorrem no mesmo par
    freq_efisco: Counter = Counter() # te -> n pares que contêm te
    freq_catmat: Counter = Counter() # tc -> n pares que contêm tc

    for _, row in df.iterrows():
        toks_efisco = _tokens_limpos(row["item_efisco"])
        toks_catmat = _tokens_limpos(row["item_catmat"])

        for te in set(toks_efisco):   # set: conta 1x por par (frequencia documental)
            freq_efisco[te] += 1
        for tc in set(toks_catmat):
            freq_catmat[tc] += 1

        # set() nos dois lados: cnt = nº de PARES em que ambos co-ocorrem
        # (co-ocorrência documental), na MESMA base de contagem de
        # freq_efisco/freq_catmat. Sem o set(), o produto cruz com repetição
        # inflava o numerador do PMI para tokens repetidos dentro de um texto.
        for te in set(toks_efisco):
            for tc in set(toks_catmat):
                if te != tc:
                    cooc[(te, tc)] += 1

    N = len(df)  # numero total de pares (denominador do PMI)

    # 1. Tokens do efisco que sao prefixo/sufixo de tokens do catmat -> abreviacaoDe
    #
    # Guarda de precisao: se o token curto TAMBEM e uma palavra estabelecida no
    # vocabulario do CATMAT, ele nao e abreviacao de nada — e um termo por
    # direito proprio. Sem isto o alinhamento por prefixo produz arestas falsas
    # como AGULHA -> AGULHADO (porque "AGULHADO".startswith("AGULHA")), que
    # depois contaminam a normalizacao e a ancoragem do registro inteiro.
    FREQ_PALAVRA_PROPRIA = 3
    pares_abreviacao: list[tuple] = []
    n_bloqueadas = 0
    for (te, tc), cnt in cooc.items():
        if len(te) >= len(tc) or cnt < 3:
            continue
        if not (tc.startswith(te) or tc.endswith(te)):
            continue
        if freq_catmat.get(te, 0) >= FREQ_PALAVRA_PROPRIA:
            n_bloqueadas += 1
            continue
        pares_abreviacao.append((te, tc, cnt))

    for te, tc, cnt in pares_abreviacao:
        nid_termo = _add_no(G, "Termo", te)
        nid_conc = _id_no("Conceito", tc)
        if nid_conc not in G:
            _add_no(G, "Conceito", tc)
        peso = min(0.95, 0.5 + cnt * 0.05)
        _add_aresta(G, nid_termo, nid_conc, "abreviacaoDe",
                    peso=peso, fonte="historico", frequencia=cnt)

    if n_bloqueadas:
        log.info("  -> %d falsas abreviacoes bloqueadas (token e palavra propria)",
                 n_bloqueadas)

    # 2. PMI real: PMI(te,tc) = log( P(te,tc) / (P(te)*P(tc)) )
    #              = log( cnt*N / (freq_efisco[te] * freq_catmat[tc]) )
    # Usamos PPMI (positive PMI): max(PMI, 0) — descarta pares negativos.
    # Normalizado para [0,1] via NPMI = PMI / (-log P(te,tc)).
    import math
    for (te, tc), cnt in cooc.items():
        if cnt < 3 or len(te) < 4 or len(tc) < 4:
            continue
        # Evita pares que já viraram abreviacaoDe
        if tc.startswith(te) or te.startswith(tc):
            continue

        p_te  = freq_efisco[te] / N
        p_tc  = freq_catmat[tc] / N
        p_co  = cnt / N

        if p_te <= 0 or p_tc <= 0 or p_co <= 0:
            continue

        pmi_val = math.log(p_co / (p_te * p_tc))  # pode ser negativo

        # PPMI: zera negativos
        if pmi_val <= 0:
            continue

        # Normaliza: NPMI = PMI / -log(P(co))  → [-1, 1]; aqui fica em [0,1]
        npmi = pmi_val / (-math.log(p_co))
        npmi = max(0.0, min(1.0, npmi))

        if npmi < 0.10:   # filtra ruido de baixo sinal (NPMI >= 0.10)
            continue

        nid_te = _add_no(G, "Termo", te)
        nid_tc_conc = _id_no("Conceito", tc)
        if nid_tc_conc not in G:
            nid_tc_conc = _add_no(G, "Conceito", tc)

        peso = min(0.8, npmi)
        _add_aresta(G, nid_te, nid_tc_conc, "relacionadoA",
                    peso=peso, fonte="historico", frequencia=cnt)

    log.info("  -> %d nos Termo minerados", sum(1 for n, d in G.nodes(data=True) if d.get("tipo") == "Termo"))


# ---- (c) Semente lexical curada ---------------------------------------------

def _construir_lexical_curado(G: nx.DiGraph) -> None:
    """
    Secao 3.6-c: abreviacoes/siglas do dominio de compras (curadas manualmente).
    Peso menor que a verteba, pois sao genericas.
    """
    log.info("(c) Inserindo semente lexical curada...")

    # Abreviacoes
    for abrev, forma_longa in ABREVIACOES_CURADAS.items():
        nid_t = _add_no(G, "Termo", abrev)
        nid_c = _add_no(G, "Conceito", forma_longa)
        _add_aresta(G, nid_t, nid_c, "abreviacaoDe", peso=0.95, fonte="curado")
        _add_aresta(G, nid_t, nid_c, "formaCanonicaDe", peso=0.95, fonte="curado")

    # Variantes ortograficas
    for variante, canonica in VARIANTES_ORTOGRAFICAS.items():
        nid_t = _add_no(G, "Termo", variante)
        nid_c = _add_no(G, "Conceito", canonica)
        _add_aresta(G, nid_t, nid_c, "varianteOrtograficaDe", peso=0.9, fonte="curado")
        _add_aresta(G, nid_t, nid_c, "formaCanonicaDe", peso=0.9, fonte="curado")

    # Sinonimos curados
    for termo, sinonimo in SINONIMOS_CURADOS.items():
        nid_a = _add_no(G, "Conceito", termo)
        nid_b = _add_no(G, "Conceito", sinonimo)
        _add_aresta(G, nid_a, nid_b, "sinonimoDe", peso=0.9, fonte="curado")
        _add_aresta(G, nid_b, nid_a, "sinonimoDe", peso=0.9, fonte="curado")

    # Hierarquias curadas
    for hiper, hipos in HIERARQUIA_CURADA.items():
        nid_hiper = _add_no(G, "Conceito", hiper)
        for hipo in hipos:
            nid_hipo = _add_no(G, "Conceito", hipo)
            _add_aresta(G, nid_hipo, nid_hiper, "hiperonimoDe", peso=0.85, fonte="curado")

    log.info("  -> semente lexical curada inserida (%d abrev., %d variantes, %d sinonimos, %d hierarquias)",
             len(ABREVIACOES_CURADAS), len(VARIANTES_ORTOGRAFICAS),
             len(SINONIMOS_CURADOS), sum(len(v) for v in HIERARQUIA_CURADA.values()))


# ---- (d) Inducao por embeddings ---------------------------------------------

# Parametros da inducao (sec. 3.6-d). fastText por subpalavras: cobre OOV,
# que e exatamente a fraqueza medida na fase [1] do pipeline.
INDUCAO_MIN_COUNT   = 3      # token precisa aparecer em >= 3 registros
INDUCAO_DIM         = 100
INDUCAO_JANELA      = 5
INDUCAO_EPOCAS      = 30
INDUCAO_TOPN        = 5      # vizinhos propostos por termo
INDUCAO_SIM_MINIMA  = 0.72   # abaixo disto o vizinho vira ruido
INDUCAO_MAX_TERMOS  = 800    # so os mais frequentes (custo de curadoria)


def _construir_inducao_embeddings(G: nx.DiGraph, df: pd.DataFrame) -> None:
    """
    Secao 3.6-d: inducao por embeddings — recall de sinonimos.

    Treina fastText NO CORPUS DE CATALOGO (nao em prosa generica, como o
    documento faz questao de frisar) e propoe os vizinhos mais proximos de
    cada termo frequente como candidatos a sinonimoDe.

    Ponto central da secao: "Tudo entra como candidato, nao como verdade".
    Por isso as arestas nascem com fonte='embeddings', peso derivado da
    similaridade e validadoPor='' — vao para a fila de curadoria (3.6-e) e
    entram na expansao com peso menor que verteba/historico.
    """
    log.info("(d) Induzindo sinonimos por embeddings (fastText)...")

    try:
        from gensim.models import FastText
    except ImportError:
        log.warning("  -> gensim ausente; etapa (d) pulada.")
        return

    # Corpus: cada registro (efisco + catmat) e uma "sentenca" de tokens.
    corpus: list[list[str]] = []
    for _, row in df.iterrows():
        for col in ("item_efisco", "item_catmat"):
            toks = _tokens_limpos(row.get(col, ""))
            if len(toks) >= 2:
                corpus.append(toks)

    if len(corpus) < 50:
        log.warning("  -> corpus pequeno demais (%d); etapa (d) pulada.", len(corpus))
        return

    modelo = FastText(
        vector_size=INDUCAO_DIM,
        window=INDUCAO_JANELA,
        min_count=INDUCAO_MIN_COUNT,
        sg=1,              # skip-gram: melhor para termos raros de dominio
        workers=4,
        seed=42,
    )
    modelo.build_vocab(corpus)
    modelo.train(corpus, total_examples=len(corpus), epochs=INDUCAO_EPOCAS)

    # Frequencia documental para priorizar os termos que realmente importam.
    freq: Counter = Counter()
    for sent in corpus:
        for tk in set(sent):
            freq[tk] += 1

    vocab = set(modelo.wv.index_to_key)
    termos_alvo = [t for t, _ in freq.most_common(INDUCAO_MAX_TERMOS) if t in vocab]

    n_arestas = 0
    for termo in termos_alvo:
        try:
            vizinhos = modelo.wv.most_similar(termo, topn=INDUCAO_TOPN)
        except KeyError:
            continue

        nid_termo = _add_no(G, "Conceito", termo)
        for vizinho, sim in vizinhos:
            if sim < INDUCAO_SIM_MINIMA or vizinho == termo:
                continue
            # Ja coberto por relacao mais forte (curada ou minerada)? nao duplica.
            nid_viz = _id_no("Conceito", vizinho)
            if nid_viz in G and G.has_edge(nid_termo, nid_viz):
                continue
            nid_viz = _add_no(G, "Conceito", vizinho)
            # Peso teto 0.6: sempre abaixo de curado (0.85-0.95) e do historico.
            _add_aresta(
                G, nid_termo, nid_viz, "sinonimoDe",
                peso=round(min(0.6, float(sim) * 0.7), 3),
                fonte="embeddings",
                frequencia=freq[termo],
            )
            n_arestas += 1

    log.info("  -> %d candidatos a sinonimo induzidos (%d termos, vocab=%d)",
             n_arestas, len(termos_alvo), len(vocab))


# ---- (e) Fila de curadoria --------------------------------------------------

def fila_curadoria(G: nx.MultiDiGraph, limite: int = 200) -> list[dict]:
    """
    Secao 3.6-e: fila priorizada por FREQUENCIA x INCERTEZA.

    O especialista valida/rejeita as arestas duvidosas; curar os termos de
    alta frequencia resolve a maior parte do trafego (a logica de custo que o
    documento defende na sec. 3.11).

    Incerteza = 1 - peso  → aresta fraca e muito usada sobe no topo.
    Arestas ja validadas (validadoPor preenchido) e arestas curadas de origem
    nao entram na fila.
    """
    itens: list[dict] = []
    for origem, destino, dados in G.edges(data=True):
        if dados.get("validadoPor"):
            continue
        if dados.get("fonte") == "curado":
            continue

        peso = float(dados.get("peso", 0.5))
        freq = int(dados.get("frequencia", 1))
        incerteza = 1.0 - peso
        itens.append({
            "origem":     G.nodes[origem].get("label", origem),
            "destino":    G.nodes[destino].get("label", destino),
            "relacao":    dados.get("relacao", "?"),
            "fonte":      dados.get("fonte", "?"),
            "peso":       round(peso, 3),
            "frequencia": freq,
            "prioridade": round(freq * incerteza, 3),
        })

    itens.sort(key=lambda d: -d["prioridade"])
    return itens[:limite]


def exportar_fila_curadoria(G: nx.MultiDiGraph,
                            destino: Optional[Path] = None,
                            limite: int = 200) -> Path:
    """Grava a fila de curadoria em CSV para o especialista trabalhar nela."""
    destino = destino or (DATA_DIR / "fila_curadoria.csv")
    fila = fila_curadoria(G, limite=limite)
    df_fila = pd.DataFrame(fila)
    # Coluna em branco onde o especialista escreve o veredito.
    df_fila["veredito"] = ""      # aceito | rejeitado
    df_fila["validado_por"] = ""
    df_fila.to_csv(destino, index=False, sep="|", encoding="utf-8-sig")
    log.info("Fila de curadoria: %d arestas -> %s", len(df_fila), destino)
    return destino


def aplicar_curadoria(G: nx.MultiDiGraph, csv_curado: Path) -> int:
    """
    Le o CSV devolvido pelo especialista e aplica os vereditos ao grafo:
    'aceito' promove o peso e grava validadoPor; 'rejeitado' remove a aresta.

    Fecha o ciclo de governanca da sec. 3.9 (a fila so vale se o resultado
    volta para o grafo).
    """
    if not Path(csv_curado).exists():
        log.info("Sem CSV de curadoria em %s — nada a aplicar.", csv_curado)
        return 0

    df_cur = pd.read_csv(csv_curado, sep="|", dtype=str, encoding="utf-8-sig").fillna("")
    n_aplicados = 0

    for _, row in df_cur.iterrows():
        veredito = str(row.get("veredito", "")).strip().lower()
        if veredito not in ("aceito", "rejeitado"):
            continue

        relacao = row.get("relacao", "")
        # Descobre os ids reais dos nos a partir dos labels.
        nid_o = nid_d = None
        for tipo in TIPOS_NO:
            cand_o = _id_no(tipo, row.get("origem", ""))
            cand_d = _id_no(tipo, row.get("destino", ""))
            if cand_o in G:
                nid_o = cand_o
            if cand_d in G:
                nid_d = cand_d
        if not nid_o or not nid_d or not G.has_edge(nid_o, nid_d, key=relacao):
            continue

        if veredito == "rejeitado":
            G.remove_edge(nid_o, nid_d, key=relacao)
        else:
            G[nid_o][nid_d][relacao]["peso"] = 0.95
            G[nid_o][nid_d][relacao]["validadoPor"] = str(
                row.get("validado_por", "especialista")
            ) or "especialista"
        n_aplicados += 1

    log.info("Curadoria aplicada: %d arestas atualizadas.", n_aplicados)
    return n_aplicados


# ---- (f) Unidades -----------------------------------------------------------

def _construir_unidades(G: nx.DiGraph) -> None:
    """
    Secao 3.6-f: tabela de Unidade + convertePara.
    """
    log.info("(f) Inserindo tabela de unidades...")
    nos_unidade: dict[str, str] = {}

    for nome, props in UNIDADES.items():
        nid = _add_no(G, "Unidade", nome,
                      categoria=props["categoria"],
                      base=props["base"],
                      fator=props.get("fator_para_base", 1.0))
        nos_unidade[nome] = nid

    # Arestas convertePara para unidades da mesma categoria
    por_categoria: dict[str, list] = defaultdict(list)
    for nome, props in UNIDADES.items():
        por_categoria[props["categoria"]].append(nome)

    for cat, nomes in por_categoria.items():
        for i, n1 in enumerate(nomes):
            for n2 in nomes[i+1:]:
                f1 = UNIDADES[n1].get("fator_para_base", 1.0)
                f2 = UNIDADES[n2].get("fator_para_base", 1.0)
                fator = f1 / f2
                _add_aresta(G, nos_unidade[n1], nos_unidade[n2],
                            "convertePara", peso=1.0, fonte="curado",
                            frequencia=1)
                G[nos_unidade[n1]][nos_unidade[n2]]["convertePara"]["fator"] = fator

    log.info("  -> %d nos Unidade", len(nos_unidade))


# ---- Auxiliares -------------------------------------------------------------

def _tokens_limpos(texto: str) -> list[str]:
    t = normalizar_texto(texto, manter_maiusculas=True)
    toks = tokenizar(t)
    return [tk for tk in remover_stopwords(toks) if len(tk) > 2]


_RE_DIM = re.compile(r"\b\d[\d\.,]*\s*(?:MM|CM|M|ML|L|G|KG|MG|G_GAUGE|FR)\b", re.IGNORECASE)
_RE_NUM = re.compile(r"\b\d[\d\.,]*\b")


def _e_valor_numerico(texto: str) -> bool:
    return bool(_RE_DIM.search(texto) or _RE_NUM.search(texto))


def _associar_unidade(G: nx.DiGraph, nid_val: str, texto: str) -> None:
    """Liga ValorDeAtributo a Unidade via temUnidade."""
    for nome_unidade in UNIDADES:
        if re.search(r"\b" + nome_unidade + r"\b", texto, re.IGNORECASE):
            nid_u = _id_no("Unidade", nome_unidade)
            if nid_u in G:
                _add_aresta(G, nid_val, nid_u, "temUnidade",
                            peso=0.9, fonte="historico")
            break


# ---------------------------------------------------------------------------
# Funcao principal de construcao
# ---------------------------------------------------------------------------

def construir_grafo(arquivo: str = "principal", inducao: bool = True) -> nx.MultiDiGraph:
    """
    Executa o pipeline completo da Secao 3.6 e retorna o grafo.

    Etapas:
        (a) Semente estrutural (verteba) — CATMAT/PDM
        (b) Mineracao no historico (carne) — pares efisco<->catmat
        (c) Semente lexical curada (tempero) — abreviacoes/sinonimos
        (d) Inducao por embeddings (fastText) — candidatos a sinonimo
        (e) Curadoria — reaplica vereditos ja emitidos
        (f) Unidades e numeros

    Args:
        inducao: liga a etapa (d). Desligar acelera execucoes de teste.

    Returns:
        nx.MultiDiGraph com nos e arestas tipados conforme sec. 3.5.
    """
    df = carregar_dados(arquivo)
    G = nx.MultiDiGraph(
        nome="Rede Semantica CATMAT<->eFisco",
        versao=VERSAO_LEXICO,
        data=HOJE,
    )

    _construir_verteba(G, df)
    _construir_jargao_historico(G, df)
    _construir_lexical_curado(G)
    if inducao:
        _construir_inducao_embeddings(G, df)
    _construir_unidades(G)

    # (e) Reaplica vereditos de curadoria de rodadas anteriores, se houver.
    aplicar_curadoria(G, DATA_DIR / "fila_curadoria_revisada.csv")

    log.info("Grafo construido: %d nos | %d arestas", G.number_of_nodes(), G.number_of_edges())
    _resumir_grafo(G)
    return G


def _resumir_grafo(G: nx.MultiDiGraph) -> None:
    nos_por_tipo: Counter = Counter(d["tipo"] for _, d in G.nodes(data=True))
    arestas_por_relacao: Counter = Counter(
        d["relacao"] for _, _, d in G.edges(data=True)
    )
    log.info("=== Resumo do grafo ===")
    for tipo, cnt in sorted(nos_por_tipo.items()):
        log.info("  Nos %s: %d", tipo, cnt)
    for rel, cnt in sorted(arestas_por_relacao.items()):
        log.info("  Arestas %s: %d", rel, cnt)


# ---------------------------------------------------------------------------
# Servico de consulta (sec. 3.7)
# ---------------------------------------------------------------------------

_RELACOES_CANONICAS = ("formaCanonicaDe", "abreviacaoDe", "varianteOrtograficaDe")


def sentidos_de(G: nx.MultiDiGraph, termo: str) -> list[tuple[str, float]]:
    """Lista os Conceitos (sentidos) aos quais um Termo se liga, com o peso da
    aresta. Mais de um elemento = termo ambiguo, candidato a desambiguacao."""
    termo_norm = normalizar_texto(termo, manter_maiusculas=True)
    nid = _id_no("Termo", termo_norm)
    if nid not in G:
        return []
    sentidos: dict[str, float] = {}
    for _, dest, data in G.out_edges(nid, data=True):
        if data.get("relacao") in _RELACOES_CANONICAS:
            peso = float(data.get("peso", 0.5))
            # Mesmo destino por relacoes diferentes: fica o peso maior.
            if peso > sentidos.get(dest, 0.0):
                sentidos[dest] = peso
    return sorted(sentidos.items(), key=lambda kv: -kv[1])


def desambiguar(
    G: nx.MultiDiGraph,
    termo: str,
    contexto: Optional[list[str]] = None,
    decaimento: float = 0.5,
    max_saltos: int = 2,
) -> Optional[str]:
    """
    Passo 3 do servico de consulta (sec. 3.7):
    "quando um termo liga a varios Conceitos, escolher o de maior ativacao
    acumulada dada a vizinhanca do registro".

    Espalha ativacao a partir dos OUTROS tokens do registro (o contexto) e
    pontua cada sentido candidato pela ativacao que chegou ate ele. Resolve o
    caso 'manga (fruta) x manga (vestuario)' citado na sec. 3.3-3.

    Sem contexto, ou com contexto que nao ativa nenhum candidato, cai no
    sentido de maior peso de aresta — o comportamento antigo, agora explicito
    como desempate e nao como regra.
    """
    candidatos = sentidos_de(G, termo)
    if not candidatos:
        return None
    if len(candidatos) == 1 or not contexto:
        return G.nodes[candidatos[0][0]]["label"]

    # Ativacao a partir do contexto, excluindo o proprio termo.
    termo_norm = normalizar_texto(termo, manter_maiusculas=True)
    seeds = [c for c in contexto
             if normalizar_texto(c, manter_maiusculas=True) != termo_norm]
    if not seeds:
        return G.nodes[candidatos[0][0]]["label"]

    ativacao_labels = expandir_por_ativacao(
        G, seeds, limiar=0.05, decaimento=decaimento, max_saltos=max_saltos
    )

    melhor_nid, melhor_score = candidatos[0][0], -1.0
    for nid, peso_aresta in candidatos:
        label = G.nodes[nid].get("label", "")
        # Ativacao acumulada que o contexto entregou a este sentido,
        # modulada pela confianca da aresta Termo->Conceito.
        score = ativacao_labels.get(label, 0.0) * peso_aresta
        if score > melhor_score:
            melhor_nid, melhor_score = nid, score

    # Nenhum candidato foi tocado pelo contexto: desempata pelo peso.
    if melhor_score <= 0.0:
        melhor_nid = candidatos[0][0]

    return G.nodes[melhor_nid]["label"]


def normalizar_termo(
    G: nx.MultiDiGraph,
    termo: str,
    contexto: Optional[list[str]] = None,
) -> str:
    """
    Passo 1 do servico de consulta (sec. 3.7):
    Dado um token, retorna sua forma canonica via abreviacaoDe /
    varianteOrtograficaDe / formaCanonicaDe.

    Quando o termo e ambiguo (liga a mais de um Conceito) e ha `contexto`,
    o sentido e escolhido por ativacao acumulada (passo 3 da sec. 3.7) em vez
    da primeira aresta encontrada.
    """
    termo_norm = normalizar_texto(termo, manter_maiusculas=True)
    canonico = desambiguar(G, termo_norm, contexto=contexto)
    return canonico if canonico else termo_norm


# Relacoes por onde a ativacao pode fluir (sec. 3.7-2: "espalhar ativacao
# seguindo SO relacoes relevantes (sinonimo/hiperonimo)").
#
# `relacionadoA` fica de fora por padrao: e a relacao mais numerosa do grafo
# (vem do PMI, aos milhares) e a mais fraca semanticamente. Deixa-la propagar
# produz a "explosao de contexto" que a propria sec. 3.11 lista como risco —
# na pratica, uma agulha espinhal puxava CATGUT e POLIDIOXANONA, que sao
# materiais de sutura, so porque coocorrem no corpus.
RELACOES_EXPANSAO = ("sinonimoDe", "hiperonimoDe")


def expandir_por_ativacao(
    G: nx.MultiDiGraph,
    conceitos: list[str],
    limiar: float = 0.2,
    decaimento: float = 0.5,
    max_saltos: int = 2,
    relacoes: tuple = RELACOES_EXPANSAO,
    peso_relacionado: float = 0.25,
) -> dict[str, float]:
    """
    Passo 2 do servico de consulta (sec. 3.7):
    Spreading activation CONSTRAINED a partir de uma lista de conceitos.

    A(v) = sum_u A(u) * peso(u,v) * decaimento^dist,  cortando abaixo do limiar.

    Args:
        relacoes: relacoes por onde a ativacao flui (padrao: sinonimo/hiperonimo).
        peso_relacionado: se `relacionadoA` for incluido em `relacoes`, sua
            contribuicao e amortecida por este fator — associacao fraca nao
            deve pesar como equivalencia de sentido.
    """
    ativacao: dict[str, float] = {}
    for c in conceitos:
        nid = _id_no("Conceito", normalizar_texto(c, manter_maiusculas=True))
        if nid in G:
            ativacao[nid] = 1.0

    for salto in range(max_saltos):
        nova_ativacao = dict(ativacao)
        for nid, a in ativacao.items():
            if a < limiar:
                continue
            for _, dest, data in G.out_edges(nid, data=True):
                rel = data.get("relacao")
                if rel not in relacoes:
                    continue
                amortecimento = peso_relacionado if rel == "relacionadoA" else 1.0
                contrib = a * data.get("peso", 0.5) * (decaimento ** salto) * amortecimento
                nova_ativacao[dest] = nova_ativacao.get(dest, 0.0) + contrib
        ativacao = nova_ativacao

    return {G.nodes[n]["label"]: v for n, v in ativacao.items()
            if v >= limiar and n in G.nodes}


def ancorar_pdm(G: nx.MultiDiGraph, conceito: str) -> Optional[str]:
    """
    Passo 4 do servico de consulta (sec. 3.7):
    Dado um conceito, retorna o PDM ao qual ele mapeia (mapeiaParaPDM).

    Quando o conceito mapeia para mais de um PDM, vence o de maior peso —
    e nao o primeiro que a ordem de insercao no grafo entregar.
    """
    nid = _id_no("Conceito", normalizar_texto(conceito, manter_maiusculas=True))
    if nid not in G:
        return None
    melhor, melhor_peso = None, -1.0
    for _, dest, data in G.out_edges(nid, data=True):
        if data.get("relacao") == "mapeiaParaPDM":
            peso = float(data.get("peso", 0.5))
            if peso > melhor_peso:
                melhor, melhor_peso = G.nodes[dest]["label"], peso
    return melhor


def _indice_tokens_pdm(G: nx.MultiDiGraph) -> dict[str, list]:
    """
    Indice token -> [(rotulo do PDM, e_nucleo)] em cujo ROTULO o token aparece.

    `e_nucleo` marca o primeiro token do rotulo. Rotulos de PDM sao sintagmas
    com o substantivo-nucleo a frente e modificadores atras — AGULHA PUNCAO
    OSSEA, LAMINA BISTURI, FRALDA DESCARTAVEL. Distinguir os dois papeis
    impede que um modificador generico ("DESCARTAVEL") ancore um item no PDM
    errado so por aparecer no rotulo dele.
    """
    if "_idx_tokens_pdm" in G.graph:
        return G.graph["_idx_tokens_pdm"]
    idx: dict[str, list] = defaultdict(list)
    for _, dados in G.nodes(data=True):
        if dados.get("tipo") != "PDM":
            continue
        label = dados.get("label", "")
        toks = _tokens_limpos(label)
        for pos, tk in enumerate(toks):
            idx[tk].append((label, pos == 0))
    G.graph["_idx_tokens_pdm"] = idx
    return idx


# Um token que aponta para dezenas de PDMs (ESTERIL, DESCARTAVEL) nao diz nada
# sobre QUAL PDM e; um que aponta para poucos e altamente discriminativo. O voto
# e dividido pelo alcance do token — a mesma logica do IDF.
PESO_NUCLEO_PDM = 3.0    # token e o substantivo-nucleo do rotulo do PDM
PESO_MODIF_PDM  = 0.4    # token e so um modificador do rotulo
PESO_ATRIBUTO   = 1.0    # token aparece como valor de atributo do PDM


def ancorar_registro(G: nx.MultiDiGraph, tokens: list[str]) -> Optional[str]:
    """
    Ancoragem em nivel de REGISTRO (passo 4 da sec. 3.7).

    Combina duas evidencias, ambas ponderadas por especificidade:

    1. O token aparece no ROTULO do PDM ("AGULHA" em "AGULHA PUNCAO OSSEA").
       E a evidencia forte: e o nome da classe, nao uma caracteristica dela.
    2. O token tem aresta mapeiaParaPDM (veio de valor de atributo).
       Evidencia fraca: "ESTERIL" e valor de atributo de 22 PDMs distintos.

    Sem a ponderacao por especificidade, os tokens genericos de embalagem e
    esterilidade — que aparecem em quase todo item hospitalar — decidiam a
    ancoragem sozinhos, e uma agulha espinhal ancorava em LAMINA BISTURI.
    """
    idx_rotulos = _indice_tokens_pdm(G)
    votos: dict[str, float] = defaultdict(float)

    for tk in tokens:
        tk_norm = normalizar_texto(tk, manter_maiusculas=True)

        # 1. Token no rotulo do PDM — nucleo pesa muito mais que modificador
        pdms_rotulo = idx_rotulos.get(tk_norm, ())
        if pdms_rotulo:
            alcance = len(pdms_rotulo) ** 0.5
            for label, e_nucleo in pdms_rotulo:
                base = PESO_NUCLEO_PDM if e_nucleo else PESO_MODIF_PDM
                votos[label] += base / alcance

        # 2. Token como valor de atributo mapeado
        nid = _id_no("Conceito", tk_norm)
        if nid not in G:
            continue
        arestas = [
            (G.nodes[d]["label"], float(dt.get("peso", 0.5)))
            for _, d, dt in G.out_edges(nid, data=True)
            if dt.get("relacao") == "mapeiaParaPDM"
        ]
        if not arestas:
            continue
        fator = PESO_ATRIBUTO / len(arestas)
        for label, peso in arestas:
            votos[label] += peso * fator

    if not votos:
        return None
    return max(votos.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Visualizacao
# ---------------------------------------------------------------------------

def _vizinhanca_pdm(G: nx.MultiDiGraph, pdm_label: str, raio: int = 2) -> nx.MultiDiGraph:
    """Extrai subgrafo centrado em um PDM ate `raio` saltos."""
    nid_pdm = _id_no("PDM", pdm_label)
    if nid_pdm not in G:
        # Busca parcial
        matches = [n for n, d in G.nodes(data=True)
                   if d.get("tipo") == "PDM" and pdm_label.upper() in d.get("label", "")]
        if not matches:
            return G
        nid_pdm = matches[0]

    nos_sub = {nid_pdm}
    fronteira = {nid_pdm}
    for _ in range(raio):
        nova = set()
        for n in fronteira:
            nova |= set(G.predecessors(n)) | set(G.successors(n))
        fronteira = nova - nos_sub
        nos_sub |= fronteira

    return G.subgraph(nos_sub).copy()


def visualizar_grafo(
    G: nx.MultiDiGraph,
    pdm_foco: Optional[str] = None,
    max_nos: int = 80,
    titulo: str = "Rede Semantica CATMAT <-> e-Fisco",
    salvar_em: Optional[Path] = None,
) -> None:
    """
    Gera visualizacao do grafo ou de um subgrafo centrado em um PDM.

    Cores por tipo de no (ver COR_NO).
    Estilos de aresta por relacao (ver ESTILO_ARESTA).
    """
    if pdm_foco:
        sub = _vizinhanca_pdm(G, pdm_foco, raio=2)
        titulo = f"Vizinhanca de '{pdm_foco}'"
    else:
        # Subgrafo dos nos mais conectados ate max_nos
        graus = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        top_nos = [n for n, _ in graus[:max_nos]]
        sub = G.subgraph(top_nos).copy()

    if sub.number_of_nodes() == 0:
        log.warning("Subgrafo vazio — nada para visualizar.")
        return

    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    # Layout
    try:
        pos = nx.kamada_kawai_layout(sub)
    except Exception:
        pos = nx.spring_layout(sub, k=1.5, seed=42)

    # Desenha nos por tipo
    for tipo, cor in COR_NO.items():
        nos_tipo = [n for n, d in sub.nodes(data=True) if d.get("tipo") == tipo]
        if not nos_tipo:
            continue
        tamanho = {"PDM": 900, "Conceito": 500, "Termo": 350,
                   "ValorDeAtributo": 250, "Unidade": 400}.get(tipo, 300)
        nx.draw_networkx_nodes(sub, pos, nodelist=nos_tipo,
                               node_color=cor, node_size=tamanho,
                               alpha=0.9, ax=ax)

    # Rotulos dos nos (apenas os mais conectados)
    nos_para_rotulo = {n for n, _ in sorted(sub.degree(), key=lambda x: x[1], reverse=True)[:40]}
    rotulos = {n: sub.nodes[n].get("label", n)[:25] for n in nos_para_rotulo}
    nx.draw_networkx_labels(sub, pos, labels=rotulos, font_size=6,
                            font_color="white", ax=ax)

    # Desenha arestas por tipo de relacao
    for relacao, estilo in ESTILO_ARESTA.items():
        arestas_rel = [
            (u, v) for u, v, d in sub.edges(data=True)
            if d.get("relacao") == relacao
        ]
        if not arestas_rel:
            continue
        nx.draw_networkx_edges(
            sub, pos, edgelist=arestas_rel,
            edge_color=estilo["color"],
            style=estilo["style"],
            width=estilo["width"],
            alpha=0.7,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

    # Legenda de nos
    patches_no = [
        mpatches.Patch(color=cor, label=tipo)
        for tipo, cor in COR_NO.items()
    ]
    # Legenda de arestas
    patches_aresta = [
        mpatches.Patch(color=est["color"], label=rel, linestyle=est["style"])
        for rel, est in ESTILO_ARESTA.items()
        if any(d.get("relacao") == rel for _, _, d in sub.edges(data=True))
    ]
    leg1 = ax.legend(handles=patches_no, loc="upper left", fontsize=7,
                     title="Tipo de No", facecolor="#1a1a1a", labelcolor="white",
                     title_fontsize=8)
    leg1.get_title().set_color("white")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=patches_aresta, loc="lower left", fontsize=7,
                     title="Relacao", facecolor="#1a1a1a", labelcolor="white",
                     title_fontsize=8)
    leg2.get_title().set_color("white")

    ax.set_title(titulo, color="white", fontsize=13, pad=12)
    ax.axis("off")

    destino = salvar_em or DATA_DIR / "rede_semantica.png"
    plt.tight_layout()
    plt.savefig(destino, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info("Visualizacao salva em: %s", destino)


def exportar_graphml(G: nx.MultiDiGraph, destino: Optional[Path] = None) -> Path:
    """Exporta o grafo em GraphML para importacao no Neo4j / Gephi."""
    destino = destino or DATA_DIR / "rede_semantica.graphml"
    # GraphML nao suporta dict como atributo de no; converte para str se necessario
    G_export = G.copy()
    for n, d in G_export.nodes(data=True):
        for k, v in list(d.items()):
            if not isinstance(v, (str, int, float, bool)):
                G_export.nodes[n][k] = str(v)
    for u, v, k, d in G_export.edges(data=True, keys=True):
        for attr, val in list(d.items()):
            if not isinstance(val, (str, int, float, bool)):
                G_export[u][v][k][attr] = str(val)
    nx.write_graphml(G_export, destino)
    log.info("Grafo exportado em GraphML: %s", destino)
    return destino


# ---------------------------------------------------------------------------
# Vista SKOS / RDF (sec. 3.5)
# ---------------------------------------------------------------------------

def exportar_skos(G: nx.MultiDiGraph, destino: Optional[Path] = None) -> Optional[Path]:
    """
    Exporta a vista SKOS do lexico, conforme a recomendacao da sec. 3.5:
    "comece em property graph e exporte uma vista SKOS quando precisar
    integrar formalmente com a ontologia".

    Mapeamento adotado:
        Conceito              -> skos:Concept com skos:prefLabel
        Termo (abrev/variante)-> skos:altLabel do Conceito canonico
        hiperonimoDe          -> skos:broader / skos:narrower
        sinonimoDe            -> skos:closeMatch entre Conceitos
        relacionadoA          -> skos:related
        mapeiaParaPDM         -> skos:broadMatch para o no PDM
        PDM                   -> skos:Concept no esquema de classes

    O peso de cada aresta vai como anotacao, para nao se perder na traducao.
    """
    try:
        from rdflib import Graph as RDFGraph, Literal, Namespace, URIRef
        from rdflib.namespace import SKOS, RDF, DCTERMS, XSD
    except ImportError:
        log.warning("rdflib ausente — vista SKOS nao exportada.")
        return None

    destino = destino or DATA_DIR / "rede_semantica_skos.ttl"
    MMH = Namespace(f"{IRI_BASE}#")

    g = RDFGraph()
    g.bind("skos", SKOS)
    g.bind("mmh", MMH)
    g.bind("dcterms", DCTERMS)

    esquema = URIRef(f"{IRI_BASE}#esquema")
    g.add((esquema, RDF.type, SKOS.ConceptScheme))
    g.add((esquema, SKOS.prefLabel, Literal("Lexico de dominio CATMAT <-> e-Fisco", lang="pt")))
    g.add((esquema, DCTERMS.hasVersion, Literal(VERSAO_LEXICO)))
    g.add((esquema, DCTERMS.date, Literal(HOJE, datatype=XSD.date)))

    def _uri(nid: str) -> URIRef:
        # ID do no ja e seguro para IRI (TIPO::TEXTO_NORMALIZADO).
        return URIRef(f"{IRI_BASE}#{nid.replace('::', '.')}")

    # Conceitos e PDMs viram skos:Concept.
    for nid, dados in G.nodes(data=True):
        tipo = dados.get("tipo")
        if tipo not in ("Conceito", "PDM", "ValorDeAtributo", "Unidade"):
            continue
        uri = _uri(nid)
        g.add((uri, RDF.type, SKOS.Concept))
        g.add((uri, SKOS.inScheme, esquema))
        g.add((uri, SKOS.prefLabel, Literal(dados.get("label", ""), lang="pt")))
        g.add((uri, MMH.tipoNo, Literal(tipo)))
        if tipo == "PDM":
            g.add((uri, SKOS.topConceptOf, esquema))
            g.add((esquema, SKOS.hasTopConcept, uri))

    _MAPA_SKOS = {
        "hiperonimoDe":  SKOS.broader,
        "sinonimoDe":    SKOS.closeMatch,
        "relacionadoA":  SKOS.related,
        "mapeiaParaPDM": SKOS.broadMatch,
    }

    n_alt = n_rel = 0
    for origem, destino_no, dados in G.edges(data=True):
        relacao = dados.get("relacao", "")
        tipo_origem = G.nodes[origem].get("tipo")

        # Termo -> Conceito vira altLabel no Conceito (nao um recurso proprio).
        if relacao in _RELACOES_CANONICAS and tipo_origem == "Termo":
            g.add((_uri(destino_no), SKOS.altLabel,
                   Literal(G.nodes[origem].get("label", ""), lang="pt")))
            n_alt += 1
            continue

        pred = _MAPA_SKOS.get(relacao)
        if pred is None:
            continue
        u_o, u_d = _uri(origem), _uri(destino_no)
        g.add((u_o, pred, u_d))
        if pred == SKOS.broader:
            g.add((u_d, SKOS.narrower, u_o))
        n_rel += 1

    g.serialize(destination=str(destino), format="turtle")
    log.info("Vista SKOS exportada: %s (%d triplas | %d altLabel | %d relacoes)",
             destino, len(g), n_alt, n_rel)
    return destino


# ---------------------------------------------------------------------------
# Execucao direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Constroi o grafo completo
    G = construir_grafo("principal")

    # Visualizacao 1: vizinhanca de um PDM especifico
    visualizar_grafo(
        G,
        pdm_foco="AGULHA PUNCAO OSSEA",
        titulo="Vizinhanca: AGULHA PUNCAO OSSEA",
        salvar_em=DATA_DIR / "rede_semantica_agulha.png",
    )

    # Visualizacao 2: top nos do grafo completo
    visualizar_grafo(
        G,
        pdm_foco=None,
        max_nos=80,
        titulo="Rede Semantica CATMAT <-> e-Fisco (top 80 nos por grau)",
        salvar_em=DATA_DIR / "rede_semantica_geral.png",
    )

    # Exporta GraphML para Neo4j/Gephi
    exportar_graphml(G)

    # Demonstracao do servico de consulta (sec. 3.7)
    print("\n=== Servico de consulta (sec. 3.7) ===")
    exemplos = ["INOX", "QUICKLE", "C/", "UND", "ESTERIL"]
    for t in exemplos:
        print(f"  normalizar('{t}') -> '{normalizar_termo(G, t)}'")

    print("\n  expansao por spreading activation a partir de ['AGULHA PUNCAO OSSEA']:")
    ativ = expandir_por_ativacao(G, ["AGULHA PUNCAO OSSEA"], limiar=0.1)
    for label, score in sorted(ativ.items(), key=lambda x: -x[1])[:8]:
        print(f"    {label}: {score:.3f}")
