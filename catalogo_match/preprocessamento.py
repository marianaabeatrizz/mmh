"""
preprocessamento.py
Pipeline de pre-processamento de texto de catalogo, com integracao opcional da
rede semantica.

Etapas do pipeline (sem grafo):
    1. Ingestao e validacao dos CSVs
    2. Remocao de boilerplate juridico (campo item_efisco)
    3. Normalizacao textual (caixa, acentos, pontuacao)
    4. Extracao de atributos estruturados (campo item_catmat)
    5. Normalizacao de vocabulario (sinonimos e variantes ortograficas)
    6. Tokenizacao e remocao de stopwords
    7. Serializacao do dataset processado

Etapas adicionais com grafo (sec. 3.7 de rede_semantica_catmat):
    5g. Normalizacao via grafo        -- lookup token->forma canonica (abreviacaoDe/varianteOrtograficaDe)
    5e. Expansao por spreading        -- propaga ativacao para sinonimos/hiperonimos relevantes
    5p. Ancoragem PDM                 -- liga cada item ao seu PDM/Classe no catalogo
    5v. Documento virtual expandido   -- forma canonica + termos ativados (entrada para embeddings)

Colunas adicionadas ao DataFrame:
    Sem grafo: efisco_processado | catmat_processado | catmat_atributos
    Com grafo: + efisco_normalizado | doc_virtual_expandido | pdm_ancoragem

O que era CONSTANTE de dominio aqui (boilerplate juridico, stopwords, sinonimos)
agora vem do perfil ativo (`config.perfil_ativo()`), carregado de
config/perfis/<dominio>.yaml. As funcoes de normalizacao puramente textuais
-- `normalizar_texto`, `tokenizar` -- nao dependem de dominio e ficaram aqui.
"""

import re
import unicodedata
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

from .config import COLUNAS_OBRIGATORIAS, contexto, perfil_ativo

if TYPE_CHECKING:
    import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dominio: nada de constante aqui
# ---------------------------------------------------------------------------
# O boilerplate juridico, as stopwords e o mapa de sinonimos eram tres blocos
# literais neste arquivo. Sairam para config/perfis/<dominio>.yaml: o que o
# pipeline sabe sobre agulhas deixou de ser codigo.

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def carregar_dados(
    arquivo: str = "",
    caminho_personalizado: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Carrega um CSV do dataset ativo e valida as colunas minimas.

    `arquivo` e a chave logica do dataset ("principal", "test", ...) ou um nome
    de arquivo; o dataset ativo resolve o caminho, o separador e o encoding.
    Ao ler, o contexto completa o perfil com o que der para induzir do corpus
    (so tem efeito quando o perfil nao traz o bloco -- ver config.induzir_perfil).
    """
    ctx = contexto()

    if caminho_personalizado is not None:
        path = Path(caminho_personalizado)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        df = pd.read_csv(path, sep=ctx.dataset.separador, dtype=str,
                         encoding=ctx.dataset.encoding)
        df.columns = df.columns.str.strip()
        faltando = set(COLUNAS_OBRIGATORIAS) - set(df.columns)
        if faltando:
            raise ValueError(f"Colunas ausentes no CSV: {faltando}")
    else:
        path = ctx.dataset.caminho(arquivo)
        df = ctx.dataset.ler(arquivo)

    # Preenche NaN com string vazia para não quebrar os pipelines de texto
    for coluna in COLUNAS_OBRIGATORIAS:
        df[coluna] = df[coluna].fillna("")

    ctx.completar_com_dados(df)

    log.info("Carregado: %s  (%d registros)", path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# Etapa 2 – Remoção de boilerplate jurídico (eFisco)
# ---------------------------------------------------------------------------

def remover_boilerplate(texto: str) -> str:
    """
    Remove cláusulas legais e frases padronizadas do texto e-Fisco.

    Os padrões vêm do perfil (`texto.boilerplate`). Perfil sem boilerplate
    declarado nem induzido devolve o texto intacto — que é o comportamento
    correto: é melhor deixar ruído do que apagar especificação por engano.
    """
    padrao = perfil_ativo().re_boilerplate
    if padrao is not None:
        texto = padrao.sub(" ", texto)
    # Remove referências residuais do tipo "L.8078/90"
    texto = re.sub(r"\bL\.?\d+/\d+\b", " ", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()


# ---------------------------------------------------------------------------
# Etapa 3 – Normalização textual
# ---------------------------------------------------------------------------

def normalizar_texto(texto: str, manter_maiusculas: bool = False) -> str:
    """
    Normalização base:
      - Remove acentos (NFD → ASCII)
      - Converte para maiúsculas (padrão do corpus) ou minúsculas
      - Remove caracteres não alfanuméricos, exceto espaço, vírgula e hífen
      - Colapsa espaços múltiplos
    """
    # Normalização de acentos
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")

    texto = texto.upper() if manter_maiusculas else texto.lower()

    # Remove aspas (simples, duplas, curly) e outros ruídos
    texto = texto.translate(str.maketrans("", "", chr(0x22)+chr(0x27)+chr(0x201c)+chr(0x201d)+chr(0x2018)+chr(0x2019)))
    # Mantém apenas alfanuméricos, espaço, vírgula, ponto, hífen e parênteses
    texto = re.sub(r"[^\w\s,\.\-\(\)/]", " ", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()


# ---------------------------------------------------------------------------
# Etapa 4 – Extração de atributos CATMAT
# ---------------------------------------------------------------------------

def extrair_atributos_catmat(texto: str) -> dict[str, str]:
    """
    Parseia o formato estruturado CATMAT no texto original (antes de normalizar),
    preservando o separador ":" entre chave e valor.

    Formato: "TIPO_PRODUTO, ATRIBUTO_1: VALOR_1 , ATRIBUTO_2: VALOR_2 ,"
    Retorna dict com tipo_produto, atributos individuais e texto_completo.
    """
    atributos: dict[str, str] = {}
    if not texto.strip():
        return atributos

    # Divide no texto bruto para preservar os ":" dos pares chave:valor
    segmentos = [s.strip() for s in texto.split(",") if s.strip()]
    if not segmentos:
        return atributos

    # Primeiro segmento sem ":" e o nome/tipo do produto
    if ":" not in segmentos[0]:
        atributos["tipo_produto"] = normalizar_texto(segmentos[0], manter_maiusculas=True)
        segmentos = segmentos[1:]

    ultimo_campo = None
    for seg in segmentos:
        if ":" in seg:
            chave, _, valor = seg.partition(":")
            chave_norm = normalizar_texto(chave, manter_maiusculas=False)
            chave_slug = re.sub(r"\s+", "_", chave_norm.strip())
            atributos[chave_slug] = normalizar_texto(valor, manter_maiusculas=True)
            ultimo_campo = chave_slug
        elif ultimo_campo:
            # Segmento sem ":" e continuacao do valor anterior (ex: "AJUSTAVEL" apos dimensoes)
            atributos[ultimo_campo] += " " + normalizar_texto(seg, manter_maiusculas=True)

    atributos["texto_completo"] = normalizar_texto(texto, manter_maiusculas=True)
    return atributos
def atributos_catmat_para_texto(atributos: dict[str, str]) -> str:
    """Reconstrói texto linear a partir dos atributos extraídos (sem o tipo_produto)."""
    partes = []
    if "tipo_produto" in atributos:
        partes.append(atributos["tipo_produto"])
    for chave, valor in atributos.items():
        if chave not in ("tipo_produto", "texto_completo"):
            partes.append(valor)
    return " ".join(partes)


# ---------------------------------------------------------------------------
# Etapa 5 – Normalização de vocabulário (sinônimos)
# ---------------------------------------------------------------------------

def aplicar_sinonimos(texto: str) -> str:
    """Substitui variantes ortograficas por formas canonicas (perfil ativo)."""
    for padrao, substituto in perfil_ativo().sinonimos_regex.items():
        texto = re.sub(padrao, substituto, texto, flags=re.IGNORECASE)
    return texto


# ---------------------------------------------------------------------------
# Etapa 6 – Tokenização e remoção de stopwords
# ---------------------------------------------------------------------------

def tokenizar(texto: str) -> list[str]:
    """Tokenização simples por espaço após limpeza de pontuação residual."""
    texto = re.sub(r"[,\.\-\(\)/]", " ", texto)
    return [t for t in texto.split() if t]


def remover_stopwords(tokens: list[str]) -> list[str]:
    stopwords = perfil_ativo().stopwords
    return [t for t in tokens if t.lower() not in stopwords]


def preprocessar_texto_efisco(texto: str) -> str:
    """Pipeline completo para um texto eFisco → string limpa."""
    texto = remover_boilerplate(texto)
    texto = aplicar_sinonimos(texto)
    # manter_maiusculas=True: preserva o caixa-alta padrão do corpus, para
    # ficar consistente com catmat_processado (senão TF-IDF/embeddings veriam
    # 'agulha' vs 'AGULHA' como formas distintas).
    texto = normalizar_texto(texto, manter_maiusculas=True)
    tokens = tokenizar(texto)
    tokens = remover_stopwords(tokens)
    return " ".join(tokens)


def preprocessar_texto_catmat(texto: str) -> str:
    """Pipeline completo para um texto CATMAT → string limpa (via atributos)."""
    atributos = extrair_atributos_catmat(texto)
    texto_reconstruido = atributos_catmat_para_texto(atributos)
    texto_reconstruido = aplicar_sinonimos(texto_reconstruido)
    tokens = tokenizar(texto_reconstruido)
    tokens = remover_stopwords(tokens)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Etapa 7 – Pipeline principal sobre o DataFrame
# ---------------------------------------------------------------------------

def preprocessar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o pipeline completo ao DataFrame e adiciona as colunas:
      - efisco_processado   : texto eFisco limpo
      - catmat_processado   : texto CATMAT limpo
      - catmat_atributos    : dict com atributos extraídos do CATMAT
    """
    log.info("Iniciando pré-processamento de %d registros...", len(df))

    df = df.copy()

    df["efisco_processado"] = df["item_efisco"].apply(preprocessar_texto_efisco)
    log.info("✓ item_efisco processado")

    df["catmat_processado"] = df["item_catmat"].apply(preprocessar_texto_catmat)
    log.info("✓ item_catmat processado")

    df["catmat_atributos"] = df["item_catmat"].apply(extrair_atributos_catmat)
    log.info("✓ atributos CATMAT extraídos")

    return df


def salvar_dataset(df: pd.DataFrame, destino: Optional[Path] = None) -> Path:
    """Salva o dataset processado em CSV, na pasta de resultados do dataset."""
    ds = contexto().dataset
    if destino is None:
        ds.dir_resultados.mkdir(parents=True, exist_ok=True)
        destino = ds.dir_resultados / f"{ds.nome}_preprocessado.csv"
    df.to_csv(destino, sep=ds.separador, index=False, encoding=ds.encoding)
    log.info("Dataset salvo em: %s", destino)
    return destino


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    from .config import ativar, listar_datasets

    ap = argparse.ArgumentParser(
        description="Pre-processa um dataset e grava o CSV processado.")
    ap.add_argument("--dataset", default="",
                    help=f"dataset a processar. Disponiveis: {', '.join(listar_datasets())}")
    ap.add_argument("--arquivo", default="",
                    help="chave logica do arquivo no dataset (padrao: o do YAML)")
    ap.add_argument("--perfil", default="", help="forca outro perfil de dominio")
    args = ap.parse_args()

    ativar(args.dataset, perfil=args.perfil)

    df_raw = carregar_dados(args.arquivo)
    df_proc = preprocessar_dataset(df_raw)

    amostra = df_proc[["item_efisco", "efisco_processado", "catmat_processado"]].head(3)
    print("\n=== Amostra pré-processamento ===")
    for _, row in amostra.iterrows():
        print(f"\neFisco original : {row['item_efisco'][:120]}...")
        print(f"eFisco limpo    : {row['efisco_processado'][:120]}")
        print(f"CATMAT limpo    : {row['catmat_processado'][:120]}")

    salvar_dataset(df_proc)


if __name__ == "__main__":
    _cli()
