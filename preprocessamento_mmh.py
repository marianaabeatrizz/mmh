"""
preprocessamento_mmh.py
Pipeline de pre-processamento para dados MMH com integracao opcional da rede semantica.

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
    5p. Ancoragem PDM                 -- liga cada item ao seu PDM/Classe no catalogo CATMAT
    5v. Documento virtual expandido   -- forma canonica + termos ativados (entrada para embeddings)

Colunas adicionadas ao DataFrame:
    Sem grafo: efisco_processado | catmat_processado | catmat_atributos
    Com grafo: + efisco_normalizado | doc_virtual_expandido | pdm_ancoragem
"""

import re
import unicodedata
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurações e constantes
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
DADOS_DIR = DATA_DIR / "dados"

# Arquivos disponíveis na pasta MMH
ARQUIVOS = {
    "principal":  DADOS_DIR / "20260408_ground_truth_mmh_limpa.csv",
    "test":       DADOS_DIR / "20260408_ground_truth_mmh_test.csv",
    "opme_test":  DADOS_DIR / "20260408_ground_truth_mmh_opme_test.csv",
    "legado":     DADOS_DIR / "20260405_ground_truth_mmh_limpa.csv",
}

# Padrões de boilerplate jurídico recorrentes nos textos eFisco
_BOILERPLATE = [
    r"O PRODUTO DEVERA(?: ESTAR DE ACORDO COM| OBEDECER A)[^,\.]*",
    r"COMBINADO COM O? ?ART(?:IGO)?\.? ?\d+[^,\.]*",
    r"COMB\.? ?C/? ?O? ?ART(?:IGO)?\.? ?\d+[^,\.]*",
    r"ART(?:IGO)?\.? ?\d+ (?:DA )?L(?:EI)?\.? ?[\d\.]+/\d+[^,\.]*",
    r"PORT(?:ARIA)?\.? ?CONJ\.? ?N\.? ?\d+[^,\.]*",
    r"DECRETO[- ]LEI ?\d+/\d+[^,\.]*",
    r"ROTULAGEM RESPEITANDO[^,\.]*",
    r"APRESENTACAO CONFORME[^,\.]*",
    r"CONTENDO DADOS DE (?:IDENTIFICACAO|PROCEDENCIA|VALIDADE)[^,\.]*",
    r"(?:REG(?:ISTRO)?|REG\.?) (?:NO |N\.? ?)?M\.?S\.?(?:/?ANVISA)?",
    r"EMBALADO EM COMBINADO COM[^,\.]*",
    r"A APRESENTACAO DO PRODUTO[^,\.]*",
    r"PADROES CONFORME LEGISLACAO[^,\.]*",
]
_RE_BOILERPLATE = re.compile("|".join(_BOILERPLATE), re.IGNORECASE)

# Stopwords português + domínio (boilerplate residual após limpeza)
STOPWORDS = {
    "a", "ao", "aos", "as", "com", "da", "das", "de", "do", "dos",
    "e", "em", "na", "nas", "no", "nos", "o", "os", "ou", "para",
    "por", "se", "um", "uma", "uns", "umas",
    # Resíduos de domínio
    "produto", "devera", "obedecer", "estar", "acordo", "combinado",
    "contendo", "dados", "conforme", "apresentacao", "embalado",
    "embalagem", "individual", "rotulagem", "respeitando",
}

# Mapa de sinônimos/variantes ortográficas frequentes nos dados eFisco
SINONIMOS = {
    # Modelos de agulha
    r"\bQUICKLE\b":   "QUINCKE",
    r"\bQUINKER\b":   "QUINCKE",
    r"\bTOUHY\b":     "TUOHY",
    # Gauge / calibre
    r"\b(\d+)\s*GA\b": r"\1G",       # "18GA" -> "18G"
    r"\bGAUGE\b":     "G",
    # Conectores
    r"\bLUER-LOCK\b": "LUER LOCK",
    r"\bLUER LOCK\b": "LUER LOCK",   # já correto, mas garante espaço
    # Materiais
    r"\bACO INOX\b":  "ACO INOXIDAVEL",
    r"\bINOX\b":      "ACO INOXIDAVEL",
    # Esterilidade
    r"\bESTERILIZADO\b": "ESTERIL",
    # Dimensões – normaliza espaçamento em "18G X 3"
    r"\bX(?=\s*\d)":  "X",
}

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def carregar_dados(
    arquivo: str = "principal",
    caminho_personalizado: Optional[Path] = None,
) -> pd.DataFrame:
    """Carrega um CSV MMH (separador pipe) e valida as colunas mínimas."""
    path = caminho_personalizado or ARQUIVOS.get(arquivo)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    colunas_obrigatorias = {"item_efisco", "item_catmat"}
    faltando = colunas_obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")

    # Preenche NaN com string vazia para não quebrar os pipelines de texto
    df["item_efisco"] = df["item_efisco"].fillna("")
    df["item_catmat"] = df["item_catmat"].fillna("")

    log.info("Carregado: %s  (%d registros)", path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# Etapa 2 – Remoção de boilerplate jurídico (eFisco)
# ---------------------------------------------------------------------------

def remover_boilerplate(texto: str) -> str:
    """Remove cláusulas legais e frases padronizadas do texto eFisco."""
    texto = _RE_BOILERPLATE.sub(" ", texto)
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
    """Substitui variantes ortográficas por formas canônicas."""
    for padrao, substituto in SINONIMOS.items():
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
    return [t for t in tokens if t.lower() not in STOPWORDS]


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
    """Salva o dataset processado em CSV (separador pipe)."""
    if destino is None:
        destino = DATA_DIR / "mmh_preprocessado.csv"
    df.to_csv(destino, sep="|", index=False, encoding="utf-8-sig")
    log.info("Dataset salvo em: %s", destino)
    return destino


# ---------------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_raw = carregar_dados("principal")

    df_proc = preprocessar_dataset(df_raw)

    # Amostra de verificação
    amostra = df_proc[["item_efisco", "efisco_processado", "catmat_processado"]].head(3)
    print("\n=== Amostra pré-processamento ===")
    for _, row in amostra.iterrows():
        print(f"\neFisco original : {row['item_efisco'][:120]}...")
        print(f"eFisco limpo    : {row['efisco_processado'][:120]}")
        print(f"CATMAT limpo    : {row['catmat_processado'][:120]}")

    salvar_dataset(df_proc)
