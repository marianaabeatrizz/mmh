"""
config.py — o ponto único onde o pipeline sabe QUAL dataset e QUAL domínio.

Antes desta camada, o conhecimento de domínio estava embutido em seis módulos:
as regex de boilerplate jurídico no pré-processamento, as abreviações e a tabela
de unidades na rede semântica, as características definidoras na ontologia, a
lista de famílias no blocking, o esquema de atributos nos prompts da LLM. Trocar
de dataset significava editar código em seis lugares — e, na prática, significava
não trocar.

Aqui esse conhecimento é DADO, em dois objetos:

    DatasetSpec    — onde estão os arquivos, como lê-los, onde escrever a saída.
                     Um YAML por dataset em config/datasets/.

    PerfilDominio  — o que o pipeline sabe sobre o domínio: léxico, famílias,
                     unidades, esquema de atributos, papéis nos prompts. Um YAML
                     por domínio em config/perfis/, sobre config/perfis/base.yaml.

Os dois são independentes de propósito: dois datasets do mesmo domínio (o MMH
inteiro e o recorte OPME, por exemplo) compartilham um perfil; um dataset de
domínio desconhecido roda com o perfil `base` e o que der para INDUZIR do próprio
corpus (ver `induzir_perfil`). O que não dá para induzir com honestidade —
cláusula legal, fator de conversão de unidade — fica de fora e é registrado como
lacuna, em vez de ser inventado.

Uso:

    from catalogo_match.config import contexto, ativar

    ativar("mmh")                  # a CLI faz isso antes de qualquer leitura
    ctx = contexto()
    ctx.dataset.caminho("principal")
    ctx.perfil.familias
"""

from __future__ import annotations

import os
import re
import copy
import logging
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

log = logging.getLogger(__name__)

PACOTE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACOTE_DIR.parent
CONFIG_DIR = BASE_DIR / "config"
DATASETS_DIR = CONFIG_DIR / "datasets"
PERFIS_DIR = CONFIG_DIR / "perfis"


def carregar_dotenv(nome: str = ".env") -> None:
    """
    Carrega variáveis de um `.env` (raiz do repositório ou diretório atual) para
    `os.environ`, sem sobrescrever o que já está no ambiente e sem depender de
    python-dotenv. Chamado no import deste módulo — o ponto por onde todo módulo
    do pacote passa —, para que a grade modular, a varredura e o serviço vejam
    a mesma chave que o pipeline. (Antes só o pipeline a carregava, e a grade com
    `--graphrag-ms` gravava uma chave vazia no workspace do índice.)
    """
    vistos = set()
    for p in (BASE_DIR / nome, Path.cwd() / nome):
        p = p.resolve()
        if p in vistos or not p.exists():
            continue
        vistos.add(p)
        try:
            for linha in p.read_text(encoding="utf-8").splitlines():
                linha = linha.strip()
                if not linha or linha.startswith("#") or "=" not in linha:
                    continue
                chave, _, valor = linha.partition("=")
                chave = chave.strip()
                valor = valor.strip().strip('"').strip("'")
                if chave and valor:
                    os.environ.setdefault(chave, valor)
        except OSError as exc:
            log.warning(".env ilegível (%s): %s", p, exc)


carregar_dotenv()

DATASET_PADRAO = os.environ.get("CM_DATASET", "mmh")

# ---------------------------------------------------------------------------
# Esquema interno
# ---------------------------------------------------------------------------
# Todos os datasets deste projeto são mapeamentos e-Fisco -> CATMAT, então os
# nomes de coluna não são conhecimento de domínio: são o contrato interno que o
# resto do código assume. Um dataset cujo CSV use outros nomes declara o
# mapeamento em `colunas:` e o carregador renomeia na entrada — o código a
# jusante continua vendo estes nomes, e só estes.

COLUNAS_CONSULTA = ("codigo_efisco", "item_efisco", "classe_efisco")
COLUNAS_CATALOGO = ("codigo_catmat", "item_catmat", "classe_catmat")
COLUNAS_OBRIGATORIAS = ("item_efisco", "item_catmat")


def sanitizar(nome: str) -> str:
    """Rótulo livre -> identificador em caixa alta, sem acento nem pontuação."""
    txt = unicodedata.normalize("NFD", str(nome))
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return re.sub(r"[^A-Za-z0-9]+", "_", txt.upper()).strip("_")


def _fundir(base: dict, cima: dict) -> dict:
    """Merge recursivo: `cima` vence, dicionário a dicionário."""
    saida = copy.deepcopy(base)
    for chave, valor in (cima or {}).items():
        if isinstance(valor, dict) and isinstance(saida.get(chave), dict):
            saida[chave] = _fundir(saida[chave], valor)
        else:
            saida[chave] = copy.deepcopy(valor)
    return saida


# ---------------------------------------------------------------------------
# Perfil de domínio
# ---------------------------------------------------------------------------

@dataclass
class PerfilDominio:
    """
    Todo o conhecimento de domínio do pipeline, em um objeto.

    Os campos são preenchidos pelo YAML do perfil; os que ficarem vazios podem
    ser completados por `induzir_perfil` a partir do corpus. `proveniencia`
    registra, bloco a bloco, se o que está em uso veio de curadoria ou de
    indução — é o que permite ler um resultado sabendo quanto dele depende de
    conhecimento humano.
    """

    nome: str = "base"
    descricao: str = "itens de catálogo de compras públicas"
    prefixo: str = "dom"
    iri_ontologia: str = ""
    iri_lexico: str = ""
    versao_lexico: str = "1.0"
    llm: dict = field(default_factory=dict)

    boilerplate: tuple[str, ...] = ()
    stopwords: frozenset[str] = frozenset()
    sinonimos_regex: dict[str, str] = field(default_factory=dict)

    abreviacoes: dict[str, str] = field(default_factory=dict)
    variantes: dict[str, str] = field(default_factory=dict)
    sinonimos: dict[str, str] = field(default_factory=dict)
    hierarquia: dict[str, list[str]] = field(default_factory=dict)
    pdm_foco_exemplo: str = ""
    exemplos_demo: tuple[str, ...] = ()

    unidades: dict[str, dict] = field(default_factory=dict)
    familias_extra: tuple[str, ...] = ()
    atributos: dict[str, dict] = field(default_factory=dict)
    atributos_definidores: dict[str, list[str]] = field(default_factory=dict)
    extracao: tuple[dict, ...] = ()
    entidade_stopwords: frozenset[str] = frozenset()
    max_itens_por_entidade: int = 200
    medidas: dict = field(default_factory=dict)

    proveniencia: dict[str, str] = field(default_factory=dict)
    lacunas: tuple[str, ...] = ()

    # -- caches derivados (invalidados por `_invalidar`) --------------------
    _re_boilerplate: Any = field(default=None, repr=False)
    _re_medida: Any = field(default=None, repr=False)
    _re_calibre: Any = field(default=None, repr=False)
    _extracao_compilada: Any = field(default=None, repr=False)
    _familias: Any = field(default=None, repr=False)

    # -----------------------------------------------------------------------
    # Construção
    # -----------------------------------------------------------------------

    BLOCOS = ("boilerplate", "stopwords", "sinonimos_regex", "abreviacoes",
              "variantes", "sinonimos", "hierarquia", "unidades", "atributos",
              "atributos_definidores", "extracao", "entidade_stopwords",
              "familias_extra", "medidas")

    @classmethod
    def de_dict(cls, dados: dict) -> "PerfilDominio":
        texto = dados.get("texto") or {}
        lexico = dados.get("lexico") or {}
        entidades = dados.get("entidades") or {}
        perfil = cls(
            nome=dados.get("nome", "base"),
            descricao=dados.get("descricao") or cls.descricao,
            prefixo=dados.get("prefixo", "dom"),
            iri_ontologia=dados.get("iri_ontologia", ""),
            iri_lexico=dados.get("iri_lexico", ""),
            versao_lexico=str(dados.get("versao_lexico", "1.0")),
            llm=dict(dados.get("llm") or {}),
            boilerplate=tuple(texto.get("boilerplate") or ()),
            stopwords=frozenset(texto.get("stopwords") or ()),
            sinonimos_regex=dict(texto.get("sinonimos_regex") or {}),
            abreviacoes=dict(lexico.get("abreviacoes") or {}),
            variantes=dict(lexico.get("variantes") or {}),
            sinonimos=dict(lexico.get("sinonimos") or {}),
            hierarquia={k: list(v) for k, v in (lexico.get("hierarquia") or {}).items()},
            pdm_foco_exemplo=lexico.get("pdm_foco_exemplo", ""),
            exemplos_demo=tuple(lexico.get("exemplos_demo") or ()),
            unidades={k: dict(v) for k, v in (dados.get("unidades") or {}).items()},
            familias_extra=tuple(dados.get("familias") or ()),
            atributos={k: dict(v or {}) for k, v in (dados.get("atributos") or {}).items()},
            atributos_definidores={k: list(v) for k, v in
                                   (dados.get("atributos_definidores") or {}).items()},
            extracao=tuple(dados.get("extracao") or ()),
            entidade_stopwords=frozenset(entidades.get("stopwords") or ()),
            max_itens_por_entidade=int(entidades.get("max_itens_por_entidade", 200)),
            medidas=dict(dados.get("medidas") or {}),
        )
        for bloco in cls.BLOCOS:
            perfil.proveniencia[bloco] = "curado" if getattr(perfil, bloco) else "vazio"
        return perfil

    def _invalidar(self) -> None:
        self._re_boilerplate = None
        self._re_medida = None
        self._re_calibre = None
        self._extracao_compilada = None
        self._familias = None

    # -----------------------------------------------------------------------
    # Famílias
    # -----------------------------------------------------------------------

    @property
    def familias(self) -> tuple[str, ...]:
        """
        Famílias para o blocking por subsunção (fase [3]).

        União da lista `familias:` do perfil com as chaves da hierarquia curada.

        NÃO inclui as chaves de `atributos_definidores`, embora seja tentador:
        as execuções publicadas em RESULTADOS.md rodaram com a lista explícita, e
        três famílias que têm definidoras (ESPARADRAPO, FRASCO, BOLSA) nunca
        estiveram no blocking. Unir os dois conjuntos alargaria o bloco e mexeria
        nos números sem que ninguém tivesse pedido. O perfil deixa a escolha
        visível: para incluí-las, basta acrescentá-las em `familias:`.
        """
        if self._familias is None:
            todas = set(self.hierarquia) | set(self.familias_extra)
            self._familias = tuple(sorted(f for f in todas if f))
        return self._familias

    def familia_de(self, pdm: str) -> str:
        """
        Família (chave de `atributos_definidores`) a que um PDM pertence.

        Casamento por contenção no rótulo sanitizado, na ordem de declaração do
        perfil — a primeira que casar vence, então o perfil deve declarar a
        família mais específica antes da mais genérica.
        """
        pdm_up = sanitizar(pdm)
        for familia in self.atributos_definidores:
            if familia in pdm_up:
                return familia
        return ""

    # -----------------------------------------------------------------------
    # Texto
    # -----------------------------------------------------------------------

    @property
    def re_boilerplate(self):
        """Regex única com todos os padrões de boilerplate, ou None se não há."""
        if self._re_boilerplate is None and self.boilerplate:
            self._re_boilerplate = re.compile("|".join(self.boilerplate), re.IGNORECASE)
        return self._re_boilerplate

    # -----------------------------------------------------------------------
    # Atributos: esquema, prompt, extração
    # -----------------------------------------------------------------------

    @property
    def chaves_atributos(self) -> tuple[str, ...]:
        """Chaves do esquema, na ordem de declaração (a ordem vai para o prompt)."""
        return tuple(self.atributos)

    def bloco_prompt_atributos(self) -> str:
        """As linhas de esquema que entram no prompt de extração da fase [2]."""
        linhas = []
        for chave, spec in self.atributos.items():
            desc = (spec or {}).get("descricao", "").strip()
            linhas.append(f'  "{chave}"' + (f": {desc}" if desc else ""))
        return "\n".join(linhas)

    def canonicalizar(self, bruto: dict) -> dict:
        """
        Normaliza o JSON da LLM para o mesmo formato do extrator regex, de modo
        que a ontologia compare chaves e valores alinhados dos dois lados.

        As regras por atributo vêm do perfil (`tipo`, `sufixo`), e não de um `if`
        por nome de campo: é o que permite a um domínio novo ter `principio_ativo`
        e `concentracao` sem tocar neste arquivo.
        """
        from .preprocessamento import normalizar_texto

        if not isinstance(bruto, dict):
            return {}
        atribs: dict[str, str] = {}
        for chave in self.chaves_atributos:
            val = bruto.get(chave)
            if val is None:
                continue
            val = normalizar_texto(str(val), manter_maiusculas=True).strip()
            if not val:
                continue
            spec = self.atributos.get(chave) or {}
            tipo = spec.get("tipo", "texto")
            if tipo == "numero_sufixado":
                m = re.search(r"(\d[\d,\.]*)", val)
                if not m:
                    continue
                val = m.group(1) + str(spec.get("sufixo", ""))
            elif tipo in ("unidade", "numero"):
                val = val.upper()
            atribs[chave] = val
        return atribs

    @property
    def extracao_compilada(self) -> tuple:
        """Regras de extração com a regex já compilada."""
        if self._extracao_compilada is None:
            regras = []
            for regra in self.extracao:
                try:
                    padrao = re.compile(regra["regex"], re.IGNORECASE)
                except (KeyError, re.error) as exc:
                    log.warning("Regra de extração ignorada (%s): %s", exc, regra)
                    continue
                regras.append((padrao, regra))
            self._extracao_compilada = tuple(regras)
        return self._extracao_compilada

    def extrair_regex(self, texto: str) -> dict:
        """
        Fallback determinístico da camada de percepção (sem LLM/chave/API).

        Cada regra do perfil diz quais atributos preenche e de onde: um grupo de
        captura (`{"calibre": 1}`), o casamento inteiro (`0`) ou um valor fixo
        quando a presença do padrão já é a informação (`valor_fixo`).
        """
        from .preprocessamento import normalizar_texto

        atribs: dict[str, str] = {}
        t = normalizar_texto(texto, manter_maiusculas=True)
        for padrao, regra in self.extracao_compilada:
            m = padrao.search(t)
            if not m:
                continue
            fixo = regra.get("valor_fixo")
            for chave, grupo in (regra.get("atributos") or {}).items():
                if fixo is not None:
                    atribs[chave] = fixo
                    continue
                try:
                    valor = m.group(grupo)
                except (IndexError, re.error):
                    continue
                if not valor:
                    continue
                valor = valor.strip()
                if regra.get("sufixo"):
                    valor = valor + str(regra["sufixo"])
                if regra.get("normalizar"):
                    valor = normalizar_texto(valor, manter_maiusculas=True)
                elif regra.get("caixa_alta", True):
                    valor = valor.upper()
                atribs[chave] = valor
        return atribs

    # -----------------------------------------------------------------------
    # Medidas (pós-processador de unidades)
    # -----------------------------------------------------------------------

    def medidas_de(self, texto: str) -> set:
        """Conjunto de pares (valor, unidade) normalizados presentes no texto."""
        if self._re_medida is None:
            padrao = self.medidas.get("regex")
            if not padrao:
                unidades = sorted(self.unidades, key=len, reverse=True) or ["MM", "CM", "ML", "G"]
                padrao = (r"(\d+(?:[\.,]\d+)?)\s*("
                          + "|".join(re.escape(u) for u in unidades) + r")\b")
            self._re_medida = re.compile(padrao, re.IGNORECASE)
            calibre = self.medidas.get("regex_calibre")
            self._re_calibre = re.compile(calibre, re.IGNORECASE) if calibre else False
        achados = {
            (v.replace(",", "."), u.upper()) for v, u in self._re_medida.findall(texto or "")
        }
        if self._re_calibre:
            rotulo = self.medidas.get("rotulo_calibre", "CALIBRE")
            achados |= {(v, rotulo) for v in self._re_calibre.findall(texto or "")}
        return achados

    # -----------------------------------------------------------------------
    # Papéis nos prompts
    # -----------------------------------------------------------------------

    def papel(self, qual: str) -> str:
        """
        Frase de papel para um prompt de LLM, com queda para texto neutro.

        Prompt é conteúdo de domínio: "auditor de compras públicas hospitalares"
        não serve para um catálogo de medicamentos. O perfil traz a frase; sem
        perfil, ela é montada a partir do `descricao` do domínio.
        """
        pronto = (self.llm or {}).get(qual)
        if pronto:
            return pronto
        padroes = {
            "especialista": f"Especialista em {self.descricao}.",
            "auditor": f"Voce e um auditor de {self.descricao}.",
            "auditor_curto": "um auditor de compras públicas",
            "extrator": ("Você é um motor de extração de atributos para itens de "
                         f"{self.descricao}."),
        }
        return padroes.get(qual, "")

    # -----------------------------------------------------------------------

    def resumo(self) -> dict:
        """Proveniência do conhecimento em uso — vai para o YAML de resultados."""
        return {
            "NOME": self.nome,
            "DESCRICAO": self.descricao,
            "TAMANHOS": {
                "boilerplate": len(self.boilerplate),
                "stopwords": len(self.stopwords),
                "abreviacoes": len(self.abreviacoes),
                "variantes": len(self.variantes),
                "sinonimos": len(self.sinonimos),
                "hierarquia": sum(len(v) for v in self.hierarquia.values()),
                "unidades": len(self.unidades),
                "familias": len(self.familias),
                "atributos": len(self.atributos),
                "atributos_definidores": len(self.atributos_definidores),
                "regras_extracao": len(self.extracao),
            },
            "PROVENIENCIA": dict(self.proveniencia),
            "LACUNAS": list(self.lacunas),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Onde estão os dados de um dataset e para onde vão os seus resultados."""

    nome: str
    descricao: str = ""
    perfil: str = "base"
    arquivos: dict[str, str] = field(default_factory=dict)
    separador: str = "|"
    encoding: str = "utf-8-sig"
    colunas: dict[str, str] = field(default_factory=dict)
    arquivo_padrao: str = "principal"
    # Subpasta dentro de resultados/<dataset>/, usada para isolar variantes de
    # uma mesma execucao (o `--model` do pipeline compara LLMs lado a lado).
    sub_resultados: str = ""

    @classmethod
    def de_dict(cls, nome: str, dados: dict) -> "DatasetSpec":
        return cls(
            nome=nome,
            descricao=dados.get("descricao", ""),
            perfil=dados.get("perfil", "base"),
            arquivos=dict(dados.get("arquivos") or {}),
            separador=dados.get("separador", "|"),
            encoding=dados.get("encoding", "utf-8-sig"),
            colunas=dict(dados.get("colunas") or {}),
            arquivo_padrao=dados.get("arquivo_padrao", "principal"),
        )

    # -- caminhos -----------------------------------------------------------

    def caminho(self, chave: str = "") -> Path:
        """
        Caminho de um dos arquivos do dataset.

        Aceita a chave lógica (`"principal"`, `"test"`), um nome de arquivo solto
        (procurado em `dados/<dataset>/`) ou um caminho já resolvido — para que
        `--arquivo` da CLI continue aceitando o que sempre aceitou.
        """
        chave = chave or self.arquivo_padrao
        alvo = self.arquivos.get(chave, chave)
        p = Path(alvo)
        if p.is_absolute():
            return p
        candidatos = [BASE_DIR / p, self.dir_dados / p, BASE_DIR / "dados" / p]
        for cand in candidatos:
            if cand.exists():
                return cand
        return candidatos[0]

    @property
    def dir_dados(self) -> Path:
        return BASE_DIR / "dados" / self.nome

    @property
    def dir_resultados(self) -> Path:
        base = BASE_DIR / "resultados" / self.nome
        return base / self.sub_resultados if self.sub_resultados else base

    @property
    def dir_cache(self) -> Path:
        return BASE_DIR / "cache" / self.nome

    @property
    def dir_cache_embeddings(self) -> Path:
        return self.dir_cache / "embeddings"

    @property
    def ws_graphrag(self) -> Path:
        return BASE_DIR / "graphrag_workspace" / self.nome

    @property
    def arquivo_curadoria(self) -> Path:
        """Fila de curadoria JÁ revisada — entrada do pipeline, versionada."""
        return self.dir_dados / "fila_curadoria_revisada.csv"

    def preparar_diretorios(self) -> None:
        for d in (self.dir_resultados, self.dir_cache, self.dir_cache_embeddings):
            d.mkdir(parents=True, exist_ok=True)

    # -- leitura ------------------------------------------------------------

    def ler(self, chave: str = ""):
        """
        Lê um CSV do dataset e devolve o DataFrame com as colunas no esquema
        interno (ver COLUNAS_*), aplicando o `colunas:` do YAML quando houver.
        """
        import pandas as pd

        caminho = self.caminho(chave)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        df = pd.read_csv(caminho, sep=self.separador, dtype=str, encoding=self.encoding)
        df.columns = df.columns.str.strip()
        if self.colunas:
            renomear = {origem: destino for destino, origem in self.colunas.items()
                        if origem in df.columns}
            df = df.rename(columns=renomear)
        faltando = set(COLUNAS_OBRIGATORIAS) - set(df.columns)
        if faltando:
            raise ValueError(
                f"Colunas ausentes em {caminho.name}: {sorted(faltando)}. "
                f"Declare o mapeamento em config/datasets/{self.nome}.yaml (`colunas:`)."
            )
        return df


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def listar_datasets() -> list[str]:
    return sorted(p.stem for p in DATASETS_DIR.glob("*.yaml")) if DATASETS_DIR.exists() else []


def listar_perfis() -> list[str]:
    return sorted(p.stem for p in PERFIS_DIR.glob("*.yaml")) if PERFIS_DIR.exists() else []


def carregar_dataset(nome: str) -> DatasetSpec:
    caminho = DATASETS_DIR / f"{nome}.yaml"
    if not caminho.exists():
        disponiveis = ", ".join(listar_datasets()) or "(nenhum)"
        raise SystemExit(
            f"Dataset '{nome}' não encontrado em {DATASETS_DIR}.\n"
            f"Disponíveis: {disponiveis}\n"
            f"Ver docs/NOVO-DATASET.md para criar um."
        )
    dados = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    return DatasetSpec.de_dict(dados.get("nome", nome), dados)


def _ler_perfil_yaml(nome: str, visitados: tuple[str, ...] = ()) -> dict:
    """
    Dicionário de `perfis/<nome>.yaml` já fundido sobre o perfil que ele declara
    em `herda:` (recursivo). `base` é a raiz de todos e não herda de ninguém.

    A herança existe para que um perfil multi-domínio possa DIZER "é o MMH mais
    massa, dose e concentração" em vinte linhas, em vez de copiar o YAML inteiro
    e deixar duas cópias do mesmo léxico divergirem em silêncio.
    """
    if nome in visitados:
        raise ValueError(f"herança circular de perfis: {' -> '.join(visitados + (nome,))}")
    caminho = PERFIS_DIR / f"{nome}.yaml"
    if not caminho.exists():
        log.warning("Perfil '%s' não existe em %s — usando só o perfil base "
                    "(o domínio será induzido do corpus).", nome, PERFIS_DIR)
        return {}
    proprio = yaml.safe_load(caminho.read_text(encoding="utf-8")) or {}
    pai = proprio.pop("herda", None)
    if nome == "base" or not pai or pai == "base":
        return proprio
    return _fundir(_ler_perfil_yaml(str(pai), visitados + (nome,)), proprio)


def carregar_perfil(nome: str) -> PerfilDominio:
    """Carrega `perfis/<nome>.yaml` (e o que ele herda) sobre `perfis/base.yaml`."""
    dados: dict = {}
    base = PERFIS_DIR / "base.yaml"
    if base.exists():
        dados = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
        dados.pop("herda", None)
    if nome and nome != "base":
        dados = _fundir(dados, _ler_perfil_yaml(nome))
    return PerfilDominio.de_dict(dados)


# ---------------------------------------------------------------------------
# Indução: o que fazer quando não há perfil
# ---------------------------------------------------------------------------
# Um dataset de domínio novo chega sem léxico, sem famílias e sem esquema de
# atributos. A alternativa a inventá-los é derivá-los do próprio corpus, que é o
# que estas funções fazem — cada uma preenche UM bloco vazio do perfil e marca a
# proveniência como "induzido".
#
# O que NÃO é induzido, e por quê:
#
#   boilerplate legal  — só entram frases que o corpus repete em >= LIMIAR_FRASE
#                        dos registros. Cláusula que aparece em 10% dos textos
#                        pode ser especificação, e apagá-la destrói sinal.
#   fator de conversão — dizer que 1 CH = 0,33 MM é conhecimento externo. Unidade
#                        induzida entra sem fator: compara por igualdade exata e
#                        não converte.
#   sinônimo/abreviação — a rede semântica já induz do histórico (mineração de
#                        co-ocorrência, §3.6-b) e por embeddings (§3.6-d). Não há
#                        por que duplicar aqui.

LIMIAR_FRASE = 0.40        # frase em >= 40% dos registros é boilerplate
LIMIAR_STOPWORD = 0.50     # token em >= 50% dos registros não discrimina
MIN_ITENS_FAMILIA = 5      # família precisa de 5 itens para ter definidoras
LIMIAR_DEFINIDORA = 0.70   # atributo presente em >= 70% da família é definidor
MAX_DEFINIDORAS = 2        # a regra SWRL fica frágil com mais que isso
MAX_ATRIBUTOS = 20         # teto do esquema induzido (custo de prompt)
TAM_FRASE = 5              # n-grama usado na detecção de boilerplate


def _tokens(texto: str) -> list[str]:
    from .preprocessamento import normalizar_texto, tokenizar
    return tokenizar(normalizar_texto(texto or "", manter_maiusculas=True))


def _induzir_frases(textos: list[str]) -> list[str]:
    """
    Frases repetidas em quase todo registro — boilerplate por definição.

    Conta n-gramas de TAM_FRASE tokens, retém os que aparecem em >= LIMIAR_FRASE
    dos registros e devolve cada um como regex literal. Conservador de propósito:
    o dano de apagar especificação é maior que o de deixar ruído.
    """
    if not textos:
        return []
    df_ngrama: Counter = Counter()
    for texto in textos:
        toks = _tokens(texto)
        vistos = {" ".join(toks[i:i + TAM_FRASE])
                  for i in range(max(0, len(toks) - TAM_FRASE + 1))}
        df_ngrama.update(vistos)
    corte = max(2, int(LIMIAR_FRASE * len(textos)))
    frequentes = sorted((ng for ng, n in df_ngrama.items() if n >= corte),
                        key=lambda ng: (-df_ngrama[ng], ng))
    # Absorve n-gramas contidos em outros já aceitos, para não gerar 30 regex
    # que casam o mesmo trecho.
    aceitas: list[str] = []
    for ng in frequentes:
        if not any(ng in ja for ja in aceitas):
            aceitas.append(ng)
    return [re.escape(f) for f in aceitas[:30]]


def _induzir_stopwords(textos: list[str]) -> set[str]:
    """Tokens presentes em >= LIMIAR_STOPWORD dos registros."""
    if not textos:
        return set()
    df_token: Counter = Counter()
    for texto in textos:
        df_token.update(set(_tokens(texto)))
    corte = LIMIAR_STOPWORD * len(textos)
    return {t.lower() for t, n in df_token.items() if n >= corte and not t.isdigit()}


def _induzir_atributos(atributos_por_item: list[dict]) -> dict[str, dict]:
    """
    Esquema de atributos, lido dos rótulos estruturados do catálogo.

    O CATMAT já vem como `TIPO, ATRIBUTO: VALOR , ...`, então as chaves existem
    no dado — o que falta é escolher quais valem e descrevê-las para o prompt.
    A descrição é montada com os valores mais frequentes de cada chave, que é
    informação do corpus, não invenção.
    """
    freq: Counter = Counter()
    valores: dict[str, Counter] = defaultdict(Counter)
    for atribs in atributos_por_item:
        for chave, valor in (atribs or {}).items():
            if chave in ("tipo_produto", "texto_completo") or not valor:
                continue
            freq[chave] += 1
            valores[chave][str(valor)[:40]] += 1

    total = max(1, len(atributos_por_item))
    esquema: dict[str, dict] = {}
    for chave, n in freq.most_common(MAX_ATRIBUTOS):
        if n < max(3, 0.01 * total):
            continue
        exemplos = [v for v, _ in valores[chave].most_common(3)]
        amostra = [v for v, _ in valores[chave].most_common(30)]
        numericos = sum(1 for v in amostra if re.fullmatch(r"[\d,\.]+", v))
        curtos = sum(1 for v in amostra if len(v) <= 4)
        if amostra and numericos / len(amostra) > 0.7:
            tipo = "numero"
        elif amostra and curtos / len(amostra) > 0.7:
            tipo = "unidade"
        else:
            tipo = "texto"
        esquema[chave] = {
            "tipo": tipo,
            "descricao": "ex " + ", ".join(f'"{e}"' for e in exemplos) if exemplos else "",
            "origem": "induzido",
        }
    return esquema


def _induzir_familias(pdms: list[str]) -> list[str]:
    """
    Famílias candidatas: o substantivo-núcleo do rótulo do PDM.

    Em catálogo de compras o rótulo começa pelo tipo do produto ("AGULHA PUNÇÃO
    ÓSSEA", "TUBO HOSPITALAR"), então o primeiro token com 3+ letras que se
    repete em vários PDMs distintos é um bom candidato a família.
    """
    por_nucleo: dict[str, set] = defaultdict(set)
    for pdm in pdms:
        toks = [t for t in sanitizar(pdm).split("_") if len(t) >= 3 and not t.isdigit()]
        if toks:
            por_nucleo[toks[0]].add(sanitizar(pdm))
    return sorted(n for n, pdms_do_nucleo in por_nucleo.items() if len(pdms_do_nucleo) >= 2)


def _induzir_definidoras(pdms: list[str], atributos_por_item: list[dict],
                         familias: list[str]) -> dict[str, list[str]]:
    """
    Características definidoras por família: os atributos que quase todo item da
    família preenche. É a mesma leitura da §2.2 — o que a família sempre diz é o
    que a distingue por dentro —, medida no corpus em vez de curada.
    """
    itens_por_familia: dict[str, list[dict]] = defaultdict(list)
    for pdm, atribs in zip(pdms, atributos_por_item):
        pdm_san = sanitizar(pdm)
        for fam in familias:
            if fam in pdm_san:
                itens_por_familia[fam].append(atribs or {})
                break

    definidoras: dict[str, list[str]] = {}
    for fam, itens in itens_por_familia.items():
        if len(itens) < MIN_ITENS_FAMILIA:
            continue
        presenca: Counter = Counter()
        for atribs in itens:
            for chave, valor in atribs.items():
                if chave not in ("tipo_produto", "texto_completo") and valor:
                    presenca[chave] += 1
        escolhidas = [c for c, n in presenca.most_common()
                      if n / len(itens) >= LIMIAR_DEFINIDORA][:MAX_DEFINIDORAS]
        if escolhidas:
            definidoras[fam] = escolhidas
    return definidoras


def _induzir_hierarquia(pdms: list[str], familias: list[str]) -> dict[str, list[str]]:
    """
    Subsunção por contenção de rótulo: "AGULHA" ⊃ "AGULHA PUNÇÃO ÓSSEA".

    É a relação que o blocking usa na estratégia 2 e que a TBox vira subclasse.
    Só sai daqui o que está literalmente no rótulo — nada de semântica adivinhada.
    """
    unicos = sorted({p.strip().upper() for p in pdms if p and p.strip()})
    hierarquia: dict[str, list[str]] = defaultdict(list)
    for fam in familias:
        filhos = [p for p in unicos if fam in sanitizar(p) and sanitizar(p) != fam]
        if len(filhos) >= 2:
            hierarquia[fam] = filhos[:40]
    return dict(hierarquia)


SEPARADORES_DIMENSAO = {"X", "POR"}   # "16 G X 0,5 CM": marcador, não unidade
RATIO_UNIDADE = 0.5                   # fração das ocorrências que segue um número


def _induzir_unidades(textos: list[str], ja_conhecidas: dict) -> dict[str, dict]:
    """
    Tokens que se comportam como unidade no corpus: aparecem colados a um número
    na maior parte das vezes em que aparecem.

    O teste de proporção é o que separa unidade de palavra comum. "MM" quase só
    ocorre depois de número; "MODELO" e "CERCA" também ocorrem depois de número,
    mas muito mais fora dessa posição — e por isso não passam. Sem esse filtro a
    indução produz `CERCA`, `VIA` e `PECA` como unidades, o que envenena a tabela
    de conversão e o grafo de unidades da rede semântica.

    Entram SEM fator de conversão: o pipeline passa a reconhecê-los como unidade
    e a comparar por igualdade, mas não converte — inventar que 1 CH = 0,33 MM
    seria criar equivalência que ninguém verificou.
    """
    from .preprocessamento import normalizar_texto

    apos_numero: Counter = Counter()
    total: Counter = Counter()
    pos_numero = re.compile(r"\d[\d,\.]*\s*([A-Z]{1,6})\b")
    palavra = re.compile(r"\b([A-Z]{1,6})\b")
    for texto in textos:
        t = normalizar_texto(texto or "", manter_maiusculas=True)
        apos_numero.update(pos_numero.findall(t))
        total.update(palavra.findall(t))

    corte = max(5, 0.005 * max(1, len(textos)))
    achadas = {}
    for tok, n in apos_numero.most_common(60):
        if tok in ja_conhecidas or tok in SEPARADORES_DIMENSAO or n < corte:
            continue
        if n / max(1, total[tok]) < RATIO_UNIDADE:
            continue
        achadas[tok] = {"categoria": f"induzida_{tok}", "base": tok, "origem": "induzido"}
        if len(achadas) >= 20:
            break
    return achadas


def induzir_perfil(perfil: PerfilDominio, df) -> PerfilDominio:
    """
    Completa in place os blocos vazios do perfil a partir do corpus.

    Bloco preenchido pelo YAML não é tocado: curadoria sempre vence indução. O
    que foi induzido fica marcado em `perfil.proveniencia`, e o que não dá para
    induzir entra em `perfil.lacunas` — as duas coisas vão para o YAML de
    resultados, para que um número possa ser lido junto com o que o sustenta.
    """
    from .preprocessamento import extrair_atributos_catmat

    if "item_catmat" not in df.columns:
        return perfil

    textos_cat = df["item_catmat"].fillna("").astype(str).tolist()
    textos_efi = df.get("item_efisco", df["item_catmat"]).fillna("").astype(str).tolist()
    atributos_por_item = [extrair_atributos_catmat(t) for t in textos_cat]
    pdms = [a.get("tipo_produto", "") for a in atributos_por_item]

    induzidos: list[str] = []
    lacunas: list[str] = []

    if not perfil.atributos:
        perfil.atributos = _induzir_atributos(atributos_por_item)
        induzidos.append("atributos")
        lacunas.append("extracao: sem regras regex — a fase [2] depende de LLM "
                       "(--sem-llm extrai só o rótulo estruturado do catálogo)")

    familias = list(perfil.familias)
    if not familias:
        familias = _induzir_familias(pdms)
        perfil.familias_extra = tuple(familias)
        induzidos.append("familias_extra")

    if not perfil.atributos_definidores:
        perfil.atributos_definidores = _induzir_definidoras(pdms, atributos_por_item, familias)
        induzidos.append("atributos_definidores")

    if not perfil.hierarquia:
        perfil.hierarquia = _induzir_hierarquia(pdms, familias)
        induzidos.append("hierarquia")

    if not perfil.stopwords:
        perfil.stopwords = frozenset(_induzir_stopwords(textos_efi))
        induzidos.append("stopwords")

    if not perfil.entidade_stopwords:
        perfil.entidade_stopwords = frozenset(
            t.upper() for t in perfil.stopwords if len(t) >= 3)
        induzidos.append("entidade_stopwords")

    if not perfil.boilerplate:
        perfil.boilerplate = tuple(_induzir_frases(textos_efi))
        induzidos.append("boilerplate")
        lacunas.append("boilerplate: só frases repetidas em >=40% dos registros; "
                       "cláusula legal esparsa continua no texto")

    # Unidades só são induzidas quando o perfil não traz tabela: curadoria vence
    # indução, aqui como em todo bloco. Um perfil com tabela completa (o MMH tem
    # gauge e french) não deve receber palpite nenhum.
    if not perfil.proveniencia.get("unidades") == "curado":
        novas_unidades = _induzir_unidades(textos_cat, perfil.unidades)
        if novas_unidades:
            perfil.unidades = {**perfil.unidades, **novas_unidades}
            induzidos.append("unidades")
            lacunas.append("unidades induzidas sem fator de conversão (comparam por "
                           "igualdade, não convertem): " + ", ".join(sorted(novas_unidades)))

    if not perfil.extracao:
        lacunas.append("extracao: sem regras regex para este domínio")

    perfil._invalidar()
    for bloco in induzidos:
        perfil.proveniencia[bloco] = "induzido"
    perfil.lacunas = tuple(dict.fromkeys(list(perfil.lacunas) + lacunas))

    if induzidos:
        log.info("Perfil '%s': %d bloco(s) induzido(s) do corpus — %s",
                 perfil.nome, len(induzidos), ", ".join(induzidos))
    for lacuna in lacunas:
        log.warning("Lacuna do perfil '%s' — %s", perfil.nome, lacuna)
    return perfil


# ---------------------------------------------------------------------------
# Contexto ativo
# ---------------------------------------------------------------------------

@dataclass
class Contexto:
    """O dataset e o perfil em uso nesta execução."""

    dataset: DatasetSpec
    perfil: PerfilDominio
    _induzido: bool = False

    def completar_com_dados(self, df) -> PerfilDominio:
        """
        Completa os blocos vazios do perfil a partir do corpus, uma vez por
        execução. Chamado pelos carregadores — quem lê os dados primeiro paga.
        """
        if self._induzido:
            return self.perfil
        self._induzido = True
        return induzir_perfil(self.perfil, df)


_ATIVO: Optional[Contexto] = None


def ativar(nome: str = "", *, perfil: str = "") -> Contexto:
    """
    Define o dataset (e, opcionalmente, força outro perfil) da execução.

    A CLI chama isto antes de qualquer leitura. Como os módulos consultam
    `contexto()` dentro das funções, e não no import, trocar de dataset no meio
    do processo funciona — é assim que se comparam dois datasets num mesmo script.
    """
    global _ATIVO
    nome = nome or DATASET_PADRAO
    ds = carregar_dataset(nome)
    _ATIVO = Contexto(dataset=ds, perfil=carregar_perfil(perfil or ds.perfil))
    log.info("Dataset: %s | perfil: %s", ds.nome, _ATIVO.perfil.nome)
    return _ATIVO


def contexto() -> Contexto:
    """O contexto ativo, ativando o padrão na primeira chamada."""
    return _ATIVO if _ATIVO is not None else ativar()


def perfil_ativo() -> PerfilDominio:
    return contexto().perfil


def dataset_ativo() -> DatasetSpec:
    return contexto().dataset
