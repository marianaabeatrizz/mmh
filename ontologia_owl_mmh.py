"""
ontologia_owl_mmh.py
====================
Ontologia OWL de dominio para o casamento CATMAT <-> e-Fisco, com raciocinio
DEDUTIVO real (Pellet via owlready2).

Implementa a Secao 2 do documento tecnico:

  2.1  O CATMAT como ontologia
       Classe      ~ PDM              -> owl:Class (hierarquia real)
       Propriedades~ caracteristicas  -> owl:DatatypeProperty
       Individuo   ~ item / codigo BR -> owl:NamedIndividual

  2.2  Regra de equivalencia (deducao)
       SE  item_A pertence a Classe/PDM X
       E   item_B pertence a Classe/PDM X (ou a um PDM equivalente)
       E   para toda caracteristica definidora c de X:
             valorNormalizado(item_A,c) = valorNormalizado(item_B,c)
       ENTAO infere item_A :equivalenteA item_B

       Traduzida para SWRL de verdade (owlready2 `Imp`), executada pelo Pellet.
       Nao e heuristica em Python: e a regra rodando num reasoner.

  4.3 [4] Ontologia + Reasoner
       - equivalenteA  : deduzido pelo reasoner (SWRL)
       - broadMatch    : deduzido por SUBSUNCAO na arvore de PDMs
       - veto          : conflito explicito de caracteristica definidora

RACIOCINIO POR LOTES DE PDM
---------------------------
Uma passada unica do Pellet sobre a ABox inteira (~2000 individuos, 200+
classes, 15 regras) estoura a heap da JVM: o motor SWRL materializa os joins
intermediarios antes de filtrar.

A particao por PDM resolve isso SEM PERDER NENHUMA DEDUCAO, e a razao e a
propria forma da regra 2.2: o corpo exige `pertenceAoPDM(?a,?p)` e
`pertenceAoPDM(?b,?p)` com a MESMA variavel ?p. Dois itens de PDMs distintos
portanto jamais satisfazem o corpo — nenhuma equivalencia cruza a fronteira do
PDM, e raciocinar cada bloco isoladamente produz exatamente o mesmo conjunto
de conclusoes que a passada global produziria.

O broadMatch, que por definicao cruza PDMs, nao depende de SWRL: e lido da
hierarquia de classes da TBox (ver `subsuncao_entre`), que e montada uma vez e
nao e afetada pelo particionamento da ABox.

Sobre o veto: SWRL e monotonico e nao tem negacao por falha, entao "os
atributos divergem" nao e expressavel como regra de inferencia positiva. O veto
fica como verificacao de consistencia em Python, aplicada DEPOIS da deducao —
e esta explicitamente marcado como tal na trilha.

Uso:
    from ontologia_owl_mmh import OntologiaMMH
    onto = OntologiaMMH(pdms, hierarquia)
    onto.adicionar_item("efisco", "123", "AGULHA", {"calibre": "22G"})
    onto.adicionar_item("catmat", "456", "AGULHA", {"calibre": "22G"})
    resultado = onto.deduzir()          # roda o Pellet por lotes
    resultado[("123", "456")]           # -> {'deducao': 'equivalenteA', ...}
"""

from __future__ import annotations

import re
import time
import logging
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
IRI_ONTOLOGIA = "http://mmh.sad.pe.gov.br/ontologia/catmat-efisco.owl"

# Heap da JVM usada pelo Pellet. O padrao do owlready2 (2 GB) e apertado para
# ABox com milhares de individuos; os lotes ja mantem cada passada pequena,
# mas a folga evita falha em blocos de PDM atipicamente grandes.
JAVA_MEMORY_MB = 4000

# Teto de individuos por passada do reasoner. Um PDM sozinho maior que isto
# roda mesmo assim (nao da para dividi-lo sem perder deducoes), com aviso.
LOTE_MAX_INDIVIDUOS = 300

# ---------------------------------------------------------------------------
# Caracteristicas DEFINIDORAS por familia de PDM (sec. 2.2)
# Sao as que entram na regra SWRL: se todas coincidem, o reasoner deduz
# equivalencia. Atributos fora desta lista contribuem como evidencia fraca,
# mas nunca decidem — exatamente a distincao do documento entre "definidora"
# e "secundaria".
# ---------------------------------------------------------------------------

CARACTERISTICAS_DEFINIDORAS: dict[str, list[str]] = {
    "AGULHA":      ["calibre", "comprimento_valor"],
    "SERINGA":     ["volume_valor"],
    "CATETER":     ["material", "comprimento_valor"],
    "SONDA":       ["calibre", "material"],
    "FIO":         ["material"],
    "EQUIPO":      ["material"],
    "LUVA":        ["material"],
    "TUBO":        ["material", "calibre"],
    "DRENO":       ["material", "calibre"],
    "MASCARA":     ["material"],
    "COMPRESSA":   ["material", "dimensao"],
    "ATADURA":     ["dimensao"],
    "ESPARADRAPO": ["dimensao"],
    "FRASCO":      ["volume_valor"],
    "BOLSA":       ["volume_valor", "material"],
}

PROPRIEDADES_PDM = (
    "calibre", "comprimento_valor", "comprimento_unidade",
    "volume_valor", "volume_unidade", "dimensao",
    "conector", "esterilidade", "material", "bisel", "modelo",
)


def _sanitizar(nome: str) -> str:
    """Converte um rotulo livre num identificador valido para OWL/Python."""
    txt = unicodedata.normalize("NFD", str(nome))
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    txt = re.sub(r"[^A-Za-z0-9]+", "_", txt.upper()).strip("_")
    if not txt:
        txt = "SEM_PDM"
    if txt[0].isdigit():
        txt = "N" + txt
    return txt[:80]


def _familia_de(pdm: str) -> str:
    """Familia (chave de CARACTERISTICAS_DEFINIDORAS) a que um PDM pertence."""
    pdm_up = _sanitizar(pdm)
    for familia in CARACTERISTICAS_DEFINIDORAS:
        if familia in pdm_up:
            return familia
    return ""


def _normalizar_valor(valor: str) -> str:
    """
    valorNormalizado() da regra 2.2 — a comparacao do reasoner e SINTATICA
    (igualdade de literal), entao toda a tolerancia tem de estar aqui:
    caixa alta, sem acento, decimal com ponto, sem zeros a direita.
    """
    if valor is None:
        return ""
    txt = unicodedata.normalize("NFD", str(valor).strip())
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    txt = re.sub(r"\s+", " ", txt.upper()).strip()
    if re.fullmatch(r"-?\d+[.,]?\d*", txt):
        try:
            f = float(txt.replace(",", "."))
            txt = str(int(f)) if f == int(f) else f"{f:g}"
        except ValueError:
            pass
    return txt


class OntologiaMMH:
    """
    Ontologia OWL do dominio + motor de deducao.

    Os itens sao registrados em Python (`adicionar_item`) e so viram individuos
    OWL dentro de `deduzir()`, um lote de PDM por vez, cada lote no seu proprio
    `World` — que e descartado ao fim da passada. Sem isso a ABox acumulada
    derruba a JVM.
    """

    def __init__(self, pdms: list[str], hierarquia: Optional[dict[str, list[str]]] = None):
        import owlready2

        owlready2.reasoning.JAVA_MEMORY = JAVA_MEMORY_MB

        self._ow = owlready2
        self.pdms = list(pdms)
        self.hierarquia = hierarquia or {}

        # Registro dos itens (ainda sem OWL).
        self._pdm_de_item: dict[str, str] = {}
        self._attrs_de_item: dict[str, dict] = {}
        self._lado_de_item: dict[str, str] = {}

        # TBox de referencia: usada para subsuncao e para salvar a ontologia.
        self._mundo_tbox = owlready2.World()
        self.onto, self._classes = self._montar_tbox(self._mundo_tbox)

        log.info(
            "TBox montada: %d classes PDM | %d propriedades | %d regras SWRL",
            sum(1 for k in self._classes if k.startswith("pdm::")),
            len(PROPRIEDADES_PDM),
            len(list(self.onto.rules())),
        )

    # -----------------------------------------------------------------------
    # TBox — montada identicamente em qualquer World
    # -----------------------------------------------------------------------

    def _montar_tbox(self, mundo) -> tuple:
        """
        Constroi a TBox completa (classes, propriedades, hierarquia, regras)
        num World dado. Repetivel: cada lote monta a sua.
        """
        from owlready2 import (
            Thing, DataProperty, FunctionalProperty, SymmetricProperty, Imp,
        )

        onto = mundo.get_ontology(IRI_ONTOLOGIA)
        classes: dict[str, type] = {}

        with onto:
            class Item(Thing):
                """Qualquer item de material; superclasse dos dois lados."""

            class ItemEFisco(Item):
                """Individuo vindo do e-Fisco (texto livre)."""

            class ItemCATMAT(Item):
                """Individuo vindo do CATMAT (rotulo estruturado)."""

            class Material(Thing):
                """Raiz da arvore de PDM (2.1: Classe ~ PDM)."""

            class pertenceAoPDM(Item >> Material):
                """Item -> sua classe/PDM (2.1: Individuo ~ item)."""

            class equivalenteA(Item >> Item, SymmetricProperty):
                """Relacao deduzida pela regra 2.2."""

            for nome in PROPRIEDADES_PDM:
                # Funcional: cada item tem no maximo um valor por caracteristica.
                classes[f"prop::{nome}"] = type(
                    nome, (DataProperty, FunctionalProperty),
                    {"domain": [Item], "range": [str]},
                )

            classes["Item"] = Item
            classes["ItemEFisco"] = ItemEFisco
            classes["ItemCATMAT"] = ItemCATMAT
            classes["Material"] = Material
            classes["pertenceAoPDM"] = pertenceAoPDM
            classes["equivalenteA"] = equivalenteA

            self._montar_hierarquia(onto, classes, Material)
            self._montar_regras(Imp)

        return onto, classes

    def _classe_pdm(self, onto, classes: dict, Material, pdm: str) -> type:
        """Cria (ou recupera) a owl:Class que representa um PDM."""
        nome = _sanitizar(pdm)
        chave = f"pdm::{nome}"
        if chave in classes:
            return classes[chave]
        with onto:
            cls = type(nome, (Material,), {})
        cls.label = [str(pdm)]
        classes[chave] = cls
        return cls

    def _montar_hierarquia(self, onto, classes: dict, Material) -> None:
        """
        Arvore de classes de PDM. A hierarquia curada da rede semantica
        (hiperonimo -> hiponimos) vira subclasse OWL de verdade, e e ela que
        sustenta o broadMatch por subsuncao.
        """
        for pdm in self.pdms:
            self._classe_pdm(onto, classes, Material, pdm)

        # Familias como classes intermediarias: AGULHA_BIOPSIA ⊑ AGULHA ⊑ Material
        for familia in CARACTERISTICAS_DEFINIDORAS:
            cls_fam = self._classe_pdm(onto, classes, Material, familia)
            for pdm in self.pdms:
                nome_pdm = _sanitizar(pdm)
                if familia in nome_pdm and nome_pdm != familia:
                    cls_filho = self._classe_pdm(onto, classes, Material, pdm)
                    if cls_fam not in cls_filho.is_a:
                        cls_filho.is_a.append(cls_fam)

        for hiper, hipos in self.hierarquia.items():
            cls_hiper = self._classe_pdm(onto, classes, Material, hiper)
            for hipo in hipos:
                cls_hipo = self._classe_pdm(onto, classes, Material, hipo)
                if cls_hiper not in cls_hipo.is_a:
                    cls_hipo.is_a.append(cls_hiper)

    def _montar_regras(self, Imp) -> None:
        """
        Traduz a pseudo-regra da sec. 2.2 em SWRL executavel, uma regra por
        familia (cada familia tem suas caracteristicas definidoras).

        Forma gerada, p/ AGULHA (definidoras = calibre, comprimento_valor):

            ItemEFisco(?a), ItemCATMAT(?b),
            pertenceAoPDM(?a,?p), pertenceAoPDM(?b,?p),
            calibre(?a,?v0), calibre(?b,?v0),
            comprimento_valor(?a,?v1), comprimento_valor(?b,?v1)
            -> equivalenteA(?a,?b)

        A igualdade de valor e expressa reusando a MESMA variavel nos dois
        atomos — que e como SWRL diz "valorNormalizado(A,c) =
        valorNormalizado(B,c)" sem built-in de comparacao.

        broadMatch NAO vira regra SWRL: exigiria differentFrom(?p,?q), que o
        Pellet so honra com assercao explicita de desigualdade. A subsuncao e
        lida depois, da hierarquia da TBox (ver subsuncao_entre).
        """
        n_regras = 0
        for familia, definidoras in CARACTERISTICAS_DEFINIDORAS.items():
            if not definidoras:
                continue
            corpo = [
                "ItemEFisco(?a)", "ItemCATMAT(?b)",
                "pertenceAoPDM(?a, ?p)", "pertenceAoPDM(?b, ?p)",
            ]
            for i, attr in enumerate(definidoras):
                corpo.append(f"{attr}(?a, ?v{i})")
                corpo.append(f"{attr}(?b, ?v{i})")
            try:
                Imp().set_as_rule(f"{', '.join(corpo)} -> equivalenteA(?a, ?b)")
                n_regras += 1
            except Exception as exc:
                log.warning("Regra SWRL da familia %s rejeitada: %s", familia, exc)
        log.info("  -> %d regras SWRL de equivalencia instaladas", n_regras)

    # -----------------------------------------------------------------------
    # ABox (registro em Python; individuos OWL so na hora de raciocinar)
    # -----------------------------------------------------------------------

    def adicionar_item(self, lado: str, codigo: str, pdm: str, atributos: dict) -> None:
        """
        Registra um item. Nao cria o individuo OWL ainda — isso acontece em
        `deduzir()`, no lote do PDM correspondente.

        Args:
            lado: "efisco" ou "catmat".
            codigo: identificador unico do item.
            pdm: rotulo do PDM ancorado.
            atributos: caracteristicas ja extraidas (fase [2]).
        """
        chave = f"{lado}::{codigo}"
        if chave in self._attrs_de_item:
            return
        self._pdm_de_item[chave] = pdm or ""
        self._lado_de_item[chave] = lado
        self._attrs_de_item[chave] = {
            nome: v for nome in PROPRIEDADES_PDM
            if (v := _normalizar_valor(atributos.get(nome, "")))
        }

    # -----------------------------------------------------------------------
    # Deducao por lotes
    # -----------------------------------------------------------------------

    def _lotes_por_pdm(self, teto: int) -> list[list[str]]:
        """Agrupa as chaves de item em lotes, sem nunca partir um PDM."""
        por_pdm: dict[str, list[str]] = defaultdict(list)
        for chave, pdm in self._pdm_de_item.items():
            por_pdm[_sanitizar(pdm)].append(chave)

        lotes, atual = [], []
        for nome_pdm, chaves in sorted(por_pdm.items(), key=lambda kv: -len(kv[1])):
            if len(chaves) > teto:
                log.warning("PDM '%s' tem %d individuos (acima do teto de %d) — "
                            "roda sozinho.", nome_pdm, len(chaves), teto)
                lotes.append(chaves)
                continue
            if len(atual) + len(chaves) > teto:
                lotes.append(atual)
                atual = []
            atual.extend(chaves)
        if atual:
            lotes.append(atual)
        return lotes

    def deduzir(self, teto_lote: int = LOTE_MAX_INDIVIDUOS) -> dict[tuple[str, str], dict]:
        """
        Roda o Pellet lote a lote e devolve as equivalencias deduzidas.

        Returns:
            dict[(cod_efisco, cod_catmat)] -> {
                'deducao': 'equivalenteA', 'score': float,
                'explicacao': list[str], 'fonte': 'reasoner'
            }
        """
        from owlready2 import World, sync_reasoner_pellet

        if not self._attrs_de_item:
            return {}

        lotes = self._lotes_por_pdm(teto_lote)
        log.info("Deducao por lotes: %d individuos em %d lotes (teto %d).",
                 len(self._attrs_de_item), len(lotes), teto_lote)

        resultados: dict[tuple[str, str], dict] = {}
        n_falhas = 0
        t0 = time.time()

        for n_lote, chaves in enumerate(lotes, 1):
            mundo = World()
            try:
                onto, classes = self._montar_tbox(mundo)
                Material = classes["Material"]
                nome_para_chave: dict[str, str] = {}

                with onto:
                    for chave in chaves:
                        lado = self._lado_de_item[chave]
                        codigo = chave.split("::", 1)[1]
                        cls_item = classes["ItemEFisco"] if lado == "efisco" \
                            else classes["ItemCATMAT"]
                        nome_ind = _sanitizar(f"{lado}_{codigo}")
                        ind = cls_item(nome_ind)

                        pdm = self._pdm_de_item[chave] or "SEM_PDM"
                        cls_pdm = self._classe_pdm(onto, classes, Material, pdm)
                        ind.pertenceAoPDM.append(
                            cls_pdm(_sanitizar(f"PDM_{pdm}"))
                        )
                        for nome, valor in self._attrs_de_item[chave].items():
                            setattr(ind, nome, valor)
                        nome_para_chave[nome_ind] = chave

                with onto:
                    sync_reasoner_pellet(mundo, infer_property_values=True,
                                         infer_data_property_values=False, debug=0)

                self._colher(onto, nome_para_chave, resultados)

            except Exception as exc:
                n_falhas += 1
                log.error("Lote %d/%d (%d individuos) falhou: %s: %s",
                          n_lote, len(lotes), len(chaves),
                          type(exc).__name__, str(exc)[:160])
            finally:
                try:
                    mundo.close()
                except Exception:
                    pass

            if n_lote % 5 == 0 or n_lote == len(lotes):
                log.info("  lote %d/%d — %d equivalencias ate aqui (%.0fs)",
                         n_lote, len(lotes), len(resultados), time.time() - t0)

        log.info("  -> Pellet concluiu em %.1fs: %d equivalenteA deduzidas"
                 "%s", time.time() - t0, len(resultados),
                 f" ({n_falhas} lotes falharam)" if n_falhas else "")
        return resultados

    def _colher(self, onto, nome_para_chave: dict, resultados: dict) -> None:
        """Le as equivalencias inferidas num lote ja raciocinado."""
        for ind in onto.individuals():
            chave = nome_para_chave.get(ind.name)
            if not chave or self._lado_de_item.get(chave) != "efisco":
                continue
            cod_e = chave.split("::", 1)[1]

            for alvo in getattr(ind, "equivalenteA", []):
                chave_alvo = nome_para_chave.get(getattr(alvo, "name", ""))
                if not chave_alvo or self._lado_de_item.get(chave_alvo) != "catmat":
                    continue
                cod_c = chave_alvo.split("::", 1)[1]
                if (cod_e, cod_c) in resultados:
                    continue

                familia = _familia_de(self._pdm_de_item[chave])
                definidoras = CARACTERISTICAS_DEFINIDORAS.get(familia, [])
                attrs_e = self._attrs_de_item[chave]
                resultados[(cod_e, cod_c)] = {
                    "deducao": "equivalenteA",
                    "score": 1.0,
                    "explicacao": [
                        "Deduzido pelo reasoner (SWRL, regra 2.2):",
                        f"PDM coincide: '{self._pdm_de_item[chave]}'",
                        f"Familia '{familia}' — definidoras: "
                        f"{', '.join(definidoras) or '(nenhuma)'}",
                    ] + [
                        f"  [{a}] = '{attrs_e.get(a, '')}' em ambos" for a in definidoras
                    ],
                    "fonte": "reasoner",
                }

    # -----------------------------------------------------------------------
    # Subsuncao e veto
    # -----------------------------------------------------------------------

    def subsuncao_entre(self, cod_efisco: str, cod_catmat: str) -> Optional[dict]:
        """
        broadMatch por SUBSUNCAO, lido da hierarquia de classes da TBox: o PDM
        do CATMAT e ancestral do PDM do e-Fisco.

        Ex.: e-Fisco em AGULHA_BIOPSIA e CATMAT em AGULHA — o CATMAT e um
        casamento mais generico, nao uma equivalencia. E o "subsuncao ->
        broadMatch" do fluxo da sec. 4.3 [4].
        """
        ce, cc = f"efisco::{cod_efisco}", f"catmat::{cod_catmat}"
        pdm_e, pdm_c = self._pdm_de_item.get(ce), self._pdm_de_item.get(cc)
        if not pdm_e or not pdm_c or _sanitizar(pdm_e) == _sanitizar(pdm_c):
            return None

        cls_e = self._classes.get(f"pdm::{_sanitizar(pdm_e)}")
        cls_c = self._classes.get(f"pdm::{_sanitizar(pdm_c)}")
        if cls_e is None or cls_c is None:
            return None

        try:
            if cls_c in cls_e.ancestors():
                return {
                    "deducao": "broadMatch",
                    "score": 0.55,
                    "explicacao": [
                        "Subsuncao na arvore de PDM da TBox:",
                        f"  e-Fisco '{pdm_e}' e subclasse de CATMAT '{pdm_c}'",
                        "Casamento mais generico, nao equivalencia.",
                    ],
                    "fonte": "reasoner_subsuncao",
                }
        except Exception:
            return None
        return None

    def verificar_veto(self, cod_efisco: str, cod_catmat: str) -> Optional[dict]:
        """
        Conflito explicito de caracteristica definidora.

        SWRL nao tem negacao por falha, entao isto NAO vem do reasoner — e uma
        checagem de consistencia sobre os mesmos valores normalizados. Fica
        marcado como fonte='consistencia' para que a trilha nao apresente como
        deducao o que nao e.
        """
        ce, cc = f"efisco::{cod_efisco}", f"catmat::{cod_catmat}"
        if ce not in self._attrs_de_item or cc not in self._attrs_de_item:
            return None

        familia = _familia_de(self._pdm_de_item.get(ce, ""))
        definidoras = CARACTERISTICAS_DEFINIDORAS.get(familia, [])
        attrs_e, attrs_c = self._attrs_de_item[ce], self._attrs_de_item[cc]

        for attr in definidoras:
            v_e, v_c = attrs_e.get(attr, ""), attrs_c.get(attr, "")
            if v_e and v_c and v_e != v_c:
                return {
                    "deducao": "veto",
                    "score": 0.0,
                    "explicacao": [
                        f"Conflito na caracteristica definidora [{attr}]:",
                        f"  e-Fisco = '{v_e}'  x  CATMAT = '{v_c}'",
                        "Veto por inconsistencia (checagem, nao deducao SWRL).",
                    ],
                    "fonte": "consistencia",
                }
        return None

    # -----------------------------------------------------------------------

    def salvar(self, destino: Optional[Path] = None) -> Path:
        """Grava a TBox (classes, propriedades e regras SWRL) em RDF/XML."""
        destino = destino or (DATA_DIR / "ontologia_mmh.owl")
        self.onto.save(file=str(destino), format="rdfxml")
        log.info("Ontologia OWL salva: %s", destino)
        return destino

    def estatisticas(self) -> dict:
        return {
            "n_classes_pdm": sum(1 for k in self._classes if k.startswith("pdm::")),
            "n_propriedades": len(PROPRIEDADES_PDM),
            "n_individuos": len(self._attrs_de_item),
            "n_regras_swrl": len(list(self.onto.rules())),
        }
