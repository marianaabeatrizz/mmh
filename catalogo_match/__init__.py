"""
catalogo_match — casamento neuro-simbolico de itens entre catalogos de compras.

O pacote implementa o pipeline CATMAT <-> e-Fisco das fases [0]-[9] e a grade
modular de avaliacao. Nenhum modulo conhece o dominio: o que e conhecimento
(lexico, familias, unidades, esquema de atributos, papeis nos prompts) vem do
PERFIL, e o que e corpus (arquivos, separador, saidas) vem do DATASET. Os dois
sao declarados em config/perfis/ e config/datasets/ e resolvidos por
`catalogo_match.config`.

    from catalogo_match.config import ativar
    from catalogo_match.avaliacao_modular import carregar_corpus

    ativar("mmh")                 # ou CM_DATASET=mmh no ambiente
    corpus = carregar_corpus()

Modulos:
    config                 DatasetSpec + PerfilDominio + inducao do perfil
    preprocessamento       normalizacao de texto, atributos do rotulo CATMAT
    rede_semantica         fase [1]: grafo lexico de dominio + expansao
    ontologia_owl          fase [4]: TBox OWL + SWRL + reasoner Pellet
    graphrag               fase [6] e os modulos GraphRAG da grade
    pipeline               orquestrador das fases [0]-[9]
    avaliacao_modular      grade pre x proc x pos + ranking de combinacoes
    servico_rede_semantica servico HTTP do lexico (FastAPI)
"""

__all__ = [
    "config",
    "preprocessamento",
    "rede_semantica",
    "ontologia_owl",
    "graphrag",
    "pipeline",
    "avaliacao_modular",
]
