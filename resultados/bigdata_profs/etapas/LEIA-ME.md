# Etapas congeladas do trabalho no `bigdata_profs`

Cada pasta guarda o que uma execução produziu: `ranking_combinacoes.csv` /
`avaliacao_modular.yaml` (grade) ou `stats_pipeline.json` /
`resultado_pipeline.yaml` (pipeline), mais o `run.log` filtrado. Os arquivos
pesados e regeneráveis (`resultado_pipeline_completo.csv`, grafos) ficam só na
raiz de `resultados/bigdata_profs/`, para a configuração canônica (etapa 17).

| pasta | o que é |
|---|---|
| 01_baseline_perfil_mmh | grade original, perfil `mmh`, extrator antigo |
| 02_extrator_novo_perfil_mmh | idem com o extrator de medidas corrigido (só log) |
| 03_modulos_novos_perfil_mmh | módulos novos (fusão RRF, taxonomia, GraphRAG-IDF), perfil `mmh` |
| 04_modulos_novos_compras_publicas | idem, perfil `compras_publicas` |
| 05_fusao_candidatos | fusão RRF só para escolher candidatos; score E5 |
| 06_pre_processadores | rede semântica e GraphRAG como pré-processadores |
| 07_fusao_linear | fusão linear E5 + TF-IDF (palavra, caractere) |
| 08_grade_final | grade canônica (70 combinações) |
| 09_pipeline_llm_sem_indice_ms | pipeline A: LLM, cross-encoder, adjudicação com contexto TF-IDF |
| 10_pipeline_llm_indice_ms_sem_cross | pipeline B: LLM, sem cross, índice Microsoft GraphRAG |
| 11_pipeline_sem_llm_sem_cross | pipeline C: sem API |
| 12_pipeline_llm_sem_cross_sem_adjudicacao | pipeline D: a configuração canônica |
| 13_grade_pre_graphrag_offline_amostra200 | amostra de 200: pré-processadores GraphRAG offline |
| 14_score_final_variantes | composição do score final: veto, peso da ontologia |
| 15_grade_pre_graphrag_ms_amostra200 | amostra de 200: expansão pelo índice Microsoft (`--graphrag-ms`) |
| 16_sinal_ontologia | dedução ontológica como sinal aditivo da [5b]: peso relativo 0 / 0,3 / 0,6 / 1,0 |
| 17_pipeline_final_lista | pipeline finalizado com os padrões novos: fusão, sinais (com ontologia), sem cross, adjudicação em lista p=0,1, confiança por margem — a configuração canônica (R@3 85,2%) |
| 18_peso_adjudicacao | peso da nota da LLM em lista: 0,1 / 0,2 / 0,3 / 0,5 |

Leitura completa em `docs/RESULTADOS-BIGDATA-PROFS.md`.
