Etapa 02 — extrator de medidas corrigido, perfil `mmh` (controle da correção).

Só o `run.log` está aqui: a exportação desta execução falhou num erro de código
que foi corrigido em seguida (`NameError: ctx` em `exportar`), então as métricas
válidas são as linhas `MRR ... R@3` do log. Comparar com a etapa 01:

    basico + e5 + medidas_graphrag   0,777 (01)  ->  0,778 (02)
    nada   + e5 + medidas_graphrag   0,768 (01)  ->  0,772 (02)
