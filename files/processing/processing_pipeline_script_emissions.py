#Script que foca no processamento dos dados de similaridade, 
# para geração de uma tabela de métricas de classificação
# por tema, de acordo com métodos de agregação diferentes 
from datetime import datetime
from codecarbon import EmissionsTracker
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

init = datetime.now()

#mudar aqui caso necessário
dir_results = "./mpnet_base_v2_metrics_emissions/"

## Pré Processamento Comum

#Processamento Dados Originais
path_original_data = "../../data/leandl_oesnpg_dados.parquet"
df_original = pd.read_parquet(path_original_data)

#Remove colunas não necessárias, remove valores NaN, agrupa por hash_id e tema_id e converte os vários "modelo_nivel" em uma lista
df_original_comparison = (
    df_original.dropna(subset=["modelo_nivel"])[["hash_id", "tema_id", "modelo_nivel"]]
     .groupby(["hash_id", "tema_id"])["modelo_nivel"]
     .agg(list)
     .reset_index()
     .sort_values(by="hash_id")
)

#Filtragem binária
def binary_level(level:str) -> str:
    return "ALTO" if level in ["MEDIA", "ALTA"] else "NAO-ALTO"

#Filtragem, dada uma lista de niveis
def level_filter(level_list: list) -> str:
    level_set = set(level_list)

    if(len(level_set) == 1):
        return binary_level(list(level_set)[0])
    
    else:
        if("BAIXA" in level_set) and ("MEDIA" in level_set):
            return "NAO-ALTO"
        
        return "ALTO"
        
df_original_comparison = df_original_comparison.assign(nivel_binario=df_original_comparison["modelo_nivel"].apply(level_filter))

#Processamento Resultados Similaridade

#mudar path caso necessário
path_results_data = "../mpnet_base_v2/bert_similarity_results_(mpnet_base_v2)_LeanDL_2025.parquet"
df_result = pd.read_parquet(path_results_data)

#filtra o df_result, removendo os  registros que possuiam modelo_nivel = None no df_original
df_result_filtered = df_result.merge(df_original_comparison[["hash_id", "tema_id"]], on=["hash_id", "tema_id"], how="inner" ).reset_index(drop=True)

#filtra df_result_filterd por titulo, resumo, palavras chave e area de concentração
source_filter = ["nome_producao", "descricao_resumo", "descricao_palavra_chave", "nome_area_concentracao"]

df_filter_source = df_result_filtered[df_result_filtered["SOURCE"].isin(source_filter)].reset_index(drop=True)

## Script

#definição de variações
chunk_agg_types = ["mean", "max"]

source_agg_weights = [{
                    "nome_producao": 0.25,
                    "descricao_resumo": 0.25,
                    "descricao_palavra_chave": 0.25,
                    "nome_area_concentracao": 0.25,
                    "nome": "025_todos"
                    },
                    {
                    "nome_producao": 0.3,
                    "descricao_resumo": 0.3,
                    "descricao_palavra_chave": 0.2,
                    "nome_area_concentracao": 0.2,
                    "nome": "03_producao_resumo_02_palavra_area"
                    },
                    {
                    "nome_producao": 0.2,
                    "descricao_resumo": 0.2,
                    "descricao_palavra_chave": 0.3,
                    "nome_area_concentracao": 0.3,
                    "nome": "02_producao_resumo_03_palavra_area"
                    }
]

theme_keywords_agg_weights = [{"theme": round((i/10), 2), "keywords": round(1 - (i/10),2), "nome": f"0{i}_tema_0{10-i}_palavras"} for i in range(1, 10)]

# Lista para armazenar emissões por combinação
emissions_results = []

for chunk_agg_type in chunk_agg_types:

    #Agregamento Chunks
    desired_fields = ["hash_id", "tema_id", "SOURCE", "tema_COSINE", "palavras_chave_COSINE"]

    df_agg_chunk = (
                df_filter_source[desired_fields]
                .groupby(["hash_id", "tema_id", "SOURCE"], as_index=False)
                .agg([chunk_agg_type])
                .droplevel([1], axis=1)
                )
    
    # filtra apenas grupos que tenham exatamente 4 sources
    valid_groups = (
        df_agg_chunk.groupby(["hash_id", "tema_id"])["SOURCE"]
        .nunique()
        .reset_index()
    )

    valid_groups = valid_groups[valid_groups["SOURCE"] == 4][["hash_id", "tema_id"]]

    #seleciona apenas os registros válidos (que possuem os 4 sources)
    df_agg_chunk = df_agg_chunk.merge(valid_groups, on=["hash_id", "tema_id"], how="inner")

    #Filtra a tabela original, para remover esses registros que geraram menos de 4 sources
    valid_ids = df_agg_chunk[["hash_id", "tema_id"]].drop_duplicates()

    df_original_comparison_filtered = df_original_comparison.merge(
        valid_ids,
        on=["hash_id", "tema_id"],
        how="inner"
    )
    
    for source_agg_w in source_agg_weights:

        #cria nova tabela de pesos, para facilita média ponderada
        df_agg_chunk["peso"] = df_agg_chunk["SOURCE"].map(source_agg_w)

        #agrega os 4 sources, fazendo uma média ponderada de acordo com os pesos da iteração vigente
        df_agg_sources = (
            df_agg_chunk.groupby(["hash_id", "tema_id"])[["tema_COSINE", "palavras_chave_COSINE", "peso"]]
            .apply(lambda g: pd.Series({
                "tema_mediap_sim": np.average(g["tema_COSINE"], weights=g["peso"]),
                "palavras_chave_mediap_sim": np.average(g["palavras_chave_COSINE"], weights=g["peso"])
            }))
            .reset_index()
        )

        for theme_keywords_agg_w in theme_keywords_agg_weights:

            # Inicia tracker para esta combinação específica
            combination_name = f"chunk-{chunk_agg_type}__campos-{source_agg_w['nome']}__simtema-{theme_keywords_agg_w['nome']}"
            
            tracker = EmissionsTracker(
                project_name=f"similarity_metrics_{combination_name}",
                experiment_id=combination_name,
                output_dir=dir_results,
                log_level="ERROR"  # Reduz verbose do log
            )
            
            tracker.start()
            combination_start_time = datetime.now()
            
            peso_tema = theme_keywords_agg_w["theme"]
            peso_palavras = theme_keywords_agg_w["keywords"]
            
            #df de agregamento final, correponde ao agregamento das similaridades de tema e palavras chave
            df_final_result = (
                df_agg_sources
                    .assign(
                        media_tema_palavras_cos=
                        (df_agg_sources["tema_mediap_sim"] * peso_tema 
                        + 
                        df_agg_sources["palavras_chave_mediap_sim"] * peso_palavras) 
                        / (peso_palavras + peso_tema)
                        )
                )
            

            thresholds = np.round(np.arange(0.05, 1.0, 0.05), decimals=2)

            #cria colunas para threshold, classificando em NAO-ALTO e ALTO, de acordo com a similaridade media (ponderada) obtida
            for threshold in thresholds:
                field = f"class_threshold_{str(float(threshold))}"
                df_final_result[field] = np.where(df_final_result["media_tema_palavras_cos"] > threshold, "ALTO", "NAO-ALTO")


            #tabela de comparação
            #cria uma tabela geral contendo tanto a coluna "nivel_binario" (classe esperada) e as colunas de threshold (classe prevista)
            df_compare = df_original_comparison_filtered.merge(
                df_final_result,
                on=["hash_id", "tema_id"],
                how="inner"
            )

            temas = df_compare["tema_id"].unique()

            #Onde ficarão os resultados das métricas de classificação, por tema, e por threshold
            results_classification_metrics = {}

            for tema in temas:
                results_classification_metrics[tema] = {}

                #seleciona da tabela de comparação, apenas os registros de determinado tema
                df_tema = df_compare[df_compare["tema_id"] == tema]

                for t in thresholds:

                    threshold_str = str(float(t))

                    #para cada threshold faz o cálculo das métricas de classficação
                    #comparando a coluna nivel_binario (classe_esperada)
                    #com a coluna do threshold atual (classe prevista)
                    report = classification_report(
                        df_tema["nivel_binario"],
                        df_tema[f"class_threshold_{threshold_str}"],
                        labels=["ALTO", "NAO-ALTO"],
                        target_names=["ALTO", "NAO-ALTO"],
                        output_dict=True,
                        zero_division=0
                    )

                    results_classification_metrics[tema][t] = report

            #Criação esqueleto dataframe de métricas:
            tabela_metricas = []

            agg_chunk = str(chunk_agg_type)
            agg_campos = str(source_agg_w["nome"])
            agg_sim_tema_palavras = str(theme_keywords_agg_w["nome"])

            for tema in results_classification_metrics:
                for threshold, report in results_classification_metrics[tema].items():

                    row = {
                        "tema": tema,
                        "threshold": threshold,
                        "agg_chunk": agg_chunk,
                        "agg_campos": agg_campos,
                        "agg_sim_tema_palavras": agg_sim_tema_palavras,
                    }

                    # adiciona accuracy
                    row["accuracy"] = report["accuracy"]

                    # Percorre todas as chaves (classes + macro/micro/weighted avg)
                    for label, metrics in report.items():
                        if label == "accuracy":
                            continue  # já adicionado

                        clean_label = label.replace(" ", "_")
                        for metric_name, value in metrics.items():
                            clean_metric = metric_name.replace("-", "_")
                            row[f"{clean_label}_{clean_metric}"] = value

                    tabela_metricas.append(row)

            df_metricas = pd.DataFrame(tabela_metricas)

            #mudar caso necessário
            nome_modelo = "mpnet_base_v2"

            nome_arquivo = f"metricas_por_tema__chunk-{agg_chunk}__campos-{agg_campos}__simtema-{agg_sim_tema_palavras}_{nome_modelo}.parquet"

            df_metricas.to_parquet(f"{dir_results}/parts/{nome_arquivo}", index=False)
            
            # Para o tracker e registra as emissões
            emissions = tracker.stop()
            combination_end_time = datetime.now()
            combination_duration = combination_end_time - combination_start_time
            
            # Armazena informações da combinação
            emissions_results.append({
                "chunk_agg_type": chunk_agg_type,
                "source_agg_name": source_agg_w["nome"],
                "theme_keywords_agg_name": theme_keywords_agg_w["nome"],
                "emissions_kg": emissions,
                "duration_seconds": combination_duration.total_seconds(),
                "start_time": combination_start_time.isoformat(),
                "end_time": combination_end_time.isoformat(),
                "filename": nome_arquivo
            })
            
            print(f"Arquivo salvo: {nome_arquivo}")
            print(f"Emissões CO2 desta combinação: {emissions:.6f} kg")
            print(f"Duração: {combination_duration}")
            print("-" * 80)

# Salva relatório de emissões por combinação
df_emissions = pd.DataFrame(emissions_results)
emissions_report_file = f"{dir_results}emissions_report_by_combination.parquet"
df_emissions.to_parquet(emissions_report_file, index=False)

# Salva também em CSV para fácil visualização
df_emissions.to_csv(f"{dir_results}emissions_report_by_combination.csv", index=False)

# Exibe resumo das emissões
print("\n" + "="*80)
print("RESUMO DE EMISSÕES DE CARBONO POR COMBINAÇÃO")
print("="*80)

total_emissions = df_emissions["emissions_kg"].sum()
total_duration = df_emissions["duration_seconds"].sum()

print(f"Total de combinações processadas: {len(df_emissions)}")
print(f"Emissões totais: {total_emissions:.6f} kg CO2")
print(f"Duração total: {total_duration/3600:.2f} horas")
print(f"Emissão média por combinação: {total_emissions/len(df_emissions):.6f} kg CO2")

print("\nTop 5 combinações com maiores emissões:")
top_emissions = df_emissions.nlargest(5, "emissions_kg")[
    ["chunk_agg_type", "source_agg_name", "theme_keywords_agg_name", "emissions_kg", "duration_seconds"]
]
for _, row in top_emissions.iterrows():
    print(f"  {row['chunk_agg_type']} | {row['source_agg_name']} | {row['theme_keywords_agg_name']}: "
          f"{row['emissions_kg']:.6f} kg CO2 ({row['duration_seconds']:.1f}s)")

print(f"\nRelatório detalhado salvo em: {emissions_report_file}")

end = datetime.now()
print(f"\nInicio: {init}")
print(f"Fim: {end}")
print(f"Tempo Total: {end - init}")
print(f"Emissões totais do script completo: {total_emissions:.6f} kg CO2")