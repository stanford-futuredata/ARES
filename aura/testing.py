
from aura import AuRA 

#######################################################

# Configs

gpt_and_ada_config = {

    "rag_name": "gpt_3.5_and_ada",
    "retriever_config": {
        "retrieval_model": "text-embedding-ada-002",
        "documents": "document_filepath",
        "max_doc_length": 8192,
        "max_query_length": 8192,
        "tokenizer": None,
        "reranker": None,
        "top_k": 3
    },
    "generation_llm_config": {
        "generation_llm": "gpt-3.5-turbo-0125",
        "temperature": 0.0,
        "few_shot_prompt": "",
        "system_prompt": "You are an expert question-answering system. Answer the following query by using the information in the given documents."
    }
}

rag_configs = [gpt_and_ada_config]
ares = "placeholder"
evaluation_metrics = ["Context_Relevance", "Answer_Relevance"]

#######################################################

aura = AuRA(rag_configs, ares, evaluation_metrics)


