from retriever.retriever import Retriever, OpenAI_Ada
from generator.generation_llm import Generation_LLM, OpenAI_GPT

class RAG_Model:
    def __init__(self, rag_config):
        self.rag_name = rag_config['rag_name']
        
        if "text-embedding-ada-002" == rag_config['retriever_config']["retrieval_model"]:
            self.retriever = OpenAI_Ada(config=rag_config['retriever_config'])
        if "gpt" in rag_config['generation_llm_config']["generation_llm"]:
            self.generation_llm = OpenAI_GPT(config=rag_config['generation_llm_config'])
        
        self.ares_results = None

    def initialize_rag_model(self, reinitialize_retriever=False, reinitialize_generation_llm=False):

        if not self.retriever.initialized or reinitialize_retriever:
            self.retriever.initialize_retriever()

        if not self.generation_llm.initialized or reinitialize_generation_llm:
            self.generation_llm.initialize_generation_llm()

    def generate_rag_responses(self, queries_dataframe):
        queries_dataframe['Retrieved_Documents'] = self.retriever.search(queries_dataframe['Query'].tolist())
        queries_dataframe['LLM_Model_Prompts'] = queries_dataframe.progress_apply(lambda x: self.generation_llm.generate_llm_prompt(x["Query"], x["Retrieved_Documents"]), axis=1)
        queries_dataframe['LLM_Responses'] = queries_dataframe.progress_apply(lambda x: self.generation_llm.generate_llm_prompt(x["Query"], x["Retrieved_Documents"]), axis=1)
        return queries_dataframe


        

