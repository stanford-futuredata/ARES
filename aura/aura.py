
from RAG import RAG_Model

class AuRA:
    def __init__(self, rag_configs, ares, evaluation_metrics):
        
        assert type(rag_configs) == list 
        assert type(rag_configs[0]) == dict
        self.rag_configs = rag_configs
        self.ares = ares
        self.evaluation_metrics = evaluation_metrics

        self.rag_config_to_llm_responses = None
        self.model_ranking = None

    def find_best_rag_system(self, queries_dataframe, overwrite_model_ranking=False):
        
        if self.model_ranking is None or overwrite_model_ranking:
            
            model_to_ares_evaluation = {} 
            for rag_config in self.rag_configs:
                print(f"Evaluating {rag_config['rag_name']}")
                current_rag_model = RAG_Model(rag_config)
                current_rag_model.initialize_rag_model()
                queries_dataframe_with_rag_responses = current_rag_model.generate_rag_responses(queries_dataframe)
                ares_results_for_current_rag_model = self.ares(queries_dataframe_with_rag_responses)
                model_to_ares_evaluation[rag_config['rag_name']] = ares_results_for_current_rag_model
            
            self.model_ranking = model_to_ares_evaluation

        else:
            print("AuRA Already Completed! Set overwrite_model_ranking=True to repeat AuRA")

        ###################################################
            
        print("------------------------------------------------------")
        print("AuRA Results")
        print("------------------------------------------------------")
        for metric in self.evaluation_metrics:
            print(f"Model Ranking for {metric}:")
            scores = [(rag_config['rag_name'], self.model_ranking[rag_config['rag_name']][metric]) for rag_config in self.rag_configs]
            sorted_tuples_desc = sorted(scores, key=lambda x: x[1], reverse=True)
            for current_tuple in sorted_tuples_desc:
                print(f"{current_tuple[0]}: {current_tuple[0]}")
            print("------------------------------------------------------")


        