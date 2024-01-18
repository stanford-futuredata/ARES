import subprocess 
from Evaluation_Functions import few_shot_answer_faithfulness_scoring 
from Evaluation_Functions import few_shot_answer_relevance_scoring
from Evaluation_Functions import few_shot_context_relevance_scoring
from Evaluation_Functions import calculate_accuracy
import pandas as pd

class ARES: 
    def __init__(self, in_domain_prompts_dataset, unlabeled_evaluation_set, gpt_model=None):
        #self.document_filepath = document_filepath 
        self.in_domain_prompts_dataset = pd.read_csv(in_domain_prompts_dataset, sep='\t')
        self.unlabeled_evaluation_set = pd.read_csv(unlabeled_evaluation_set, sep='\t')
        #self.gold_label_path = gold_label_path 
        #self.test_set_selection = test_set_selection 
        self.gpt_model = "gpt-3.5-turbo-1106"
        #####
        context_relevance_system_prompt = "You are an expert dialogue agent. "
        context_relevance_system_prompt += "Your task is to analyze the provided document and determine whether it is relevant for responding to the dialogue. "
        context_relevance_system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
        context_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
        context_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
        self.context_relevance_system_prompt = context_relevance_system_prompt

        #####
        answer_relevance_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
        answer_relevance_system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
        answer_relevance_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
        answer_relevance_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
        self.answer_relevance_system_prompt = answer_relevance_system_prompt
        #####
        answer_faithfulness_system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
        answer_faithfulness_system_prompt += "The answer must not offer new information beyond the context provided in the document. "
        answer_faithfulness_system_prompt += "The answer also must not contradict information provided in the document. "
        answer_faithfulness_system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
        answer_faithfulness_system_prompt += "Do not provide any additional explanation for your decision.\n\n"
        self.answer_faithfulness_system_prompt = answer_faithfulness_system_prompt
        #####

##############################################################################################################################
    
    def run(self): 
        context_relevance_answers = self.unlabeled_evaluation_set["Context_Relevance_Label"].tolist()
        answer_relevance_answers = self.unlabeled_evaluation_set["Answer_Relevance_Label"].tolist()
        answer_faithfulness_answers = self.unlabeled_evaluation_set["Answer_Faithfulness_Label"].tolist()
        context_relevance_scores = []
        answer_relevance_scores = []
        answer_faithfulness_scores = []

        for index, row in self.unlabeled_evaluation_set.iterrows():
            # Extract query, document, and answer from the row
            query = row["Query"]
            document = row["Document"]
            answer = row["Answer"]

            # Scoring
            context_score = few_shot_context_relevance_scoring(
                self.context_relevance_system_prompt, query, document, self.gpt_model, self.in_domain_prompts_dataset)

            answer_relevance_score = few_shot_answer_relevance_scoring(
                self.answer_relevance_system_prompt, query, document, answer, self.gpt_model, self.in_domain_prompts_dataset)
            answer_faithfulness_score = few_shot_answer_faithfulness_scoring(self.answer_faithfulness_system_prompt, query, document, answer, self.gpt_model, self.in_domain_prompts_dataset)

                # Append scores to respective lists
            context_relevance_scores.append(context_score)
            answer_relevance_scores.append(answer_relevance_score)
            answer_faithfulness_scores.append(answer_faithfulness_score)

        # Compile results into a dictionary
        return {
            "Context Relevance Scores": round(sum(context_relevance_scores)/len(context_relevance_scores), 3),
            "Answer Faithfulness Scores": round(sum(answer_faithfulness_scores)/len(answer_faithfulness_answers), 3),
            "Answer Relevance Scores": round(sum(answer_relevance_scores)/len(answer_relevance_answers), 3)
        }


#     def run(self): 

#         context_relevance_score = few_shot_context_relevance_scoring(self.context_relevance_system_prompt,self.unlabeled_evaluation_set,self.unlabeled_evaluation_set, "gpt-3.5-turbo", self.in_domain_prompts_dataset)
#         answer_relevance_score = few_shot_answer_relevance_scoring(self.answer_relevance_system_prompt,self.unlabeled_evaluation_set,self.unlabeled_evaluation_set, "gpt-3.5-turbo",self.in_domain_prompts_dataset)
#         answer_faithfulness_score = few_shot_answer_faithfulness_scoring(self.answer_faithfulness_system_prompt,self.unlabeled_evaluation_set,self.unlabeled_evaluation_set, "gpt-3.5-turbop",self.in_domain_prompts_dataset)
    
#         return {
#                 "Context Relevance Scores": context_relevance_scores,
#                 "Answer Faithfulness Scores": answer_faithfulness_scores,
#                 "Answer Relevance Scores": answer_relevance_scores
#             }

# ares = ARES("/future/u/manihani/ARES/example_files/evaluation_datasets.tsv", "/future/u/manihani/ARES/example_files/few_shot_prompt_filename.tsv", gpt_model = "text-davinci-003")
# results = ares.run()
# print(results)


    # def run(self):
    #     # Generate Synthetic Data
    #     synthetic_queries_filename = "output/synthetic_queries.tsv"
    #     documents_sampled = "10000"
    #     self._generate_synthetic_data(synthetic_queries_filename, documents_sampled)

    #     # Fine-tune LLM-as-a-Judge
    #     model_checkpoint_path = self._fine_tune_llm(synthetic_queries_filename)

    #     # Score RAG System
    #     scores = self._score_rag_system(model_checkpoint_path)
    #     return scores

    # def generate_synthetic_data(self):
    #     # Set default values for parameters not provided
    #     synthetic_queries_filename = "output/synthetic_queries.tsv"

    #     # Synthetic data generation
    #     subprocess.run([
    #         "python", "LLM-as-a-Judge_Adaptation/Generate_Synthetic_Queries_and_Answers.py",
    #         "--document_filepath", self.in_domain_prompts_dataset,
    #         "--few_shot_prompt_filename", self.in_domain_prompts_dataset,
    #         "--synthetic_queries_filename", synthetic_queries_filename,
    #         "--documents_sampled", documents_sampled
    #     ])
    
    # def fine_tune_llm(self): 
    #     num_epochs = 10 
    #     patience_value = 