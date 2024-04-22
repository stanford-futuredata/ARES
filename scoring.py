from ares import ARES

ues_idp_config = {
    # Dataset for in-domain prompts
    "in_domain_prompts_dataset": "/future/u/manihani/ARES/data/datasets/hotpotqa_few_shot_prompt_v1.tsv",
    
    # Dataset for unlabeled evaluation
    "unlabeled_evaluation_set": "/future/u/manihani/ARES/data/datasets_v2/hotpotqa/hotpotqa_ratio_0.5.tsv", 

    "context_relevance_system_prompt": """You are an expert dialogue agent. Your task is to analyze the provided document and determine whether 
    it is relevant for responding to the dialogue. Consider the content of the document and its relation to the provided dialogue. 
    Output your final verdict in the format: "[[Yes]]" if the document is relevant, and "[[No]]" if the document provided is not relevant. 
    Strictly adhere to this response format, your response must either be "[[Yes]]" or [["No"]], and feel free to elaborate on your response.""",

    "answer_relevance_system_prompt": """Given a question, a document, and an answer, analyze both the provided answer and document to determine if the answer is relevant to the question. 
    Evaluate whether the answer addresses all aspects of the question and relies solely on correct information from the document for its response. 
    Output your final verdict in the format: "[[Yes]]" if the answer is relevant to the given question, and "[[No]]" if the answer is not relevant. 
    Maintain strict adherence to this response format, your response must either be "[[Yes]]" or [["No"]], and feel free to elaborate on your response.""",
    
    "answer_faithfulness_system_prompt": """Given a question, a document, and an answer, assess whether the provided answer is faithful to the document's contents. 
    The answer must neither introduce information beyond what is contained in the document nor contradict any document information. 
    Output your verdict in the format: "[[Yes]]" if the answer is faithful to the document, and "[[No]]" if the answer is unfaithful. 
    Ensure strict adherence to this response format, your response must either be "[[Yes]]" or [["No"]], and feel free to elaborate on your response.""",

    "debug_mode": True,
    
    "documents": 2000, 
    
    "model_choice": "meta-llama/Llama-2-70b-chat-hf"
}

ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()
print(results)