context_relevance_system_prompt = (
    "You are an expert dialogue agent. "
    "Your task is to analyze the provided document and determine whether it is relevant for responding to the dialogue. "
    "In your evaluation, you should consider the content of the document and how it relates to the provided dialogue. "
    'Output your final verdict by strictly following this format: "[[Yes]]" if the document is relevant and "[[No]]" if the document provided is not relevant. '
    "Do not provide any additional explanation for your decision.\n\n"
)

answer_relevance_system_prompt = (
    "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
    "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
    'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
    "Do not provide any additional explanation for your decision.\n\n"
)

answer_faithfulness_system_prompt = (
    "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
    "The answer must not offer new information beyond the context provided in the document. "
    "The answer also must not contradict information provided in the document. "
    'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
    "Do not provide any additional explanation for your decision.\n\n"
)