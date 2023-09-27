
import pandas as pd
import numpy as np
import time
import openai
from tqdm import tqdm
import re

#################################################

def clean_document(document: str):
    cleaned_document = re.sub(r'\n+', '\n', document.replace("\r"," ").replace("\t"," ")).strip()
    cleaned_document = cleaned_document.replace("=", " ").replace("-", " ")
    cleaned_document = re.sub(r'\s+', ' ', cleaned_document).strip()
    cleaned_document = (" ").join(cleaned_document.split(" "))
    return cleaned_document

#################################################

def context_relevance(document: str, synth_question: str):
    #time.sleep(1)
    for _ in range(5):
        try:
            system_prompt = "Given the following question and document, you must analyze the provided document and determine whether it is sufficient for answering the question. "
            system_prompt += "In your evaluation, you should consider the content of the document and how it relates to the provided question. "
            #system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient and "[[No]]" if the document provided is not sufficient. '
            system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the document is sufficient, "[[No]]" if the document provided is not sufficient, and "[[Not sure]] if you are not sure.'
            system_prompt += "Do not provide any additional explanation for your decision."
            
            #document = document #(" ").join(clean_document(document).split(" ")[:256])
            #user_prompt = f"Question: {synth_question}\nContext: {document}"
            user_prompt = fewshot_examples_concat
            user_prompt += f"Example: {str(len(few_shot_prompt) + 1)}\n"
            user_prompt += f"Question: {synth_question}\n"
            user_prompt += f"Document: {document}\n"
            user_prompt += "Label: "

            #print("User Prompt")
            #print(user_prompt)
            #print("--------------------------------------------")
            
            messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        #model="gpt-4",
                        messages=messages
                    )    
            final_response = response["choices"][0]["message"]["content"]
            print("Question")
            print(synth_question)
            print("Document")
            print(document)
            print("Final Response")
            print(final_response)
            print('--------------------------------------------')
            #return final_response
            if "[[Yes]]" in final_response:
                return 1
            elif "[[No]]" in final_response:
                return 0
            else:
                print("Didn't extract Yes or No!")
                return 1

        except:

            print("Error querying OpenAI! Attempting again...")
            time.sleep(60)

#################################################

def answer_faithfulness(document: str, synth_question: str, answer: str):
    #time.sleep(1)
    for _ in range(5):
        try:
            system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and determine whether it is faithful to the contents of the document. "
            system_prompt += "The answer must not offer new information beyond the context provided in the document. "
            system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is faithful to the document and "[[No]]" if the answer is not faithful to the document. '
            system_prompt += "Do not provide any additional explanation for your decision."
            
            user_prompt = f"Question: {synth_question}\nContext: {clean_document(document)}\nAnswer: {answer}"
            
            messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        messages=messages
                    )    
            final_response = response["choices"][0]["message"]["content"]
            #return final_response
            if "[[Yes]]" in final_response:
                return 1
            elif "[[No]]" in final_response:
                return 0
            else:
                assert False

        except:

            print("Error querying OpenAI! Attempting again...")
            time.sleep(60)

#################################################

def answer_relevance(document: str, synth_question: str, answer: str):
    #time.sleep(1)
    for _ in range(5):
        try:
            system_prompt = "Given the following question, document, and answer, you must analyze the provided answer and document before determining whether the answer is relevant for the provided question. "
            system_prompt += "In your evaluation, you should consider whether the answer addresses all aspects of the question and provides only correct information from the document for answering the question. "
            system_prompt += 'Output your final verdict by strictly following this format: "[[Yes]]" if the answer is relevant for the given question and "[[No]]" if the answer is not relevant for the given question. '
            system_prompt += "Do not provide any additional explanation for your decision."
            
            user_prompt = f"Question: {synth_question}\nContext: {clean_document(document)}\nAnswer: {answer}"
            
            messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        messages=messages
                    )    
            final_response = response["choices"][0]["message"]["content"]
            #return final_response
            if "[[Yes]]" in final_response:
                return 1
            elif "[[No]]" in final_response:
                return 0
            else:
                assert False

        except:

            print("Error querying OpenAI! Attempting again...")
            time.sleep(60)

#################################################

few_shot_prompt_filename = "../datasets_v2/few_shot_prompt_for_context_relevance.tsv"
few_shot_prompt = pd.read_csv(few_shot_prompt_filename, sep="\t")
#few_shot_prompt = few_shot_prompt[:3]

print("Few shot prompts - Examples")
print(len(few_shot_prompt))

fewshot_examples_concat = ""
for row in range(len(few_shot_prompt)):
    fewshot_examples_concat += "Example " + str(row + 1) +":\n"
    fewshot_examples_concat += "Question: " + few_shot_prompt.iloc[row]['Question'] + "\n"
    fewshot_examples_concat += "Document: " + few_shot_prompt.iloc[row]['Document'] + "\n"
    fewshot_examples_concat += "Label: " + few_shot_prompt.iloc[row]['Label'] + "\n\n"

#################################################

evaluation_dataset_filename = "../datasets_v2/qa_logs_with_logging_details.tsv"
reformatted_filename = "../datasets_v2/LLM_Judge_Test_Set_v2_for_Training_using_GPT-3.5-16K.tsv"

evaluation_dataset = pd.read_csv(evaluation_dataset_filename, sep="\t")
columns_to_remove = evaluation_dataset.columns
#evaluation_dataset = evaluation_dataset[evaluation_dataset['reaction'] != "thumb_up"]

evaluation_dataset['Reaction'] = evaluation_dataset['reaction']
evaluation_dataset['Sources'] = [source for source in evaluation_dataset['sources'].tolist()]
evaluation_dataset['Question'] = evaluation_dataset['question']
evaluation_dataset['Document'] = evaluation_dataset['retrieval_contexts_used']
evaluation_dataset['Answer'] = evaluation_dataset['new_answer']
#evaluation_dataset['Label'] = ["" for _ in evaluation_dataset['new_answer'].tolist()]
evaluation_dataset = evaluation_dataset[evaluation_dataset['Document'].str.len() > 1]
#evaluation_dataset = evaluation_dataset[evaluation_dataset['Sources'].str.len() > 5]
evaluation_dataset = evaluation_dataset[evaluation_dataset['similarity_score'] != -1]
evaluation_dataset = evaluation_dataset.drop_duplicates(subset=['Question'])
evaluation_dataset = evaluation_dataset.drop_duplicates(subset=['Answer'])

print("evaluation_dataset")
print(len(evaluation_dataset))
print(evaluation_dataset.head())

evaluation_dataset = evaluation_dataset.sample(n=len(evaluation_dataset), random_state=44)

tqdm.pandas(desc="Completing scoring...", total=evaluation_dataset.shape[0])
evaluation_dataset['Context_Relevance'] = evaluation_dataset.progress_apply(lambda x: context_relevance(x["Document"], x["Question"]), axis=1)
#evaluation_dataset['Answer_Faithfulness'] = evaluation_dataset.progress_apply(lambda x: answer_faithfulness(x["Document"], x["Question"], x['Answer']), axis=1)
#evaluation_dataset['Answer_Relevance'] = evaluation_dataset.progress_apply(lambda x: answer_relevance(x["Document"], x["Question"], x['Answer']), axis=1)

evaluation_dataset = evaluation_dataset.drop(columns=columns_to_remove)
evaluation_dataset.to_csv(reformatted_filename, index=False, sep="\t")
print("Saved: " + reformatted_filename)

######################################################################

