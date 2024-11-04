from vllm import SamplingParams
from transformers import AutoTokenizer
from openai import OpenAI
import time

def generate_synthetic_query_vllm_approach(
    document: str, 
    synthetic_query_prompt: str, 
    prompt: str, 
    length_of_fewshot_prompt: int, 
    tokenizer: AutoTokenizer,
    model_choice: str,  
    host_url: str,
    percentiles: list, 
    for_fever_dataset: bool,
    for_wow_dataset: bool
):
    synthetic_queries = []
    max_context_length = tokenizer.model_max_length

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=host_url
    )

    prompt_without_document = prompt + "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    prompt_without_document += "Document:"

    full_prompt = prompt_without_document + document
    if for_fever_dataset:
        full_prompt += "\nStatement: "
    elif for_wow_dataset:
        full_prompt += "\nDialogue: "
    else:
        full_prompt += "\nQuestion: "

    total_tokens = len(tokenizer.encode(full_prompt))

    if total_tokens > 32000:
        zero_shot_prompt = f"Document: {document}\n"
        if for_fever_dataset:
            zero_shot_prompt += "Statement: "
        elif for_wow_dataset:
            zero_shot_prompt += "Dialogue: "
        else:
            zero_shot_prompt += "Question: "
        prompt_to_use = zero_shot_prompt
    else:
        prompt_to_use = full_prompt

    # Truncate if needed
    max_tokens = min(131072 - len(tokenizer.encode(synthetic_query_prompt)) - 256, max_context_length - len(tokenizer.encode(synthetic_query_prompt)) - 256)  # Reserve tokens for the prompt, completion, and some padding
    encoded_prompt = tokenizer.encode(prompt_to_use, truncation=True, max_length=max_tokens)
    truncated_prompt = tokenizer.decode(encoded_prompt)

    for percentile in percentiles:
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                chat_response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": synthetic_query_prompt},
                        {"role": "user", "content": truncated_prompt},
                    ],
                    temperature=percentile,
                    max_tokens=256
                )
                
                message_content = chat_response.choices[0].message.content
                synthetic_queries.append(message_content.strip())
                break  # If successful, break out of the retry loop
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying... (Attempt {retry_count + 1} of {max_retries})")
                    time.sleep(10)  # Wait for 1 second before retrying
                else:
                    print(f"Failed after {max_retries} attempts.")

    return synthetic_queries

def generate_synthetic_answer_vllm_approach(
    document: str, 
    question: str, 
    synthetic_answer_prompt: str, 
    prompt: str, 
    length_of_fewshot_prompt: int, 
    tokenizer: AutoTokenizer,
    model_choice: str,  
    host_url: str,
    for_fever_dataset: bool,
    for_wow_dataset: bool
):
    max_context_length = tokenizer.model_max_length

    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = host_url

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    prompt_without_document = prompt + "Example " + str(length_of_fewshot_prompt + 1) + ":\n"
    if for_fever_dataset:
        prompt_without_document += "Document: \nStatement: \nAnswer: "
    elif for_wow_dataset:
        prompt_without_document += "Document: \nDialogue: \nResponse: "
    else:
        prompt_without_document += "Document: \nQuestion: \nAnswer: "

    full_prompt = prompt_without_document + document + "\n"
    if for_fever_dataset:
        full_prompt += f"Statement: {question}\nAnswer: "
    elif for_wow_dataset:
        full_prompt += f"Dialogue: {question}\nResponse: "
    else:
        full_prompt += f"Question: {question}\nAnswer: "

    total_tokens = len(tokenizer.encode(full_prompt))

    if total_tokens > 32000:
        zero_shot_prompt = f"Document: {document}\n"
        if for_fever_dataset:
            zero_shot_prompt += f"Statement: {question}\nAnswer: "
        elif for_wow_dataset:
            zero_shot_prompt += f"Dialogue: {question}\nResponse: "
        else:
            zero_shot_prompt += f"Question: {question}\nAnswer: "
        prompt_to_use = zero_shot_prompt
    else:
        prompt_to_use = full_prompt

    # Truncate if needed
    max_tokens = min(131072 - len(tokenizer.encode(synthetic_answer_prompt)) - 256, max_context_length - len(tokenizer.encode(synthetic_answer_prompt)) - 256)  # Reserve tokens for the prompt, completion, and some padding
    encoded_prompt = tokenizer.encode(prompt_to_use, truncation=True, max_length=max_tokens)
    truncated_prompt = tokenizer.decode(encoded_prompt)

    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            chat_response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": synthetic_answer_prompt},
                    {"role": "user", "content": truncated_prompt},
                ],
                max_tokens=256
            )
            
            message_content = chat_response.choices[0].message.content
            return message_content.strip()
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... (Attempt {retry_count + 1} of {max_retries})")
                time.sleep(10)  # Wait for 1 second before retrying
            else:
                print(f"Failed after {max_retries} attempts.")
                return ""