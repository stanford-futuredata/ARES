
import json
from tqdm import tqdm

jsonl_file_path = '../datasets_v2/combined_aws.split-mode-smart.jsonl'
save_file_path = "../datasets_v2/docs_aws_v2_smart_chunked.jsonl"

documents = []
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in tqdm(jsonl_file):
        data = json.loads(line)
        documents.append(data)

print("documents")
print(len(documents))
print(type(documents[0]))
print(documents[0].keys())

############################################################

processed_json = []
for doc in documents:
    processed_json.append({
        "text": doc['content'],
        "source": doc['url'],
        "type": doc['doc_type'],
    })

with open(save_file_path, 'w') as output_file:
    json.dump(processed_json, output_file)

print("Saved file to: " + save_file_path)

############################################################

import pandas as pd

dataset = "../datasets_v2/LLM_Judge_Test_Set_Human_Annotations.tsv"
annotations_dataset = pd.read_csv(dataset, sep="\t")

filtered_rows = []
for i in range(len(annotations_dataset)):
    positive_val = annotations_dataset.iloc[i]['Answer_Faithfulness_Label'] == 1 or annotations_dataset.iloc[i]['Answer_Relevance_Label'] == 1
    if not (annotations_dataset.iloc[i]['Context_Relevance_Label'] == 0 and positive_val):
        filtered_rows.append(True)
    else:
        filtered_rows.append(False)

annotations_dataset['annotated'] = filtered_rows
annotations_dataset = annotations_dataset[annotations_dataset['annotated'] == True]

annotations_dataset.to_csv("../datasets_v2/LLM_Judge_Test_Set_Human_Annotations_V1.1_Filtered.tsv", sep="\t")