# ARES

To implement LLM-as-a-Judge for scoring your RAG system and comparing to other RAG configurations, you need two components:
​
- A human-labeled set of query, document, and answer triples for the evaluation criteria (e.g. context relevance, answer faithfulness, and/or answer relevance). There should be at least 50 examples but several hundred examples is ideal.
- A much larger set of unlabeled query + document pairs
​
The LLM-as-a-Judge training pipeline is three separate steps:
​
- Generate synthetic training data for fine-tuning LLM-as-a-Judge
- Fine-tuning LLM-as-a-Judge with synthetic training data
- Using fine-tuned LLM-as-a-Judge to score RAG system
​
Note: Steps #1 and #2 can be skipped if you decide to go directly with zero/few-shot LLM-as-a-Judge
​
### Installation
​
To install the necessary dependencies, run the following commands:
​
````
conda create -n llm_judge python=3.10
conda activate llm_judge
pip install -r requirements.txt
````
​
Additionally, you will need to initialize an OpenAI API key with the following command:
````
export OPENAI_API_KEY=<your key here>
````
​
## Step #1: Synthetic Data Generation
​
To generate synthetic training data, use `LLM-as-a-Judge_Adaptation/Generate_Synthetic_Queries_and_Answers.py`. Replace items in the following command with your dataset and configuration:
​
````
python LLM-as-a-Judge_Adaptation/Generate_Synthetic_Queries_and_Answers.py \
       --document_filepath <document_filepath> \
       --few_shot_prompt_filename <few_shot_prompt_filename> \
       --answer_gen_few_shot_prompt_filename <answer_gen_few_shot_prompt_filename> \
       --synthetic_queries_filename <synthetic_queries_filename> \
       --few_shot_prompt_filename <few_shot_prompt_filename>
````
​
Note: For examples files for `document_filepath`, `few_shot_prompt_filename`, and `answer_gen_few_shot_prompt_filename`, please see `example_files`.
​
## Step #2: Fine-tune LLM-as-a-Judge
​
With the generated file under `synthetic_queries_filename` from the previous step, use `LLM-as-a-Judge_Adaptation/General_Binary_Classifier.py` to train your LLM-as-a-Judge with the following command:
​
````
python LLM-as-a-Judge_Adaptation/General_Binary_Classifier.py \
       --classification_datasets <classification_datasets as list> \
       --test_set_selection <test_set_selection> \
       --label_column Context_Relevance_Label \
       --num_epochs 10 \
       --patience_value 3 \
       --learning_rate_choices 5e-6
````
​
## Step #3: Score RAG System with LLM-as-a-Judge
​
With the outputted model checkpoint from Step #2, you can now score your RAG system using LLM-as-a-Judge with following command:
​
````
python LLM-as-a-Judge_Adaptation/LLMJudge_RAG_Compared_Scoring.py \
       --alpha 0.05 \
       --num_trials 1000 \
       --evaluation_datasets <evaluation_datasets as list> \
       --checkpoints <checkpoints as list> \
       --labels <label columns as list> \
       --GPT_scoring <True or False> \
       --gold_label_path <gold_label_path>
````
​
If you want to use GPT scoring, switch `GPT_scoring` to `True`. You can leave the `checkpoints` list as blank and specify the GPT model with the tag `--gpt_model <model selected>`.
​
Note: For examples files for `evaluation_datasets` and `gold_label_path`, please see `example_files`.
