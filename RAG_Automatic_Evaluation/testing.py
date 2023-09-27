
from datasets import load_dataset

# Get the pre-processed Wikipedia knowledge source for kild
kilt_wiki = load_dataset("kilt_wikipedia")

# Get the KILT task datasets
kilt_triviaqa = load_dataset("kilt_tasks", name="triviaqa_support_only")

# Most tasks in KILT already have all required data, but KILT-TriviaQA
# only provides the question IDs, not the questions themselves.
# Thankfully, we can get the original TriviaQA data with:
trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')

# The KILT IDs can then be mapped to the TriviaQA questions with:
triviaqa_map = {}

def add_missing_data(x, trivia_qa_subset, triviaqa_map):
    i = triviaqa_map[x['id']]
    x['input'] = trivia_qa_subset[i]['question']
    x['output']['original_answer'] = trivia_qa_subset[i]['answer']['value']
    return x
    
for k in ['train', 'validation', 'test']:
    print("Starting on " + k)
    triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
    kilt_triviaqa[k] = kilt_triviaqa[k].filter(lambda x: x['id'] in triviaqa_map)
    kilt_triviaqa[k] = kilt_triviaqa[k].map(add_missing_data, fn_kwargs=dict(trivia_qa_subset=trivia_qa[k], triviaqa_map=triviaqa_map))

print("kilt_triviaqa")
print(kilt_triviaqa)