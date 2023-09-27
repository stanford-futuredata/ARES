
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq", index_name="exact", #use_dummy_dataset=True
)
# initialize with RagRetriever to do everything in one forward call
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
input_ids = inputs["input_ids"]
labels = targets["input_ids"]
outputs = model(input_ids=input_ids, labels=labels)
#outputs = model(input_ids=input_ids)

decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]



outputs_decoded = tokenizer.decode(outputs['logits'], skip_special_tokens=True)

print("outputs for retrieval + gen")
print(outputs)

# or use retriever separately
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
# 1. Encode
question_hidden_states = model.question_encoder(input_ids)[0]
# 2. Retrieve
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(
    question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
).squeeze(1)
# 3. Forward to generator
outputs = model(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
    decoder_input_ids=labels,
)

print("outputs for retrieval then gen")
print(outputs.keys())