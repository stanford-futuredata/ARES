
import requests
import time
import pandas as pd
import ast
import json
import copy
import openai
from tqdm import tqdm
import csv
from datasets import Dataset
import torch
import numpy as np
import random
import pdb
from openai import OpenAI
import os

#################################################

class Retriever:
    def __init__(self, config):
        self.retrieval_model = config.get("retrieval_model")
        self.documents = config.get("documents")
        self.max_doc_length = config.get("max_doc_length")
        self.max_query_length = config.get("max_query_length")
        self.tokenizer = config.get("tokenizer")
        self.reranker = config.get("reranker")
        self.top_k = config.get("top_k")

        self.document_index = None
        self.initialized = False

    def initialize_retriever(self):
        raise NotImplementedError("initialize_retriever method must be implemented in subclasses")

    def create_embedding(self):
        raise NotImplementedError("create_embedding method must be implemented in subclasses")
    
    def create_index(self):
        raise NotImplementedError("create_index method must be implemented in subclasses")
    
    def update_index(self):
        raise NotImplementedError("update_index method must be implemented in subclasses")
    
    def rerank_docs(self):
        raise NotImplementedError("rerank_docs method must be implemented in subclasses")
    
    def search(self, queries):
        raise NotImplementedError("search method must be implemented in subclasses")

class OpenAI_Ada(Retriever):
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def initialize_retriever(self):
        print("Loading documents...")
        self.documents = pd.read_csv(self.documents, sep="\t")
        self.create_index()

    def create_embedding(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        try:
            return self.client.embeddings.create(input = [text], model=model).data[0].embedding
        except:
            print("Error generating embedding! Attempting again...")
            time.sleep(30)
    
    def create_index(self):

        self.document_index = self.documents.drop_duplicates(subset="Document")
        tqdm.pandas(desc="Generating embeddings...", total=self.document_index.shape[0])
        self.document_index['embeddings'] = self.document_index["Document"].progress_apply(lambda x: self.create_embedding(x, model=self.retrieval_model))
        self.document_index =  self.document_index[self.document_index['embeddings'].apply(lambda x: len(x)) == 1536]

        self.document_index = Dataset.from_pandas(self.document_index)
        self.document_index.add_faiss_index(column="embeddings")
    
    def update_index(self):
        print("Index updated using OpenAI Ada")

    def rerank_docs(self):
        print("Rerank docs using OpenAI Ada")
    
    def search(self, queries):
        
        assert type(queries) == list
        assert type(queries[0]) == str
        total_retrieved_docs = []
        for question in tqdm(queries):
            
            question_embedding = self.create_embedding(question)
            question_embedding = np.array(question_embedding).astype(np.float32)
            scores, samples = self.document_index.get_nearest_examples("embeddings", question_embedding, k=self.top_k)
            current_docs = [samples["document"][j] for j in range(len(samples))]
            assert type(current_docs[0]) == str
            total_retrieved_docs.append(current_docs)

        return total_retrieved_docs
