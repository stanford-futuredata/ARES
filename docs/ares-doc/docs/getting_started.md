# Introduction

Welcome to the Automated Evaluation Framework for Retrieval-Augmented Generation Systems (ARES). ARES is a groundbreaking framework for evaluating Retrieval-Augmented Generation (RAG) models. The automated process combines synthetic data generation with fine-tuned classifiers to efficiently assess context relevance, answer faithfulness, and answer relevance, minimizing the need for extensive human annotations. ARES employs synthetic query generation and prediction-powered inference (PPI), providing accurate evaluations with statistical confidence. 

<a href="https://arxiv.org/abs/2311.09476">
    <img alt="Static Badge" src="https://img.shields.io/badge/Read-ARES%20Paper-blue?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.09476">
</a>

## Get Started

<style>
.box-link {
    flex: 0 0 calc(50% - 10px); 
    padding: 20px;
    margin: 5px;
    background: #f0f0f0;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: box-shadow 0.3s ease-in-out, border 0.3s ease-in-out;
    text-decoration: none;
    color: inherit;
}
.container {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-start;
    text-align: left;
    gap: 15px;
}
.box {
    padding: 20px;
    margin-bottom: 10px; /* Adds space between rows */
    flex: 0 0 48%; /* Ensures that only two boxes will fit in one row */
    background: #f0f0f0;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: box-shadow 0.3s ease-in-out, border 0.3s ease-in-out;
    text-decoration: none;
    color: inherit;
}
.box:hover {
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    border-color: #333; /* Or any other color for the outline */
    border: 1px solid #333;
}
.box h3 {
    margin-top: 0;
}

.box p{
    color: black;
}
</style>

<div class="container">

<a href="installation.html" class="box">
    <h3 style="margin-top: 0px;">🚀 Quick Start</h3>
    <p>Set up and try out ARES efficiently with our quick start guide!</p>
</a>


<a href="synth_gen.html" class="box">
    <h3 style="margin-top: 0px;">💪  Synthetic Generation</h3>
    <p>Discover how to automatically create synthetic datasets that closely mimic real-world scenarios for robust RAG testing.</p>
</a>

<a href="training_classifier.html" class="box">
    <h3 style="margin-top: 0px;">📊 Training Classifier</h3>
    <p>Learn how to train high-precision classifiers to determine the relevance and faithfulness of RAG outputs</p>
</a>


<a href="rag_eval.html" class="box">
    <h3 style="margin-top: 0px;">⚙️ RAG Evaluation</h3>
    <p>Configure RAG model evaluation with ARES to accurately evaluate your model's performance.</p>
</a>


</div>