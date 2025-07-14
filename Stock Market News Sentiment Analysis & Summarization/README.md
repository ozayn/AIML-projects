Built an AI-driven system to extract and summarize market sentiment from news articles using LLMs, Transformers, Prompt Engineering, and text preprocessing.

Please refer to [code/notebook.ipynb](code/notebook.ipynb) to view the code.

# Stock Market News – Sentiment Analysis & Summarization

Analyzing the sentiment of stock market news and summarizing key headlines using NLP and LLMs.  
**Azin Faghihi | May 2025**  
NLP & LLM | Great Learning AIML Course

---

## Project Summary

We built a two-part pipeline to classify stock-related news by sentiment and generate weekly summaries using LLMs.  
The sentiment model uses GloVe embeddings + a tuned Random Forest classifier.  
The summarization model uses a local LLM with prompt engineering.

---

## Problem

Our investment startup needs a system that can:
- Understand how news sentiment links to stock prices
- Summarize key positive/negative stories from weekly headlines
The goal is to support smarter, faster investment decisions.

---

## Approach

### 📊 Sentiment Analysis
- **EDA**:
  - Sentiment labels are imbalanced (~49% neutral)
  - Open, High, Low, Close prices are all highly correlated
  - Negative sentiment is generally tied to lower price values
  - Neutral news tends to be shorter and tied to higher trading volume

- **Preprocessing**:
  - Dataset size: 349 × 8, no nulls/duplicates
  - Added derived features: News Length, Volume in millions
  - Split data by date into Train / Validation / Test

- **Embedding Techniques**:
  - Word2Vec (size: 300)
  - GloVe (size: 100) ✅ (final model)
  - SentenceTransformer (size: 384)

- **Modeling**:
  - Multiple classifiers tested
  - Chose F1 as the evaluation metric (due to class imbalance)
  - Tuned hyperparameters (e.g., max depth, min samples split)
  - Final model: GloVe + Tuned Random Forest
    - Best F1 trade-off between train/validation
    - Low overfitting, solid test performance

---

### 🧠 Content Summarization
- Aggregated Date + News data to weekly granularity
- Used `llama_cpp` to run an LLM locally (LLAMA)
- Applied prompt engineering to extract top 3 positive and negative news each week
- Final output: summarized dataframe of weekly highlights

---

## Data

- 349 rows × 8 columns → aggregated to 18 weeks  
- Categorical: `Label` (sentiment)  
- Numerical: Open, High, Low, Close, Volume, News Length  
- No missing or duplicate entries  

---

## Final Models

- **Sentiment Model**:
  - Embedding: GloVe (100 dim)
  - Classifier: Tuned Random Forest
  - Metric: F1 (training, validation, test)
  - Result: Balanced performance, low variance

- **Summarization Model**:
  - Model: LLAMA (via `llama_cpp`)
  - Aggregation: weekly news summaries
  - Output: 3 positive + 3 negative news items per week

---

## Tools

- Python (NLTK, Scikit-learn, Gensim, Transformers, llama-cpp)  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## Note

This project was part of the Great Learning AIML program.  
**Proprietary content – not for public distribution.**
