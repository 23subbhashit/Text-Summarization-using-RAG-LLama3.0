# Text-Summarization-using-RAG-LLama2

- Text Summarixation on CNN News Dataset using RAG and LLama2 3.2 model.
## How to set up Access Token on Hugging Face for LLama2
- Link : https://medium.com/@lucnguyen_61589/llama-2-using-huggingface-part-1-3a29fdbaa9ed


## Text Summarization with LLaMA and FAISS

### 1. Dataset Loading:
- The CNN/DailyMail dataset is loaded, specifically the training split (`split='train'` argument).
- This dataset consists of news articles paired with human-written summaries (highlights).
- The articles and summaries are extracted into two separate lists: `articles` and `summaries`.

### 2. Model Setup:
- The LLaMA model is used for text generation (summarization). LLaMA is a **decoder-only** model, which means it generates text based on a given prompt.
- The model used in this code is **"meta-llama/Llama-3.2-1B"**, which is a large version of the LLaMA model.
- **AutoTokenizer** is loaded to tokenize the input text, while the **AutoModelForCausalLM** is set up for text generation.

### 3. Embedding and Retrieval:
#### Sentence Embeddings:
- **SentenceTransformer** model (`'all-MiniLM-L6-v2'`) is used to convert the articles into **vector embeddings**. These embeddings represent the **semantic meaning** of the articles and are used for fast retrieval of relevant context.

#### FAISS Index:
- **FAISS** (Facebook AI Similarity Search) is used to build an index of the article embeddings. This index allows you to **retrieve similar articles** based on the input query (or article).

### 4. Retrieval Process:
- The function `retrieve()` takes a query (in this case, an article), computes its embedding, and searches the FAISS index to retrieve the top **k most similar articles**.
- These related articles are then used to provide **context** for generating the summary.

### 5. Summarization:
- The function `summarize_article()` uses the retrieved context (similar articles) to generate a summary.
- It first **concatenates the related articles** and uses this as additional context for summarization.
- The input prompt for the LLaMA model is prepared by combining the context with the original article and asking the model to generate a summary.
- The model generates the summary by predicting the next tokens (words) based on the context and article, using **beam search** (a technique for improving text generation quality) to find the best sequence of tokens.

### 6. Evaluation (ROUGE Score):
- After generating a summary, the **ROUGE score** is computed to evaluate the quality of the generated summary against the reference summary (i.e., the human-written summary).
- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) is a metric used for evaluating automatic summarization by comparing the overlap between **n-grams** (e.g., unigrams, bigrams) in the generated summary and the reference summary.
- The code uses `rouge_scorer` from the **rouge_score** library to calculate scores like:
- **ROUGE-1** (unigram overlap)
- **ROUGE-2** (bigram overlap)
- **ROUGE-L** (longest common subsequence)
