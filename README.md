# Text-Summarization-using-RAG-LLama3.0

- Text Summarixation on CNN News Dataset using RAG and LLama2 3.2 model.
## How to set up Access Token on Hugging Face for LLama
- Link : https://medium.com/@lucnguyen_61589/llama-2-using-huggingface-part-1-3a29fdbaa9ed



# DECODER ONLY MODELS
- Link : https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse

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

# Detailed Explanation of LLAMA Architecture (Based on the Components in the Image)

The LLAMA architecture builds on the traditional transformer model, introducing optimizations for efficiency and performance. Here's a detailed step-by-step explanation of its workflow and components:

---

## 1. **Input Embedding**
- Input tokens (e.g., words or subwords) are converted into high-dimensional vectors through an **embedding layer**.
- These embeddings represent the semantic meaning of the tokens in the input sequence.

---

## 2. **Rotary Positional Encoding (RoPE)**
- **Purpose**: Adds positional information to the embeddings to reflect token order in the sequence.
- **Mechanism**: 
  - Instead of sinusoidal encodings, LLAMA uses **Rotary Positional Encoding (RoPE)**.
  - RoPE rotates the query and key vectors in a way that preserves **relative positional relationships**, improving performance for long sequences.

---

## 3. **RMSNorm Before Attention**
- **RMSNorm (Root Mean Square Normalization)** is applied to the embeddings before entering the attention mechanism.
- **Why RMSNorm?**:
  - It scales the input vectors, stabilizing training.
  - It is computationally simpler than LayerNorm, improving efficiency.

---

## 4. **Self-Attention (Grouped Multi-Query Attention)**
- **Purpose**: Captures relationships and dependencies between tokens in the input sequence.
- **Key Features**:
  1. **Grouped Multi-Query Attention**:
     - Unlike traditional multi-head attention where each head has its own key-value pairs, **Grouped Attention** shares key-value pairs among groups of attention heads.
     - This reduces memory usage and speeds up inference.
  2. **Key-Value (KV) Cache**:
     - During decoding (e.g., in autoregressive tasks), previously computed keys and values are cached.
     - This avoids recomputing them for each step, significantly improving efficiency.
  3. **Q (Query), K (Key), V (Value)**:
     - Derived from the input embeddings.
     - The attention computation is:
       \[
       \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
       \]
       where \(d\) is the dimensionality of the query/key vectors.

---

## 5. **Feed-Forward Network (SwiGLU)**
- After attention, a **Feed-Forward Neural Network (FFNN)** is applied to transform the token representations.
- **SwiGLU (Switchable Gated Linear Unit)**:
  - Introduces a gating mechanism to improve expressiveness.
  - Formula:
    \[
    \text{SwiGLU}(x) = (\text{Linear}_1(x) \cdot \sigma(\text{Linear}_2(x)))
    \]
    where \(\sigma\) is the sigmoid activation function.
  - **Why SwiGLU?**: More efficient and effective than ReLU-based FFNNs used in traditional transformers.

---

## 6. **RMSNorm After Feed-Forward**
- RMSNorm is applied again after the feed-forward network to stabilize outputs and improve gradient flow.

---

## 7. **Residual Connections**
- **Purpose**: Ensures that the model can learn additional features without overwriting earlier learned representations.
- Residual connections are applied around the attention and feed-forward layers, preventing gradient vanishing and improving learning.

---

## 8. **Repetition of Layers (Nx)**
- The **attention** and **feed-forward** blocks (along with RMSNorm and residual connections) are repeated \(N\) times.
- The depth \(N\) enables the model to learn hierarchical representations of the input data.

---

## 9. **Output Layer**
- After \(N\) layers, the final token representations are passed through:
  1. **Linear Layer**: Maps the token representations to logits (one for each token in the vocabulary).
  2. **Softmax Layer**: Converts logits into probabilities for each token, producing the final output.

---

## Key Enhancements in LLAMA:
1. **Memory Optimization**:
   - Grouped Multi-Query Attention reduces the size of key-value matrices.
   - KV Cache avoids redundant computations during decoding.
2. **Efficiency**:
   - SwiGLU activation and RMSNorm reduce computational overhead.
3. **Scalability**:
   - RoPE improves handling of longer sequences.
   - The overall structure ensures that the model remains lightweight and efficient.

---

## End-to-End Workflow
1. **Input Tokens**: Converted to embeddings.
2. **Positional Encodings**: RoPE adds positional relationships.
3. **Attention & Feed-Forward**:
   - Tokens interact via self-attention to capture relationships.
   - Feed-forward layers refine token representations.
4. **Repetition**: Layers are stacked \(N\) times for hierarchical learning.
5. **Output**: Final probabilities for each token are computed via the linear and softmax layers.

---

### **Conclusion**
The LLAMA architecture is an optimized transformer variant that balances efficiency, scalability, and performance. It achieves better memory usage and computational efficiency without sacrificing the model's expressiveness or accuracy.


# Retrieval-Augmented Generation (RAG)

- RAG Basics : https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c
- RAG Implementation Basics : https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7
- Lost in the middle phenomenon in LLMS : https://the-decoder.com/large-language-models-and-the-lost-middle-phenomenon/

# Vector Database vs Graph Database for RAG (Retrieval-Augmented Generation)

## 1. Information Extraction
### **Vector Database:**
- Text data is divided into chunks (smaller sections or paragraphs).
- An LLM (Large Language Model) encodes these chunks into **vector embeddings** (numerical representations that capture the meaning of the text).

### **Graph Database:**
- Focuses on extracting **relational information** (connections between entities like subjects, objects, and actions) instead of embeddings.
- Creates a **knowledge graph** where relationships between entities are explicitly represented.

---

## 2. Information Indexing
### **Vector Database:**
- Encoded vectors are stored in a **vector database**.
- These vectors are indexed based on their mathematical proximity (e.g., cosine similarity).

### **Graph Database:**
- Stores the **knowledge graph**, where nodes represent entities and edges represent relationships.
- Supports queries about relationships between entities.

---

## 3. Information Retrieval
### **Vector Database:**
- A question is encoded into a **vector representation**.
- The vector database finds the **nearest neighbors** (vectors most similar to the question).
- The closest text chunks are sent to the LLM as context to generate an answer.

### **Graph Database:**
- For questions about a specific entity or topic (e.g., "What is [X]?"), a **subgraph** of related entities and relationships is retrieved.
- This subgraph is used as context for the LLM to generate the response.

---

## Key Difference
- **Vector Database:** Focuses on finding **text chunks** that are semantically similar to the question.
- **Graph Database:** Focuses on retrieving **connected relationships** and entities relevant to the question.

Both methods aim to provide useful context to the LLM for generating high-quality responses.


# Graph Databases vs. Vector Databases in RAG

Graph Databases are favored for Retrieval Augmented Generation (RAG) when compared to Vector Databases. While Vector Databases partition and index data using LLM-encoded vectors, allowing for semantically similar vector retrieval, they may fetch irrelevant data.  
Graph Databases, on the other hand, build a knowledge base from extracted entity relationships in the text, making retrievals concise. However, it requires exact query matching which can be limiting.  
A potential solution could be to combine the strengths of both databases: indexing parsed entity relationships with vector representations in a graph database for more flexible information retrieval. It remains to be seen if such a hybrid model exists.  
After retrieving, you may want to look into filtering the candidates further by adding ranking and/or fine ranking layers that allow you to filter down candidates that do not match your business rules, are not personalized for the user, current context, or response limit.

## Process of RAG

1. **Vector Database Creation**:  
   RAG starts by converting an internal dataset into vectors and storing them in a vector database (or a database of your choosing).

2. **User Input**:  
   A user provides a query in natural language, seeking an answer or completion.

3. **Information Retrieval**:  
   The retrieval mechanism scans the vector database to identify segments that are semantically similar to the user’s query (which is also embedded). These segments are then given to the LLM to enrich its context for generating responses.

4. **Combining Data**:  
   The chosen data segments from the database are combined with the user’s initial query, creating an expanded prompt.

5. **Generating Text**:  
   The enlarged prompt, filled with added context, is then given to the LLM, which crafts the final, context-aware response.
