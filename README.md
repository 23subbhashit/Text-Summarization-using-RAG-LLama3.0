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
The LLAMA architecture is a modern adaptation of the Transformer model, optimized for efficiency and scalability, especially for large-scale NLP tasks such as text generation and summarization. Below is a detailed breakdown:

---

## **1. Input Embedding**
- **Purpose**: Converts input tokens into dense, high-dimensional vectors that encode their semantic meaning.
- **How it Works**:
  - Each token is mapped to a vector using a learned embedding matrix of size \( V \times d \), where \( V \) is the vocabulary size, and \( d \) is the embedding dimension.
- **Significance**:
  - Captures semantic relationships between tokens.
  - The dimensionality \( d \) determines the richness of the representation.

---

## **2. Rotary Positional Encoding (RoPE)**
- **Purpose**: Encodes the relative position of tokens in the sequence.
- **Mechanism**:
  - RoPE applies a rotation matrix to the query (\( Q \)) and key (\( K \)) vectors.
  - Unlike absolute positional encodings, RoPE captures relative positions, which is crucial for processing long sequences.
- **Benefits**:
  - Improved handling of long sequences.
  - Enhanced representation of positional relationships, ideal for tasks requiring a wide context.

---

## **3. RMSNorm Before Attention**
- **Purpose**: Stabilizes input embeddings before attention computation.
- **Why RMSNorm**:
  - Root Mean Square Normalization normalizes the input vector by its root mean square.
  - It is computationally simpler and faster than LayerNorm, making it suitable for large models.

---

## **4. Self-Attention with Grouped Multi-Query Attention (GMQA)**
- **Purpose**: Enables the model to focus on relevant parts of the input sequence.

### **How Self-Attention Works**:
- Tokens are transformed into:
  - **Query (Q)**: What information is being requested.
  - **Key (K)**: What information is available.
  - **Value (V)**: The actual content.
- Attention scores are computed as:
  \[
  Attention(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right)V
  \]

### **Grouped Multi-Query Attention (GMQA)**:
- Traditional transformers compute separate \( K \)-\( V \) pairs for every attention head.
- **In LLAMA**:
  - Attention heads are grouped, and \( K \)-\( V \) pairs are shared across heads within each group.
  - **Example**:
    - For \( H = 16 \) heads grouped into \( G = 4 \) groups, \( K \)-\( V \) pairs are shared within each group, reducing the total number of \( K \)-\( V \) pairs to \( G = 4 \).
- **Benefits**:
  - Reduces memory usage and computation.
  - Retains flexibility as each head computes its own \( Q \).

---

## **5. Key-Value (KV) Cache**
- **Purpose**: Speeds up inference during autoregressive tasks like text generation.

### **Problem in Decoding**:
- At each decoding step, recomputing \( K \)-\( V \) pairs for all previous tokens is computationally expensive.

### **Solution: KV Caching**:
- Previously computed \( K \)-\( V \) pairs are cached.
- During decoding:
  - Only \( K \)-\( V \) pairs for the new token are computed.
  - These new pairs are appended to the cache, which is reused for subsequent attention calculations.
- **Benefits**:
  - Avoids redundant computations.
  - Improves decoding speed and reduces latency.

---

## **6. Feed-Forward Network (SwiGLU)**
- **Purpose**: Enhances token representations via non-linear transformations.
- **How SwiGLU Works**:
  - A gating mechanism selectively activates certain features:
    \[
    \text{SwiGLU}(x) = \text{Linear}_1(x) \cdot \sigma(\text{Linear}_2(x))
    \]
    where \( \sigma \) is the sigmoid function.
- **Benefits**:
  - Improves efficiency compared to ReLU-based FFNNs.
  - Enables better feature selection.

---

## **7. RMSNorm After Feed-Forward**
- **Purpose**: Normalizes outputs of the feed-forward layer to ensure stable training.
- **Significance**:
  - Prevents issues like exploding or vanishing gradients.
  - Ensures consistent activation ranges across layers.

---

## **8. Residual Connections**
- **Purpose**: Facilitates better information flow across layers by adding the input to the layer’s output.
- **Mechanism**:
  \[
  \text{Output} = \text{Layer}(x) + x
  \]
- **Benefits**:
  - Prevents gradient vanishing in deep networks.
  - Helps learn additional features without overwriting existing ones.

---

## **9. Repetition of Layers (Nx)**
- **Purpose**: Stacks multiple attention and feed-forward blocks to learn hierarchical patterns.
- **Workflow**:
  - Each block consists of:
    - Self-attention with normalization and residual connections.
    - Feed-forward with normalization and residual connections.
  - The number of blocks \( N \) defines the model depth.
- **Significance**:
  - Deep architectures enable learning of complex patterns and features.

---

## **10. Output Layer**
- **Purpose**: Converts the final token representations into probabilities.
- **Workflow**:
  - **Linear Layer**: Maps token representations to logits (raw scores).
  - **Softmax Layer**: Converts logits into probabilities that sum to 1.
- **Significance**:
  - The token with the highest probability is selected during text generation tasks.

---

## **Key Takeaways**
- **Grouped Multi-Query Attention** and **Key-Value Caching** are critical optimizations in LLAMA, reducing memory usage and improving efficiency during training and inference.
- These innovations make LLAMA highly suitable for large-scale NLP applications, offering a balance between computational efficiency and performance.



# Retrieval-Augmented Generation (RAG)

- RAG Basics : https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c
- RAG Implementation Basics : https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7
- Lost in the middle phenomenon in LLMS : https://the-decoder.com/large-language-models-and-the-lost-middle-phenomenon/
- RAG LLAMA3 CHROMADB : https://www.kaggle.com/code/gpreda/rag-using-llama3-langchain-and-chromadb
- RAG USING GPT : https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2

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

## What is Temperature in NLP?

**Temperature** is a parameter in natural language processing (NLP) models that controls the randomness of predictions in text generation tasks. It influences how creative or deterministic the model's outputs are.

---

### How Does Temperature Work?

The temperature adjusts the probabilities of the model's predictions during sampling. It scales the model's raw outputs (logits) before converting them into probabilities.

#### Formula for Adjusting Probabilities:
\[
p_i = \frac{\exp{(l_i / T)}}{\sum_{j}{\exp{(l_j / T)}}}
\]

Where:
- \( p_i \): Probability of the \(i\)-th token.
- \( l_i \): Logit (raw score) of the \(i\)-th token.
- \( T \): Temperature value.

---

### Effects of Temperature

1. **High Temperature (\( T > 1 \)):**
   - Smoothens the probability distribution, making less likely tokens more probable.
   - Increases randomness and creativity in the output.
   - Example: If \( T = 2 \), the model might produce more varied and imaginative responses.
   - **Risk:** The output may become incoherent or irrelevant.

2. **Low Temperature (\( T < 1 \)):**
   - Sharpens the probability distribution, favoring tokens with higher initial probabilities.
   - Produces more deterministic and predictable outputs.
   - Example: If \( T = 0.5 \), the model generates safe and precise responses.
   - **Risk:** The output may lack diversity or become repetitive.

3. **Temperature = 1:**
   - No scaling is applied; logits are converted into probabilities as-is.
   - Balances creativity and coherence.

---

### Choosing the Right Temperature

- **High Temperature (e.g., 1.5–2.0):**
  - Suitable for creative tasks like story or poem generation.

- **Low Temperature (e.g., 0.2–0.7):**
  - Ideal for tasks requiring precision and reliability, like question answering or summarization.

- **Default Temperature (1):**
  - A balanced approach, offering a mix of creativity and accuracy.

---

### Example:

#### Context:
"The Eiffel Tower is in Paris, France."

1. **With High Temperature (\( T = 1.5 \)):**
   - Output:  
     *"The Eiffel Tower, a famous landmark, stands tall under the Parisian sky, captivating millions of tourists every year."*

2. **With Low Temperature (\( T = 0.5 \)):**
   - Output:  
     *"The Eiffel Tower is in Paris, France."*

---

### Summary

- **High Temperature** = More creative, less predictable.
- **Low Temperature** = More focused, less diverse.
- **Use Case-Specific**: Adjust the temperature depending on whether you need creativity or precision.
