# Chat Engine

This repository contains two implementations of a chat engine:
1. **Stock-based Chat Engine**: A chatbot that retrieves and augments stock-related information to answer user queries.
2. **Advanced Text-based Chat Engine**: A chatbot that uses tweets retrieval and language models to answer general text-based queries.

## Disclaimer

This is not a production system. It is intended for educational purposes to understand the concept of Retrieval-Augmented Generation (RAG).

## Features

- **Stock-based Chat Engine**:
  - Retrieves stock information using yfinance.
  - Fetches and preprocesses news articles related to the stock.
  - Uses TF-IDF vectorizer and cosine similarity to retrieve relevant documents.
  - Augments queries with relevant documents and stock information.
  - Generates responses using a pre-trained language model.

- **Advanced Text-based Chat Engine**:
  - Loads and preprocesses tweets.
  - Creates and uses a FAISS vector store for efficient document retrieval.
  - Uses sentence-transformers for embeddings.
  - Generates responses using a pre-trained language model with retrieval-augmented generation (RAG) capabilities.

## Setup

1. Clone the Repository
2. Create a Virtual Environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install Dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Stock-based Chat Engine

1. **Run the Stock-based Chat Engine**:
   ```sh
   python stock.py
   ```

2. **Interaction**:
   - Type your questions related to stock information.
   - Type `exit` to quit the chatbot.

### Advanced Text-based Chat Engine

1. **Run the Advanced Text-based Chat Engine**:
   ```sh
   python advanced.py
   ```

2. **Interaction**:
   - Type your questions related to tweets.
   - Type `exit` to quit the chatbot.

## Blog Posts

For more detailed descriptions and explanations of these chat engines, please refer to my blog posts:
- [From RAGs to Riches: An In-Depth Look at Retrieval-Augmented Generation](https://luminousmen.com/post/from-rags-to-riches-an-indepth-look-at-retrievalaugmented-generation)
