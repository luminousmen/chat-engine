import os
import requests

import torch
from bs4 import BeautifulSoup
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Disable parallelism in the tokenizer to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42  # Seed for reproducibility
# Set seed for reproducibility
set_seed(SEED)

# Constants for the setup
NUM_DOCS = 2  # Number of documents to retrieve
MAX_NEW_TOKENS = 256  # Maximum tokens for the response generation
NUM_RETURN_SEQUENCES = 1  # Number of response sequences to generate
MAX_LENGTH = 1000  # Maximum length of the augmented query
# TOKENIZER_MODEL = "gpt2"  # For smaller/faster model you can try gpt2
TOKENIZER_MODEL = "meta-llama/Meta-Llama-3.1-8B"  # Model for tokenization
GENERATOR_MODEL = TOKENIZER_MODEL  # Model for generation

# Class for data retrieval
class Retriever:
    @staticmethod
    def get_stock_data(symbol):
        """
        Retrieve stock data for the given symbol using yfinance.
        """
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        news = stock.news if hasattr(stock, 'news') else []
        financials = stock.quarterly_financials
        return stock_info, news, financials

    @staticmethod
    def fetch_story_content(link):
        """
        Fetch the content of a story from a given link.
        """
        try:
            response = requests.get(link)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            return content
        except Exception as e:
            return ""

    @staticmethod
    def preprocess_documents(documents):
        """
        Preprocess the documents by fetching their content and creating a TF-IDF vectorizer.
        """
        texts = [Retriever.fetch_story_content(doc.get('link', '')) for doc in documents]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return vectorizer, tfidf_matrix, documents

    @staticmethod
    def retrieve_documents(query, vectorizer, tfidf_matrix, documents, k=3):
        """
        Retrieve the most relevant documents to the query using cosine similarity.
        """
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        related_docs_indices = similarities.argsort()[-k:][::-1]
        return [documents[i] for i in related_docs_indices]

# Class for query augmentation
class Augmenter:
    max_content_length = 200  # Maximum content length for document snippets

    @staticmethod
    def stock_info_to_text(stock_info):
        """
        Convert stock information dictionary to a formatted string.
        """
        return (
            f"Company: {stock_info.get('longName', 'N/A')}\n"
            f"Industry: {stock_info.get('industry', 'N/A')}\n"
            f"Sector: {stock_info.get('sector', 'N/A')}\n"
            f"Full Time Employees: {stock_info.get('fullTimeEmployees', 'N/A')}\n"
            f"Market Cap: {stock_info.get('marketCap', 'N/A')}\n"
            f"52 Week Range: {stock_info.get('fiftyTwoWeekLow', 'N/A')} - {stock_info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"Dividend Rate: {stock_info.get('dividendRate', 'N/A')}\n"
            f"Dividend Yield: {stock_info.get('dividendYield', 'N/A')}\n"
            f"Last Dividend Value: {stock_info.get('lastDividendValue', 'N/A')}\n"
            f"Last Dividend Date: {stock_info.get('lastDividendDate', 'N/A')}\n"
            f"5 year average dividend Yield: {stock_info.get('fiveYearAvgDividendYield', 'N/A')}\n"
            f"Previous Close Price: {stock_info.get('previousClose', 'N/A')}\n"
            f"Open Price: {stock_info.get('open', 'N/A')}\n"
            f"Day Low Price: {stock_info.get('dayLow', 'N/A')}\n"
            f"Day High Price: {stock_info.get('dayHigh', 'N/A')}\n"
            f"Current Price: {stock_info.get('currentPrice', 'N/A')}\n"
            f"Target Price Range: {stock_info.get('targetLowPrice', 'N/A')} - {stock_info.get('targetHighPrice', 'N/A')}\n"
            f"Beta: {stock_info.get('beta', 'N/A')}\n"
            f"Trailing PE: {stock_info.get('trailingPE', 'N/A')}\n"
            f"Forward PE: {stock_info.get('forwardPE', 'N/A')}\n"
            f"Profit Margins: {stock_info.get('profitMargins', 'N/A')}\n"
            f"Book Value: {stock_info.get('bookValue', 'N/A')}\n"
            f"P/B: {stock_info.get('priceToBook', 'N/A')}\n"
            f"Recommendation: {stock_info.get('recommendationKey', 'N/A')}\n"
            f"Current Ratio: {stock_info.get('currentRatio', 'N/A')}\n"
            f"Total Debt: {stock_info.get('totalDebt', 'N/A')}\n"
            f"Total Revenue: {stock_info.get('totalRevenue', 'N/A')}\n"
            f"ROA: {stock_info.get('returnOnAssets', 'N/A')}\n"
            f"ROE: {stock_info.get('returnOnEquity', 'N/A')}\n"
            f"FCF: {stock_info.get('freeCashflow', 'N/A')}\n"
        )

    @staticmethod
    def generate_context_docs(documents):
        """
        Generate context snippets from documents.
        """
        return "\n".join(
            [
                f"Document {i+1}: {doc.get('title', 'No Title')} - {Retriever.fetch_story_content(doc.get('link', ''))[:Augmenter.max_content_length]}"
                for i, doc in enumerate(documents)
            ]
        )

    @staticmethod
    def augment_query_with_documents(query, documents, stock_info, financials, max_length=MAX_LENGTH):
        """
        Augment the query with relevant documents and stock info.
        """
        context_docs = Augmenter.generate_context_docs(documents)
        stock_info_text = Augmenter.stock_info_to_text(stock_info)
        augmented_query = (
            f"{context_docs}\n{stock_info_text}\nAnswer the following question based on the above context:\n{query}\nAnswer:"
        )        
        # Truncate if length exceeds max_length
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
        inputs = tokenizer(augmented_query, return_tensors='pt')
        if inputs['input_ids'].size(1) > max_length:
            truncated_inputs = tokenizer.encode_plus(
                augmented_query, max_length=max_length, truncation=True, return_tensors='pt'
            )
            augmented_query = tokenizer.decode(
                truncated_inputs['input_ids'][0], 
                skip_special_tokens=True
            )
        
        return augmented_query

# Class for response generation
class Generator:
    def __init__(self, model_name=GENERATOR_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Check if MPS (Metal Performance Shaders) is available for faster computation
        if torch.backends.mps.is_available():
            print("Using MPS")
            self.device = torch.device("mps")
            self.model.to(self.device)
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

    def generate_response(self, augmented_query, max_new_tokens=MAX_NEW_TOKENS):
        """
        Generate a response based on the augmented query.
        """
        inputs = self.tokenizer(augmented_query, return_tensors='pt').to(self.device)
        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            num_return_sequences=NUM_RETURN_SEQUENCES
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split('Answer: ')[-1]

# Main chat engine class
class ChatEngine:
    def __init__(self, retriever, augmenter, generator, symbol="SPOT"):
        self.retriever = retriever
        self.augmenter = augmenter
        self.generator = generator
        self.symbol = symbol
        self.initialize_data()

    def initialize_data(self):
        """
        Initialize data by retrieving stock info, news, financials, and processing documents.
        """
        stock_info, news, financials = self.retriever.get_stock_data(self.symbol)
        vectorizer, tfidf_matrix, documents = self.retriever.preprocess_documents(news)
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.documents = documents
        self.financials = financials
        self.stock_info = stock_info

    def chat(self, query):
        """
        Process the user's query to generate a response.
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve_documents(
            query, 
            self.vectorizer, 
            self.tfidf_matrix, 
            self.documents, 
            k=NUM_DOCS
        )
        # Augment query with relevant documents and stock info
        augmented_query = self.augmenter.augment_query_with_documents(
            query, 
            retrieved_docs, 
            self.stock_info, 
            self.financials
        )
        # Generate and return the response
        response = self.generator.generate_response(augmented_query)
        return response

    def run_chat(self):
        """
        Run the chatbot to interact with the user.
        """
        print("Chatbot is ready! Type your questions below (type 'exit' to quit):")
        # Run chat loop
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            bot_response = self.chat(user_input)
            print(f"Bot: {bot_response}")

if __name__ == "__main__":
    # Initialize components
    retriever = Retriever()
    augmenter = Augmenter()
    generator = Generator()
    chat_engine = ChatEngine(retriever, augmenter, generator)

    # Run the chat engine
    chat_engine.run_chat()
