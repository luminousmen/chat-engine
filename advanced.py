import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Disable parallelism in the tokenizer to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42  # Seed for reproducibility
# Set seed for reproducibility
set_seed(SEED)

# Constants for the setup
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Model for embeddings
GENERATOR_MODEL = "meta-llama/Meta-Llama-3.1-8B"  # Model for generation
MAX_NEW_TOKENS = 256  # Maximum tokens for the response generation
NUM_RETURN_SEQUENCES = 1  # Number of response sequences to generate
FILE_PATH = "./docs/elon_musk_tweets.csv"  # Path to the document directory
DOCUMENT_REGEXP = "*.txt"  # Regular expression for matching text files
VECTOR_STORE_PATH = "faiss"  # Path for storing the vector store

class ChatEngine:
    def __init__(self):
        # Check if MPS (Metal Performance Shaders) is available for faster computation
        if torch.backends.mps.is_available():
            print("Using MPS")
            self.device = torch.device("mps")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        # Initialize embeddings using HuggingFace
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": str(self.device)}
        )

        # Initialize the language model (LLM) for text generation
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL).to(self.device)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            return_full_text=True,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            device=self.device,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # Create the vector store from text files if it doesn't exist
        if not os.path.exists(VECTOR_STORE_PATH):
            self.create_vector_store_from_text()

    def create_vector_store_from_text(self):
        """
        Create a vector store from text files in the specified directory.
        """
        # Load documents from the directory
        loader = CSVLoader(FILE_PATH)
        documents = loader.load()
        
        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter()
        texts = splitter.split_documents(documents)
        
        # Create a FAISS vector store from the documents
        vector_store = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
        # Save the vector store locally
        vector_store.save_local(VECTOR_STORE_PATH)

    def chat(self, query):
        """
        Process the user's query to generate a response.
        """
        # Load the vector store from the local path
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Create a retriever from the vector store
        retriever = vector_store.as_retriever()
        
        # Pull the retrieval-augmented generation (RAG) prompt from the hub
        prompt = hub.pull("rlm/rag-prompt")
        
        # Create a retrieval QA chain using the LLM, retriever, and prompt
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
            verbose=False
        )
        
        # Invoke the chain with the query and get the result
        result = chain.invoke({"query": query})
        
        # Extract and return the response from the result
        return result["result"].split("Answer: ")[-1]

    def run_chat(self):
        """
        Run the chatbot to interact with the user.
        """
        print("Chatbot is ready! Type your questions below (type 'exit' to quit):")
        # Run chat loop
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            answer = self.chat(query)
            print(f"Bot: {answer}")

if __name__ == "__main__":
    # Initialize the chat engine
    chat_engine = ChatEngine()
    
    # Run the chat engine
    chat_engine.run_chat()
