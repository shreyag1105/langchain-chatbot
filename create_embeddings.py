from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

# URL to extract data from
url = "https://brainlox.com/courses/category/technical"

# Load data using WebBaseLoader
loader = WebBaseLoader(url)
documents = loader.load()

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create embeddings and store in FAISS
vector_store = FAISS.from_documents(documents, embedding_model)

# Save the vector store locally
vector_store.save_local("faiss_index")
print("Embeddings created and stored in 'faiss_index'")