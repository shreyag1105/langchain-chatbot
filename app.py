from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI  # Updated import
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize Flask app
app = Flask(__name__)
api = Api(app)

# Load the vector store
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vector_store

# Initialize the QA chain
def initialize_qa_chain(vector_store):
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key="your-openai-api-key"),  # Replace with your OpenAI API key
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain

# Load the vector store and initialize the QA chain
vector_store = load_vector_store()
qa_chain = initialize_qa_chain(vector_store)

# Define the API resource
class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_input = data.get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        try:
            response = qa_chain.run(user_input)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Add the resource to the API
api.add_resource(Chatbot, "/chat")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)