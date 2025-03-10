from langchain.document_loaders import WebBaseLoader

# URL to extract data from
url = "https://brainlox.com/courses/category/technical"

# Load data using WebBaseLoader
loader = WebBaseLoader(url)
documents = loader.load()

# Print the extracted data
print(documents)