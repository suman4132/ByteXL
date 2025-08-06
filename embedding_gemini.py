from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Single query embedding
vector = embeddings.embed_query("what is the capital of France?")
print(vector)

# Multiple documents embedding
documents = [
    "The capital of France is Paris.",
    "Paris is known for its art, fashion, and culture.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "France is located in Western Europe."
]
vectors = embeddings.embed_documents(documents)
print(vectors)