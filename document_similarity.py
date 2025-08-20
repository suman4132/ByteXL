from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.", 
    "The bird sat on the tree.",
    "The fish swam in the pond.",
    "The cat chased the mouse.",
    "The dog barked at the cat.",
    "The bird flew over the house.",
    "The fish jumped out of the water.",
    "The cat and dog played together.",
    "The bird sang a beautiful song.",
    "The fish swam with the other fish.",
    "The cat slept on the couch.",
    "The dog wagged its tail.",
    "The bird built a nest in the tree.",
    "The fish swam in circles.",
    "The cat groomed itself.",
    "The dog chased its tail.",
    "The bird perched on the fence.",
    "The fish swam near the rocks.",
    "The cat watched the bird.",
    "The dog dug a hole in the yard.",
    "The bird flew away from the cat.",
    "The fish swam to the bottom of the pond.",
    "The cat caught a mouse in the garden.",
    "The dog played fetch with its owner.",
    "The bird chirped happily in the morning.",

]
query = "What did the cat do?"
embeddings = GoogleGenerativeAIEmbeddings(
    model ="models/embedding-001"
)
# Generate embeddings for the documents
doc_embeddings = embeddings.embed_documents(documents)
# Generate embedding for the query
query_embedding = embeddings.embed_query(query)
# Calculate cosine similarity between the query embedding and document embeddings
res = cosine_similarity([query_embedding], doc_embeddings)[0]
# Sort documents by similarity score
index,score = sorted(list(enumerate(res)), key=lambda x: x[1])[-1]
# Print the top 5 most similar documents
print(query)
print(documents[index])
# Print the similarity score
print(f"Similarity score: {score:.4f}")