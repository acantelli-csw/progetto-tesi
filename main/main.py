import embedding
import main.search as search


query = "Trova i documenti relativi a machine learning"

query_embedding = embedding.get_embedding(query)

top_docs = search.similarity_search(query_embedding)


# PSEUDO prompt construction
SYSTEM = "Sei un motore di ricerca che fornisce risposte basate sui documenti forniti."
CONTEXT = top_docs
USER = query



