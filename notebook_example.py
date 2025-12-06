# Copy this code into a new cell in your rag.ipynb notebook

from embeddings_helper import get_embedding

# Example 1: Get embedding for a single document
doc_embedding = get_embedding("This is a sample document", input_type="document")

# Example 2: Get embedding for a query
query_embedding = get_embedding("What is this about?", input_type="query")

# Example 3: Get embeddings for multiple documents
docs = ["First document", "Second document", "Third document"]
multi_embeddings = get_embedding(docs, input_type="document")

print(f"\nDocument embedding dimension: {len(doc_embedding)}")
print(f"Query embedding dimension: {len(query_embedding)}")
print(f"Number of embeddings: {len(multi_embeddings)}")
