
from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
QUERY_FILE = BASE_DIR / "dataset" / "queries.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "retrieved_results.csv"

# Load queries
queries_df = pd.read_csv(QUERY_FILE)

# Check required columns
if not {"query_id", "query"}.issubset(queries_df.columns):
    raise ValueError("queries.csv must contain query_id and query columns")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_collection(
    name="ai_data_science_documents"
)

print("Documents available:", collection.count())

# Retrieve top 5 documents for each query
results_list = []

for _, row in queries_df.iterrows():

    query_id = row["query_id"]
    query_text = str(row["query"]).strip()

    if not query_text:
        continue

    print(f"\nSearching for: {query_text}")

    # Generate query embedding
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(5, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    # Store retrieved results
    for rank, doc_id in enumerate(results["ids"][0], start=1):

        metadata = results["metadatas"][0][rank - 1]
        document = results["documents"][0][rank - 1]
        distance = results["distances"][0][rank - 1]

        similarity_score = 1 - distance

        results_list.append({
            "query_id": query_id,
            "query": query_text,
            "rank": rank,
            "document_id": doc_id,
            "title": metadata.get("title", ""),
            "category": metadata.get("category", ""),
            "similarity_score": round(similarity_score, 4),
            "retrieved_text": document
        })

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nSemantic search completed successfully!")
print("Total retrieved results:", len(results_df))
print("Results saved to:", OUTPUT_FILE)

print("\nTop results:")
print(results_df[
    ["query_id", "rank", "title", "similarity_score"]
].head(10).to_string(index=False))
