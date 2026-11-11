
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Project folders
BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "dataset" / "documents"
DB_DIR = BASE_DIR / "chroma_db"
EMBEDDINGS_FILE = BASE_DIR / "dataset" / "embeddings.npy"

# Load text documents
document_files = sorted(DOC_DIR.glob("*.txt"))

if not document_files:
    raise FileNotFoundError("No text documents found!")

documents = []
document_ids = []
metadatas = []

for file_path in document_files:
    content = file_path.read_text(encoding="utf-8-sig").strip()

    if not content:
        continue

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    title_line = next(
        (line for line in lines if line.lower().startswith("title:")),
        None
    )

    category_line = next(
        (line for line in lines if line.lower().startswith("category:")),
        None
    )

    title = (
        title_line.split(":", 1)[1].strip()
        if title_line
        else lines[0]
    )

    category = (
        category_line.split(":", 1)[1].strip()
        if category_line
        else "Artificial Intelligence and Data Science"
    )

    documents.append(content)
    document_ids.append(file_path.stem)

    metadatas.append({
        "title": title,
        "category": category,
        "file_name": file_path.name
    })

if not documents:
    raise ValueError("All documents are empty!")

print("Documents loaded:", len(documents))

# Load Sentence Transformer model
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Save embeddings as NumPy file
np.save(EMBEDDINGS_FILE, embeddings)

print("\nEmbeddings generated successfully!")
print("Embedding shape:", embeddings.shape)

# Create persistent ChromaDB storage
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="ai_data_science_documents",
    metadata={"hnsw:space": "cosine"}
)

# Store documents and embeddings
collection.upsert(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("\nDocuments stored successfully in ChromaDB!")
print("Total documents in ChromaDB:", collection.count())
print("Embeddings saved at:", EMBEDDINGS_FILE)
