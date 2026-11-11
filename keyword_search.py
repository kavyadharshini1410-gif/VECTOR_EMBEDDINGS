
from pathlib import Path
import re
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "dataset" / "documents"
QUERY_FILE = BASE_DIR / "dataset" / "queries.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "keyword_results.csv"

# Common words to ignore
STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "how",
    "do", "does", "can", "to", "from", "of", "in",
    "for", "and", "why", "with"
}

# Convert text into useful keywords
def get_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return set(word for word in words if word not in STOP_WORDS)

# Load documents
documents = []

for file_path in sorted(DOC_DIR.glob("*.txt")):
    content = file_path.read_text(encoding="utf-8-sig").strip()

    if not content:
        continue

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    title_line = next(
        (line for line in lines if line.lower().startswith("title:")),
        None
    )

    title = (
        title_line.split(":", 1)[1].strip()
        if title_line else lines[0]
    )

    documents.append({
        "document_id": file_path.stem,
        "title": title,
        "text": content,
        "keywords": get_keywords(content)
    })

# Load queries
queries_df = pd.read_csv(QUERY_FILE)

results_list = []

print("Documents loaded:", len(documents))

# Search using keyword matching
for _, row in queries_df.iterrows():

    query_id = row["query_id"]
    query_text = str(row["query"]).strip()
    query_keywords = get_keywords(query_text)

    print("\nSearching for:", query_text)

    if not query_keywords:
        continue

    document_scores = []

    for document in documents:

        common_words = query_keywords.intersection(
            document["keywords"]
        )

        score = len(common_words) / len(query_keywords)

        document_scores.append({
            "query_id": query_id,
            "query": query_text,
            "document_id": document["document_id"],
            "title": document["title"],
            "keyword_score": round(score, 4),
            "retrieved_text": document["text"]
        })

    # Sort by keyword score
    document_scores.sort(
        key=lambda item: item["keyword_score"],
        reverse=True
    )

    # Store top 5 results
    for rank, result in enumerate(document_scores[:5], start=1):
        result["rank"] = rank
        results_list.append(result)

# Save results
results_df = pd.DataFrame(results_list)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nKeyword search completed successfully!")
print("Total retrieved results:", len(results_df))
print("Results saved to:", OUTPUT_FILE)

print("\nTop results:")
print(
    results_df[
        ["query_id", "rank", "title", "keyword_score"]
    ].head(10).to_string(index=False)
)
