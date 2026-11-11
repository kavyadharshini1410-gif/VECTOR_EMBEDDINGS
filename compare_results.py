
from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"

SEMANTIC_FILE = DATA_DIR / "retrieved_results.csv"
KEYWORD_FILE = DATA_DIR / "keyword_results.csv"
OUTPUT_FILE = DATA_DIR / "search_comparison.csv"

# Load both search results
semantic_df = pd.read_csv(SEMANTIC_FILE)
keyword_df = pd.read_csv(KEYWORD_FILE)

# Get the top-ranked result for each query
semantic_top = (
    semantic_df[semantic_df["rank"] == 1]
    [["query_id", "query", "title", "similarity_score"]]
    .rename(columns={
        "title": "semantic_top_result",
        "similarity_score": "semantic_score"
    })
)

keyword_top = (
    keyword_df[keyword_df["rank"] == 1]
    [["query_id", "title", "keyword_score"]]
    .rename(columns={
        "title": "keyword_top_result",
        "keyword_score": "keyword_score"
    })
)

# Compare top results
comparison_df = pd.merge(
    semantic_top,
    keyword_top,
    on="query_id",
    how="inner"
)

# Check whether both methods found the same top result
comparison_df["same_top_result"] = (
    comparison_df["semantic_top_result"]
    == comparison_df["keyword_top_result"]
)

# Find the number of common documents in the top 5
semantic_top5 = (
    semantic_df.groupby("query_id")["document_id"]
    .apply(set).to_dict()
)

keyword_top5 = (
    keyword_df.groupby("query_id")["document_id"]
    .apply(set).to_dict()
)

comparison_df["common_top5_count"] = comparison_df[
    "query_id"
].apply(
    lambda q: len(
        semantic_top5.get(q, set())
        & keyword_top5.get(q, set())
    )
)

# Save comparison results
comparison_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Search comparison completed successfully!")
print("Total queries compared:", len(comparison_df))

print("\nComparison Results:")
print(
    comparison_df[
        [
            "query_id",
            "semantic_top_result",
            "keyword_top_result",
            "same_top_result",
            "common_top5_count"
        ]
    ].to_string(index=False)
)

print("\nSame top result count:",
      comparison_df["same_top_result"].sum())

print("Comparison saved to:", OUTPUT_FILE)
