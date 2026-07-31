"""
Knowledge Graph Builder — Multi-Book Edition
=============================================

Extracts text from multiple mental health books (PDFs),
builds a SINGLE unified co-occurrence graph, and saves it for retrieval.

Each paragraph is tagged with its source book so results can show
which book the information came from.

Usage:
    python -m knowledge.ingest

Books:
    - OCD
    - ADHD
    - Autism
    - Dyslexia
    - Schizophrenia
"""

import re
import json
import os
from collections import defaultdict
from itertools import combinations

import pdfplumber


# ──────────────────────────────────────────────────────────────
# Stop Words
# ──────────────────────────────────────────────────────────────

ENGLISH_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}

# Domain stop words (common in academic books but not meaningful)
DOMAIN_STOP_WORDS = {
    "also", "may", "however", "although", "would", "could", "one",
    "two", "three", "four", "five", "first", "second", "third",
    "used", "using", "use", "many", "much", "often", "see",
    "noted", "found", "reported", "described", "suggest", "suggested",
    "chapter", "section", "figure", "table", "page", "vol",
    "new", "york", "press", "journal", "eds", "london", "university",
    "p1", "mrm", "ikj", "qc", "abe", "t1", "wu038",  # PDF artifacts
    "et", "al", "pp", "doi", "http", "https", "www", "com", "org",
    "copyright", "published", "wiley", "springer", "elsevier",
    "isbn", "edited", "editor", "editors", "handbook", "manual",
}

# Minimum word length to include in the graph
MIN_WORD_LENGTH = 3


# ──────────────────────────────────────────────────────────────
# Config — All 5 Books
# ──────────────────────────────────────────────────────────────

BOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "books")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Each book: (filename, display_name, skip_before, skip_after)
# skip_before = number of front-matter pages to skip (title, copyright, contents)
# skip_after  = page number after which to stop (references, index)
BOOKS = [
    ("ocd.pdf",      "OCD",            12,  430),
    ("adhd.pdf",     "ADHD",            0,   None),
    ("autism.pdf",   "Autism",          15,  None),
    ("dyslexia.pdf", "Dyslexia",       10,  None),
    ("sch.pdf",      "Schizophrenia",  15,  None),
]


# ──────────────────────────────────────────────────────────────
# Step 1: Extract text from a single PDF
# ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(
    pdf_path: str,
    skip_before: int = 10,
    skip_after: int | None = None,
) -> list[str]:
    """
    Extract text from each page of a PDF.
    Returns a list of page texts (one string per page).
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        end_page = skip_after if skip_after is not None else total

        for i, page in enumerate(pdf.pages):
            if i < skip_before or i >= end_page:
                continue

            text = page.extract_text()
            if text and len(text.strip()) > 50:
                pages.append(text)

    return pages


# ──────────────────────────────────────────────────────────────
# Step 2: Split into paragraphs
# ──────────────────────────────────────────────────────────────

def split_into_paragraphs(pages: list[str]) -> list[str]:
    """
    Split page texts into individual paragraphs.
    """
    paragraphs = []

    for page_text in pages:
        # Clean PDF artifacts (header/footer noise)
        cleaned = re.sub(r'P1:MRM.*?CharCount=\d+', '', page_text)
        cleaned = re.sub(r'WU038.*?\d{4}\s+\d{2}:\d{2}', '', cleaned)

        # Split by blank lines
        chunks = re.split(r'\n\s*\n', cleaned)

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 30:
                paragraphs.append(chunk)

    return paragraphs


# ──────────────────────────────────────────────────────────────
# Step 3: Clean and tokenize
# ──────────────────────────────────────────────────────────────

def clean_and_tokenize(text: str, stop_words: set) -> list[str]:
    """
    Clean a paragraph and return a list of meaningful words.
    """
    text = text.lower()

    # Fix PDF concatenation issues (camelCase splits)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower()

    # Remove punctuation, numbers, special characters
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and filter
    words = text.split()
    filtered = [
        w for w in words
        if w not in stop_words
        and w not in DOMAIN_STOP_WORDS
        and len(w) >= MIN_WORD_LENGTH
    ]

    return filtered


# ──────────────────────────────────────────────────────────────
# Step 4: Build co-occurrence graph from ALL books
# ──────────────────────────────────────────────────────────────

def build_cooccurrence_graph(
    paragraphs: list[dict],
    stop_words: set,
) -> tuple[dict, dict, dict]:
    """
    Build a unified word co-occurrence graph from all paragraphs across all books.

    Args:
        paragraphs: List of {"text": str, "source": str} dicts.
        stop_words: Set of stop words to filter out.

    Returns:
        graph:              {word1: {word2: frequency, ...}, ...}
        word_freq:          {word: total_count}
        word_to_paragraphs: {word: [paragraph_indices]}
    """
    graph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    word_freq: dict[str, int] = defaultdict(int)
    word_to_paragraphs: dict[str, list[int]] = defaultdict(list)

    for para_idx, para_data in enumerate(paragraphs):
        words = clean_and_tokenize(para_data["text"], stop_words)

        # Count word frequencies
        unique_words = set(words)
        for word in unique_words:
            word_freq[word] += 1
            word_to_paragraphs[word].append(para_idx)

        # Build co-occurrence edges (every pair of unique words)
        for w1, w2 in combinations(unique_words, 2):
            graph[w1][w2] += 1
            graph[w2][w1] += 1

    # Convert defaultdicts to regular dicts for JSON
    graph = {k: dict(v) for k, v in graph.items()}
    word_freq = dict(word_freq)
    word_to_paragraphs = {k: v for k, v in word_to_paragraphs.items()}

    print(f"  Graph nodes (unique words): {len(graph)}")
    print(f"  Graph edges (co-occurrences): {sum(len(v) for v in graph.values()) // 2}")

    return graph, word_freq, word_to_paragraphs


# ──────────────────────────────────────────────────────────────
# Step 5: Save everything
# ──────────────────────────────────────────────────────────────

def save_knowledge_graph(
    graph: dict,
    word_freq: dict,
    paragraphs: list[dict],
    word_to_paragraphs: dict,
    output_dir: str,
):
    """Save all data to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save graph
    with open(os.path.join(output_dir, "graph.json"), "w") as f:
        json.dump(graph, f)
    size_kb = os.path.getsize(os.path.join(output_dir, "graph.json")) / 1024
    print(f"  Saved graph.json ({size_kb:.0f} KB)")

    # Save word frequencies
    with open(os.path.join(output_dir, "word_freq.json"), "w") as f:
        json.dump(word_freq, f)
    print(f"  Saved word_freq.json")

    # Save paragraphs with source tags
    with open(os.path.join(output_dir, "paragraphs.json"), "w") as f:
        json.dump(paragraphs, f)
    print(f"  Saved paragraphs.json ({len(paragraphs)} paragraphs)")

    # Save word → paragraph index
    with open(os.path.join(output_dir, "word_to_paragraphs.json"), "w") as f:
        json.dump(word_to_paragraphs, f)
    print(f"  Saved word_to_paragraphs.json")


# ──────────────────────────────────────────────────────────────
# Step 6: Print top connections (verification)
# ──────────────────────────────────────────────────────────────

def print_top_connections(graph: dict, word_freq: dict, top_n: int = 15):
    """Print the most connected words and their top neighbors."""
    sorted_words = sorted(graph.items(), key=lambda x: sum(x[1].values()), reverse=True)

    print(f"\n  Top {top_n} most connected words:")
    print(f"  {'Word':<20} {'Freq':<8} {'Top 5 Neighbors (weight)'}")
    print(f"  {'─'*20} {'─'*8} {'─'*45}")

    for word, neighbors in sorted_words[:top_n]:
        top_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:5]
        neighbor_str = ", ".join(f"{n}({w})" for n, w in top_neighbors)
        print(f"  {word:<20} {word_freq.get(word, 0):<8} {neighbor_str}")


def print_book_stats(paragraphs: list[dict]):
    """Print per-book paragraph counts."""
    from collections import Counter
    source_counts = Counter(p["source"] for p in paragraphs)

    print(f"\n  Paragraphs per book:")
    print(f"  {'Book':<20} {'Paragraphs':<12}")
    print(f"  {'─'*20} {'─'*12}")
    for source, count in sorted(source_counts.items()):
        print(f"  {source:<20} {count:<12}")
    print(f"  {'─'*20} {'─'*12}")
    print(f"  {'TOTAL':<20} {len(paragraphs):<12}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("\nBuilding Unified Knowledge Graph from 5 books...")
    print("=" * 60)

    stop_words = ENGLISH_STOP_WORDS
    all_paragraphs = []  # List of {"text": str, "source": str}

    # ── Step 1 & 2: Extract and split each book ──
    for filename, book_name, skip_before, skip_after in BOOKS:
        pdf_path = os.path.join(BOOKS_DIR, filename)

        if not os.path.exists(pdf_path):
            print(f"\n  Skipping {book_name}: {filename} not found in {BOOKS_DIR}")
            continue

        print(f"\n[{book_name}] Extracting text from {filename}...")
        pages = extract_text_from_pdf(pdf_path, skip_before, skip_after)
        print(f"  Extracted {len(pages)} content pages")

        paragraphs = split_into_paragraphs(pages)
        print(f"  Split into {len(paragraphs)} paragraphs")

        # Tag each paragraph with its source book
        for para_text in paragraphs:
            all_paragraphs.append({
                "text": para_text,
                "source": book_name,
            })

    if not all_paragraphs:
        print("\nNo paragraphs extracted! Check that PDF files exist in data/books/")
        return

    # ── Stats ──
    print_book_stats(all_paragraphs)

    # ── Step 3 & 4: Build unified graph ──
    print(f"\n[Graph] Building unified co-occurrence graph from {len(all_paragraphs)} paragraphs...")
    graph, word_freq, word_to_paragraphs = build_cooccurrence_graph(
        all_paragraphs, stop_words
    )

    # ── Step 5: Save ──
    print("\n[Save] Saving to disk...")
    save_knowledge_graph(graph, word_freq, all_paragraphs, word_to_paragraphs, OUTPUT_DIR)

    # ── Step 6: Verify ──
    print("\n[Verify] Top connections in unified graph:")
    print_top_connections(graph, word_freq)

    print("\n" + "=" * 60)
    print("Unified knowledge graph built successfully!")
    print(f"   Files saved in: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
