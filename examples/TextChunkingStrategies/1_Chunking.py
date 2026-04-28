from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Edit the values below to adapt the script to your environment.
SAMPLE_TEXT = """
Artificial Intelligence (AI) has revolutionized many industries.
Machine learning is a subset of AI.

Deep learning uses neural networks with multiple layers.
It has achieved breakthrough results in computer vision.

Natural Language Processing enables computers to understand human language.
"""  # replace with your own text to experiment with different chunking strategies

# Strategy 1 — Character-based splitter
CHAR_SEPARATOR = "\n\n"  # primary split character
CHAR_CHUNK_SIZE = 100    # max characters per chunk
CHAR_CHUNK_OVERLAP = 20  # overlap between consecutive chunks

# Strategy 2 — Recursive character splitter (recommended)
RECURSIVE_CHUNK_SIZE = 100    # max characters per chunk
RECURSIVE_CHUNK_OVERLAP = 20  # overlap between consecutive chunks

# Strategy 3 — Token-based splitter
TOKEN_CHUNK_SIZE = 50    # max tokens per chunk
TOKEN_CHUNK_OVERLAP = 10  # overlap in tokens
# ──────────────────────────────────────────────────────────────────────────────

# Strategy 1: Character-based splitting
char_splitter = CharacterTextSplitter(
    separator=CHAR_SEPARATOR,
    chunk_size=CHAR_CHUNK_SIZE,
    chunk_overlap=CHAR_CHUNK_OVERLAP,
    length_function=len,
)
char_chunks = char_splitter.split_text(SAMPLE_TEXT)

# Strategy 2: Recursive splitting (RECOMMENDED)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RECURSIVE_CHUNK_SIZE,
    chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(SAMPLE_TEXT)

# Strategy 3: Token-based splitting
token_splitter = TokenTextSplitter(
    chunk_size=TOKEN_CHUNK_SIZE,
    chunk_overlap=TOKEN_CHUNK_OVERLAP,
)
token_chunks = token_splitter.split_text(SAMPLE_TEXT)

print("Character-based chunks:", len(char_chunks))
print("Recursive chunks:", len(recursive_chunks))
print("Token-based chunks:", len(token_chunks))
