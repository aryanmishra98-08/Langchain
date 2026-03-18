from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# Sample text
sample_text = """
Artificial Intelligence (AI) has revolutionized many industries.
Machine learning is a subset of AI.

Deep learning uses neural networks with multiple layers.
It has achieved breakthrough results in computer vision.

Natural Language Processing enables computers to understand human language.
"""

# Strategy 1: Character-based splitting
char_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
char_chunks = char_splitter.split_text(sample_text)

# Strategy 2: Recursive splitting (RECOMMENDED)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(sample_text)

# Strategy 3: Token-based splitting
token_splitter = TokenTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)
token_chunks = token_splitter.split_text(sample_text)

print("Character-based chunks:", len(char_chunks))
print("Recursive chunks:", len(recursive_chunks))
print("Token-based chunks:", len(token_chunks))
