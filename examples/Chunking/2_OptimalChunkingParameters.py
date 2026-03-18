from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load a single text file
base_dir = Path(__file__).resolve().parents[1]
file_path = base_dir / "data" / "TheFrenchRevolution.txt"
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()


def create_optimized_splitter(use_case: str):
    """
    Return optimized splitter based on use case
    """
    configs = {
        "general": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "code": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "separators": ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "],
        },
        "qa": {
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        "long_context": {
            "chunk_size": 2000,
            "chunk_overlap": 400,
        }
    }

    config = configs.get(use_case, configs["general"])

    return RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        separators=config.get("separators", ["\n\n", "\n", " ", ""]),
    )


# Usage
splitter = create_optimized_splitter("qa")
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
print(f"Sample chunk content: {chunks[0].page_content[:200]}")
