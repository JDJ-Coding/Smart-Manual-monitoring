"""One-time script to build the FAISS vector DB from PDF manuals."""
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE = os.path.dirname(os.path.abspath(__file__))
MANUAL_DIR = os.path.join(BASE, "manuals")
DB_PATH = os.path.join(BASE, "manual_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300


def get_embeddings():
    print("Initializing Hugging Face embedding model...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def main():
    # Clean old data
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    embeddings = get_embeddings()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    pdf_files = sorted([f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("No PDF files found in manuals/")
        return

    print(f"Found {len(pdf_files)} PDF file(s)")

    all_docs = []
    for idx, f in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {f}")
        try:
            loader = PyPDFLoader(os.path.join(MANUAL_DIR, f))
            docs = loader.load_and_split(text_splitter)
        except Exception as e:
            print(f"  -> ERROR: Failed to parse PDF ({f}): {e}")
            continue

        for doc in docs:
            doc.metadata["source"] = f
            if "page" in doc.metadata:
                doc.metadata["page"] += 1

        if docs:
            all_docs.extend(docs)
            print(f"  -> {len(docs)} chunks extracted")

    if not all_docs:
        print("No documents to process")
        return

    # Create vector DB from all documents at once
    print(f"\nCreating vector database with {len(all_docs)} chunks...")

    # Use pickle format to avoid Korean path encoding issues with FAISS C++ library
    import tempfile
    import pickle

    # Create FAISS index
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # Save using pickle to avoid encoding issues
    os.makedirs(DB_PATH, exist_ok=True)

    # Save FAISS index using pickle
    with open(os.path.join(DB_PATH, "index.pkl"), "wb") as f:
        pickle.dump(vectorstore.serialize_to_bytes(), f)

    print(f"Vector DB saved to {DB_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
