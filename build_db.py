"""One-time script to build the FAISS vector DB from PDF manuals."""
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

BASE = os.path.dirname(os.path.abspath(__file__))
MANUAL_DIR = os.path.join(BASE, "manuals")
CACHE_DIR = os.path.join(BASE, "manual_cache")
DB_PATH = os.path.join(BASE, "manual_db")
EMBEDDING_MODEL = "models/text-embedding-004"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300


def main():
    api_key = os.environ.get("MY_API_KEY_GOOGLE")
    if not api_key:
        print("ERROR: MY_API_KEY_GOOGLE environment variable is not set.")
        print("Run: export MY_API_KEY_GOOGLE=your_key")
        return

    # Clean old data
    for d in [CACHE_DIR, DB_PATH]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

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

    for idx, f in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {f}")
        loader = PyPDFLoader(os.path.join(MANUAL_DIR, f))
        docs = loader.load_and_split(text_splitter)
        for doc in docs:
            doc.metadata["source"] = f
            if "page" in doc.metadata:
                doc.metadata["page"] += 1
        if docs:
            temp_db = FAISS.from_documents(docs, embeddings)
            temp_db.save_local(os.path.join(CACHE_DIR, f))
            print(f"  -> {len(docs)} chunks embedded")

    # Merge
    print("Merging databases...")
    valid_caches = [
        f for f in os.listdir(CACHE_DIR)
        if os.path.isdir(os.path.join(CACHE_DIR, f))
    ]
    if not valid_caches:
        print("No caches to merge")
        return

    base_db = FAISS.load_local(
        os.path.join(CACHE_DIR, valid_caches[0]), embeddings,
        allow_dangerous_deserialization=True,
    )
    for cache_name in valid_caches[1:]:
        sub_db = FAISS.load_local(
            os.path.join(CACHE_DIR, cache_name), embeddings,
            allow_dangerous_deserialization=True,
        )
        base_db.merge_from(sub_db)
    base_db.save_local(DB_PATH)
    print(f"Vector DB saved to {DB_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
