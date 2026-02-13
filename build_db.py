"""One-time script to build the FAISS vector DB from PDF manuals."""
import os
import shutil
import hashlib
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE = os.path.dirname(os.path.abspath(__file__))
MANUAL_DIR = os.path.join(BASE, "manuals")
CACHE_DIR = os.path.join(BASE, "manual_cache")
DB_PATH = os.path.join(BASE, "manual_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300


def get_embeddings():
    try:
        print("Initializing Hugging Face embedding model...")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"WARNING: Hugging Face 임베딩 초기화 실패. 로컬 해시 임베딩으로 대체합니다. {e}")

        class LocalHashEmbeddings:
            def __init__(self, dim: int = 384):
                self.dim = dim

            def _embed_text(self, text: str):
                vec = np.zeros(self.dim, dtype=np.float32)
                for token in text.lower().split():
                    idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                    vec[idx] += 1.0
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()

            def embed_documents(self, texts):
                return [self._embed_text(t) for t in texts]

            def embed_query(self, text):
                return self._embed_text(text)

        return LocalHashEmbeddings()


def main():
    # Clean old data
    for d in [CACHE_DIR, DB_PATH]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(CACHE_DIR, exist_ok=True)

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

    for idx, f in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {f}")
        try:
            loader = PyPDFLoader(os.path.join(MANUAL_DIR, f))
            docs = loader.load_and_split(text_splitter)
        except Exception as e:
            print(f"  -> ERROR: PDF 파싱 실패 ({f}): {e}")
            continue

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
