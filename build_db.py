import os
import re
import hashlib
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from sentence_transformers import SentenceTransformer

load_dotenv()

PDF_DIR = "pdfs"
PERSIST_DIR = "./chroma_db_gemini"


# ── Embedding ────────────────────────────────────────────────
class VietnameseEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        print("📥 Loading embedding model...")
        self.model = SentenceTransformer("dangvantuan/vietnamese-embedding")
        self.model.max_seq_length = 256
        print("✅ Model ready!")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(
            input,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()


# ── Collection name ─────────────────────────────────────────
def sanitize_collection_name(name: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    if not name or not name[0].isalnum():
        name = 'pdf_' + name
    return name[:480]


def get_collection_name(pdf_path: str) -> str:
    content = Path(pdf_path).read_bytes()
    hash_val = hashlib.md5(content).hexdigest()[:12]
    stem = Path(pdf_path).stem
    return f"pdf_{sanitize_collection_name(stem)}_{hash_val}"


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":

    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"📁 Created {PDF_DIR}")
        exit(0)

    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDF found.")
        exit(0)

    embed_fn = VietnameseEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )

    for pdf_file in sorted(pdf_files):
        col_name = get_collection_name(str(pdf_file))

        existing_cols = [c.name for c in chroma_client.list_collections()]
        if col_name in existing_cols:
            print(f"✅ Skip: {pdf_file.name}")
            continue

        print(f"\n🔄 Processing: {pdf_file.name}")

        try:
            loader = PDFPlumberLoader(str(pdf_file))
            docs = loader.load()
        except Exception as e:
            print(f"❌ PDF error: {e}")
            continue

        chunks = text_splitter.split_documents(docs)

        try:
            collection = chroma_client.create_collection(
                name=col_name,
                embedding_function=embed_fn,
            )

            # 🔥 ADD BATCH (QUAN TRỌNG)
            batch_size = 100

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]

                collection.add(
                    documents=[c.page_content for c in batch],
                    ids=[f"chunk_{i+j}" for j in range(len(batch))],
                    metadatas=[
                        {
                            "source": pdf_file.name,
                            "page": c.metadata.get("page", 0),
                        }
                        for c in batch
                    ],
                )

            print(f"✅ Added {len(chunks)} chunks")

        except Exception as e:
            print(f"❌ Chroma error: {e}")
            try:
                chroma_client.delete_collection(col_name)
            except:
                pass

    print("\n🎉 DONE!")