import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader  # thay PyPDFLoader
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import hashlib
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer

load_dotenv()

PDF_DIR = "pdfs"
PERSIST_DIR = "./chroma_db_gemini"

# LỚP EMBEDDING CHUẨN (768 DIMENSIONS)
class VietnameseEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        print("📥 Đang khởi tạo model embedding tiếng Việt...")
        self.model = SentenceTransformer("dangvantuan/vietnamese-embedding")
        self.model.max_seq_length = 256
        print("✅ Model sẵn sàng!")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()

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

if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"📁 Đã tạo thư mục {PDF_DIR}. Hãy copy các file PDF vào đây.")
        exit(0)
    
    embed_fn = VietnameseEmbeddingFunction()
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # Tăng chunk_size và overlap để giữ ngữ cảnh
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", " ", ""]
    )

    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    if not pdf_files:
        print("⚠️ Không tìm thấy file PDF nào trong thư mục 'pdfs'.")
        exit(0)

    for pdf_file in sorted(pdf_files):
        col_name = get_collection_name(str(pdf_file))
        existing_cols = [c.name for c in chroma_client.list_collections()]
        
        if col_name in existing_cols:
            print(f"✅ Bỏ qua (đã tồn tại): {pdf_file.name}")
            continue

        print(f"🔄 Đang xử lý: {pdf_file.name}")
        loader = PDFPlumberLoader(str(pdf_file))
        docs = loader.load()
        if not docs:
            print(f"⚠️ Không đọc được nội dung PDF: {pdf_file.name}")
            continue
        
        print(f"   📄 Số trang: {len(docs)}")
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            print(f"⚠️ Không tạo được chunk nào từ {pdf_file.name}")
            continue
        
        print(f"   ✂️ Số chunk: {len(chunks)}")
        if chunks:
            print(f"   📝 Mẫu chunk 0: {chunks[0].page_content[:200]}...")

        # Tạo collection mới
        collection = chroma_client.create_collection(
            name=col_name, 
            embedding_function=embed_fn
        )
        
        # Thêm dữ liệu với metadata đầy đủ
        collection.add(
            documents=[c.page_content for c in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{
                "source": pdf_file.name,
                "page": c.metadata.get("page", 0),
                "chunk_id": i
            } for i, c in enumerate(chunks)]
        )
        print(f"   ✅ Đã thêm {len(chunks)} đoạn vào collection '{col_name}'")
    
    print("🎉 Hoàn tất xây dựng DB!")