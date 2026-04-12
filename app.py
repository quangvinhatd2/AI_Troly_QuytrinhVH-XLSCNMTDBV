import os
import chromadb
import time
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

load_dotenv()

# --- CẤU HÌNH ---
API_KEY = os.getenv("GEMINI_API_KEY")
PERSIST_DIR = "./chroma_db_gemini"
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL URL từ Neon

if not API_KEY:
    raise ValueError("❌ Thiếu GEMINI_API_KEY trong file .env")
if not DATABASE_URL:
    raise ValueError("❌ Thiếu DATABASE_URL (PostgreSQL) trong file .env")

# Embedding function
class VietnameseEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("dangvantuan/vietnamese-embedding")
        self.model.max_seq_length = 256
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, convert_to_numpy=True).tolist()

embed_fn = VietnameseEmbeddingFunction()
client = genai.Client(api_key=API_KEY)

# Tìm model Gemini
available_models = [m.name for m in client.models.list()]
MODEL_NAME = None
for candidate in ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-2.0-flash"]:
    if candidate in available_models:
        MODEL_NAME = candidate
        break
if not MODEL_NAME and available_models:
    MODEL_NAME = available_models[0]
print(f"✅ Dùng model: {MODEL_NAME}")

# Kết nối ChromaDB
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

# Load collections
pdf_collections = {}
try:
    for col in chroma_client.list_collections():
        if col.name.startswith("pdf_"):
            parts = col.name.split("_")
            if len(parts) > 2 and len(parts[-1]) == 12 and parts[-1].isalnum():
                name_parts = parts[1:-1]
            else:
                name_parts = parts[1:]
            display_name = " ".join(name_parts).title() if name_parts else col.name
            collection_obj = chroma_client.get_collection(name=col.name, embedding_function=embed_fn)
            pdf_collections[col.name] = {"name": display_name, "collection": collection_obj}
    print(f"✅ Đã load {len(pdf_collections)} quy trình.")
except Exception as e:
    print(f"⚠️ Lỗi DB: {e}")

app = Flask(__name__)

# ==================== KHỞI TẠO CƠ SỞ DỮ LIỆU POSTGRESQL ====================
def init_history_db():
    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id SERIAL PRIMARY KEY,
                    question TEXT,
                    answer TEXT,
                    timestamp TEXT,
                    collection_name TEXT
                )''')
    conn.commit()
    conn.close()

def save_question_answer(question, answer, collection_name):
    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO history (question, answer, timestamp, collection_name) VALUES (%s, %s, %s, %s)",
              (question, answer, timestamp, collection_name))
    conn.commit()
    conn.close()

# Khởi tạo bảng
init_history_db()

# ==================== TRÍCH DẪN NOTEBOOKLM ====================
def retrieve_with_metadata(question: str, collection_name: str, k=30):
    try:
        col = pdf_collections[collection_name]["collection"]
        results = col.query(query_texts=[question], n_results=k)
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results else [1.0] * len(documents)
        
        chunks_with_meta = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            keyword_score = 0
            important_words = [w for w in question.lower().split() if len(w) > 3]
            for word in important_words:
                if word in doc.lower():
                    keyword_score += 0.1
            relevance = (1 - dist) + keyword_score
            
            chunks_with_meta.append({
                "content": doc,
                "source": meta.get("source", "Không rõ nguồn"),
                "page": meta.get("page", "?"),
                "chunk_id": meta.get("chunk_id", i),
                "relevance_score": relevance
            })
        chunks_with_meta.sort(key=lambda x: x['relevance_score'], reverse=True)
        return chunks_with_meta[:k]
    except Exception as e:
        print(f"❌ Lỗi truy vấn: {e}")
        return []

def format_response_with_small_citations(text: str) -> str:
    pattern = r'\_\(Nguồn:\s*(.*?)\)\_'
    text = re.sub(pattern, r'<small class="citation-source">(Nguồn: \1)</small>', text)
    pattern2 = r'\(Nguồn:\s*(.*?)\)'
    text = re.sub(pattern2, r'<small class="citation-source">(Nguồn: \1)</small>', text)
    return text

def ask_gemini_notebooklm(question: str, collection_name: str, response_level=3):
    if not MODEL_NAME:
        return "❌ Không tìm thấy model Gemini khả dụng."

    chunks = retrieve_with_metadata(question, collection_name, k=30)
    if not chunks:
        return "❌ Không tìm thấy thông tin trong tài liệu."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Đoạn {i} từ {chunk['source']}, trang {chunk['page']}]:\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    print(f"🔍 Số chunk lấy được: {len(chunks)}")
    print(f"🔍 Độ dài context: {len(context)} ký tự")

    # Phân loại câu hỏi
    question_lower = question.lower()
    if any(word in question_lower for word in ["trình tự", "các bước", "làm thế nào", "cách thức"]):
        q_type = "quy trình"
        instruction = "Liệt kê các bước theo thứ tự."
    elif any(word in question_lower for word in ["định nghĩa", "là gì", "khái niệm"]):
        q_type = "định nghĩa"
        instruction = "Trả lời ngắn gọn định nghĩa."
    elif any(word in question_lower for word in ["số liệu", "giá trị", "thông số"]):
        q_type = "số liệu"
        instruction = "Trích dẫn đúng số liệu, đơn vị."
    elif any(word in question_lower for word in ["tại sao", "lý do"]):
        q_type = "giải thích"
        instruction = "Giải thích nguyên nhân dựa trên tài liệu."
    elif any(word in question_lower for word in ["so sánh", "khác nhau"]):
        q_type = "so sánh"
        instruction = "So sánh rõ ràng."
    else:
        q_type = "chung"
        instruction = "Trả lời trực diện, đúng trọng tâm."

    prompt_content = f"""
Bạn là chuyên gia phân tích tài liệu quy trình vận hành thủy điện. Dưới đây là nhiều đoạn trích từ các tài liệu gốc.

**NHIỆM VỤ:** Trả lời câu hỏi bằng cách **LIỆT KÊ ĐẦY ĐỦ** tất cả các thông tin có liên quan. Đặc biệt với câu hỏi dạng liệt kê (ví dụ "các bảo vệ dừng máy"), bạn phải:
- Đọc kỹ từng đoạn trích.
- Trích xuất tất cả các mục, phân loại theo nhóm nếu có (bảo vệ điện, bảo vệ nhiệt, cơ khí...).
- Không bỏ sót bất kỳ mục nào, kể cả các mục nhỏ.

**QUY TẮC TRÌNH BÀY:**
1. Mỗi ý / mỗi bước / mỗi mục liệt kê đều xuống dòng, dùng số thứ tự hoặc dấu gạch đầu dòng.
2. Sau mỗi ý, trích dẫn nguồn với cú pháp: `_(Nguồn: tên file, trang X)_`
3. Nếu có cảnh báo, in đậm **CẢNH BÁO**.

### LOẠI CÂU HỎI: {q_type}
### HƯỚNG DẪN: {instruction}

### CÁC ĐOẠN TRÍCH TỪ TÀI LIỆU:
{context}

### CÂU HỎI: {question}

### TRẢ LỜI (liệt kê đầy đủ, không thiếu mục nào, không nói "không có thông tin" nếu tài liệu có):
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_content
            )
            raw_answer = response.text
            formatted_answer = format_response_with_small_citations(raw_answer)
            return formatted_answer
        except Exception as e:
            err_msg = str(e)
            if ("429" in err_msg or "503" in err_msg) and attempt < max_retries - 1:
                wait = (attempt + 1) * 2
                print(f"⚠️ Lỗi {err_msg}, thử lại sau {wait}s...")
                time.sleep(wait)
                continue
            return f"⚠️ Lỗi API: {err_msg}"

# ==================== ROUTES ====================
@app.route("/")
def home():
    if not pdf_collections:
        return """
        <h2>Chưa có quy trình nào trong database.</h2>
        <p>Hãy chạy <code>python build_db.py</code> trước, sau đó đưa file PDF vào thư mục <code>pdfs/</code>.</p>
        """
    pdf_list = [{"id": name, "name": info["name"]} for name, info in pdf_collections.items()]
    return render_template("index.html", pdf_list=pdf_list)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data or "question" not in data or "collection_name" not in data:
            return jsonify({"answer": "Thiếu câu hỏi hoặc tên quy trình"})
        answer = ask_gemini_notebooklm(data["question"], data["collection_name"])
        # Lưu lịch sử vào PostgreSQL
        save_question_answer(data["question"], answer, data["collection_name"])
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Lỗi server: {str(e)}"})

@app.route("/history", methods=["GET"])
def get_history():
    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor(cursor_factory=RealDictCursor)
    c.execute("SELECT id, question, answer, timestamp, collection_name FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)

@app.route("/history_html")
def history_html():
    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute("SELECT question, answer, timestamp, collection_name FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    html = """
    <html>
    <head><title>Lịch sử hỏi đáp</title>
    <style>
        body{font-family:Arial;margin:20px; background:#f5f5f5;}
        h1{color:#1a73e8;}
        .entry{background:white; margin-bottom:20px; padding:15px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.1);}
        .question{font-weight:bold; color:#1a73e8; margin-bottom:5px;}
        .answer{margin:10px 0; line-height:1.5;}
        .meta{color:#777; font-size:12px; border-top:1px solid #eee; padding-top:8px; margin-top:8px;}
    </style>
    </head>
    <body>
    <h1>📜 Lịch sử câu hỏi</h1>
    """
    for row in rows:
        html += f"""
        <div class='entry'>
            <div class='question'>❓ {row[0]}</div>
            <div class='answer'>{row[1]}</div>
            <div class='meta'>📂 {row[3]} | 🕒 {row[2]}</div>
        </div>
        """
    html += "</body></html>"
    return html

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)