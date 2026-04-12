import os
import re
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
PERSIST_DIR = "./chroma_db_gemini"
DATABASE_URL = os.getenv("DATABASE_URL")

if not API_KEY:
    raise ValueError("❌ Thiếu GEMINI_API_KEY")
if not DATABASE_URL:
    raise ValueError("❌ Thiếu DATABASE_URL")

app = Flask(__name__)

# =============================================================
# ✅ LAZY GLOBALS — không có gì nặng chạy lúc import
# =============================================================
_genai_client = None
_model_name   = None
_embed_fn     = None
_chroma_client = None
_pdf_collections = None   # None = chưa load, {} = đã load nhưng rỗng

def get_genai_client():
    global _genai_client, _model_name
    if _genai_client is None:
        import google.genai as genai
        _genai_client = genai.Client(api_key=API_KEY)
        available = [m.name for m in _genai_client.models.list()]
        for candidate in ["gemini-2.0-flash", "models/gemini-2.0-flash",
                          "gemini-1.5-flash", "models/gemini-1.5-flash"]:
            if candidate in available:
                _model_name = candidate
                break
        if not _model_name and available:
            _model_name = available[0]
        print(f"✅ Gemini client sẵn sàng, model: {_model_name}")
    return _genai_client

def get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from sentence_transformers import SentenceTransformer
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
        print("⏳ Load SentenceTransformer...")
        _model = SentenceTransformer("dangvantuan/vietnamese-embedding")
        _model.max_seq_length = 256

        class _EmbedFn(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return _model.encode(input, convert_to_numpy=True).tolist()

        _embed_fn = _EmbedFn()
        print("✅ SentenceTransformer sẵn sàng.")
    return _embed_fn

def get_chroma_collections():
    global _chroma_client, _pdf_collections
    if _pdf_collections is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        embed_fn = get_embed_fn()
        _pdf_collections = {}
        try:
            for col in _chroma_client.list_collections():
                if col.name.startswith("pdf_"):
                    parts = col.name.split("_")
                    if len(parts) > 2 and len(parts[-1]) == 12 and parts[-1].isalnum():
                        name_parts = parts[1:-1]
                    else:
                        name_parts = parts[1:]
                    display_name = " ".join(name_parts).title() if name_parts else col.name
                    collection_obj = _chroma_client.get_collection(
                        name=col.name, embedding_function=embed_fn
                    )
                    _pdf_collections[col.name] = {
                        "name": display_name,
                        "collection": collection_obj
                    }
            print(f"✅ Đã load {len(_pdf_collections)} quy trình.")
        except Exception as e:
            print(f"⚠️ Lỗi load ChromaDB: {e}")
    return _pdf_collections

# =============================================================
# DATABASE
# =============================================================
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
    try:
        conn = psycopg2.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute(
            "INSERT INTO history (question, answer, timestamp, collection_name) VALUES (%s,%s,%s,%s)",
            (question, answer, datetime.now().isoformat(), collection_name)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Lỗi lưu history: {e}")

# Khởi tạo bảng ngay (chỉ là SQL nhẹ, không block)
try:
    init_history_db()
except Exception as e:
    print(f"⚠️ Không init được DB: {e}")

# =============================================================
# RAG + GEMINI
# =============================================================
def retrieve_with_metadata(question: str, collection_name: str, k=30):
    pdf_collections = get_chroma_collections()
    try:
        col = pdf_collections[collection_name]["collection"]
        results = col.query(query_texts=[question], n_results=k)
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results.get('distances', [[1.0]*len(documents)])[0]
        chunks = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            kw_score = sum(0.1 for w in question.lower().split()
                           if len(w) > 3 and w in doc.lower())
            chunks.append({
                "content": doc,
                "source": meta.get("source", "Không rõ nguồn"),
                "page": meta.get("page", "?"),
                "relevance_score": (1 - dist) + kw_score
            })
        chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        return chunks[:k]
    except Exception as e:
        print(f"❌ Lỗi truy vấn ChromaDB: {e}")
        return []

def format_citations(text: str) -> str:
    text = re.sub(r'\_\(Nguồn:\s*(.*?)\)\_',
                  r'<small class="citation-source">(Nguồn: \1)</small>', text)
    text = re.sub(r'\(Nguồn:\s*(.*?)\)',
                  r'<small class="citation-source">(Nguồn: \1)</small>', text)
    return text

def ask_gemini(question: str, collection_name: str):
    client = get_genai_client()
    if not _model_name:
        return "❌ Không tìm thấy model Gemini."

    chunks = retrieve_with_metadata(question, collection_name)
    if not chunks:
        return "❌ Không tìm thấy thông tin trong tài liệu."

    context = "\n\n---\n\n".join(
        f"[Đoạn {i} từ {c['source']}, trang {c['page']}]:\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    )

    ql = question.lower()
    if any(w in ql for w in ["trình tự", "các bước", "làm thế nào", "cách thức"]):
        q_type, instruction = "quy trình", "Liệt kê các bước theo thứ tự."
    elif any(w in ql for w in ["định nghĩa", "là gì", "khái niệm"]):
        q_type, instruction = "định nghĩa", "Trả lời ngắn gọn định nghĩa."
    elif any(w in ql for w in ["số liệu", "giá trị", "thông số"]):
        q_type, instruction = "số liệu", "Trích dẫn đúng số liệu, đơn vị."
    elif any(w in ql for w in ["tại sao", "lý do"]):
        q_type, instruction = "giải thích", "Giải thích nguyên nhân dựa trên tài liệu."
    elif any(w in ql for w in ["so sánh", "khác nhau"]):
        q_type, instruction = "so sánh", "So sánh rõ ràng."
    else:
        q_type, instruction = "chung", "Trả lời trực diện, đúng trọng tâm."

    prompt = f"""Bạn là chuyên gia phân tích tài liệu quy trình vận hành thủy điện.

**NHIỆM VỤ:** Liệt kê đầy đủ tất cả thông tin liên quan, không bỏ sót mục nào.

**QUY TẮC:**
1. Mỗi mục xuống dòng, dùng số thứ tự hoặc dấu gạch đầu dòng.
2. Sau mỗi mục trích nguồn: `_(Nguồn: tên file, trang X)_`
3. Cảnh báo quan trọng: in đậm **CẢNH BÁO**.

### LOẠI: {q_type} | HƯỚNG DẪN: {instruction}

### TÀI LIỆU:
{context}

### CÂU HỎI: {question}

### TRẢ LỜI:"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(model=_model_name, contents=prompt)
            return format_citations(response.text)
        except Exception as e:
            err = str(e)
            if ("429" in err or "503" in err) and attempt < 2:
                time.sleep((attempt + 1) * 2)
            else:
                return f"⚠️ Lỗi API: {err}"

# =============================================================
# ROUTES
# =============================================================
@app.route("/")
def home():
    pdf_collections = get_chroma_collections()
    if not pdf_collections:
        return ("<h2>Chưa có quy trình nào.</h2>"
                "<p>Hãy chạy <code>python build_db.py</code> trước.</p>")
    pdf_list = [{"id": k, "name": v["name"]} for k, v in pdf_collections.items()]
    return render_template("index.html", pdf_list=pdf_list)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data or "question" not in data or "collection_name" not in data:
            return jsonify({"answer": "Thiếu câu hỏi hoặc tên quy trình"})
        answer = ask_gemini(data["question"], data["collection_name"])
        save_question_answer(data["question"], answer, data["collection_name"])
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Lỗi server: {str(e)}"})

@app.route("/history")
def get_history():
    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor(cursor_factory=RealDictCursor)
    c.execute("SELECT id, question, answer, timestamp, collection_name "
              "FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)

@app.route("/history_html")
def history_html():
    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute("SELECT question, answer, timestamp, collection_name "
              "FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    html = """<html><head><title>Lịch sử</title><style>
    body{font-family:Arial;margin:20px;background:#f5f5f5;}
    h1{color:#1a73e8;}
    .entry{background:white;margin-bottom:20px;padding:15px;border-radius:8px;
           box-shadow:0 1px 3px rgba(0,0,0,.1);}
    .question{font-weight:bold;color:#1a73e8;}
    .answer{margin:10px 0;line-height:1.5;}
    .meta{color:#777;font-size:12px;border-top:1px solid #eee;padding-top:8px;margin-top:8px;}
    </style></head><body><h1>📜 Lịch sử câu hỏi</h1>"""
    for row in rows:
        html += (f"<div class='entry'>"
                 f"<div class='question'>❓ {row[0]}</div>"
                 f"<div class='answer'>{row[1]}</div>"
                 f"<div class='meta'>📂 {row[3]} | 🕒 {row[2]}</div>"
                 f"</div>")
    html += "</body></html>"
    return html

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)