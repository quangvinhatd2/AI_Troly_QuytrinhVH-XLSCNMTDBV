import os
import re
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash, send_from_directory)
from dotenv import load_dotenv

load_dotenv()

API_KEY        = os.getenv("GEMINI_API_KEY")
DATABASE_URL   = os.getenv("DATABASE_URL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SECRET_KEY     = os.getenv("SECRET_KEY", "change-me-please")
PERSIST_DIR    = "./chroma_db_gemini"
PDF_DIR        = "./pdfs"

if not API_KEY:
    raise ValueError("❌ Thiếu GEMINI_API_KEY")
if not DATABASE_URL:
    raise ValueError("❌ Thiếu DATABASE_URL")

app = Flask(__name__)
app.secret_key = SECRET_KEY

# =============================================================
# LAZY GLOBALS
# =============================================================
_genai_client    = None
_model_name      = None
_embed_fn        = None
_chroma_client   = None
_pdf_collections = None


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


def get_pdf_collections(force_refresh: bool = False):
    global _chroma_client, _pdf_collections
    if _pdf_collections is None or force_refresh:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        embed_fn = get_embed_fn()
        _pdf_collections = {}
        try:
            # Sửa lỗi: đảm bảo clear cũ nếu refresh
            current_cols = _chroma_client.list_collections()
            for col in current_cols:
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
                        "id": col.name, # Thêm ID để dễ xử lý trong template
                        "name": display_name,
                        "collection": collection_obj,
                    }
            print(f"✅ Đã load {len(_pdf_collections)} quy trình.")
        except Exception as e:
            print(f"⚠️ Lỗi load ChromaDB: {e}")
            _pdf_collections = {}
    return _pdf_collections


def get_pdf_files():
    if not os.path.exists(PDF_DIR):
        return []
    files = []
    for f in sorted(os.listdir(PDF_DIR)):
        if f.lower().endswith(".pdf"):
            size = os.path.getsize(os.path.join(PDF_DIR, f))
            files.append({
                "name": f,
                "size": f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024
                        else f"{size // 1024} KB"
            })
    return files


# =============================================================
# ADMIN AUTH - XỬ LÝ LỖI KHÔNG VÀO ĐƯỢC ADMIN
# =============================================================
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        # Sửa lỗi lấy mật khẩu: request.form.get("password")
        if request.form.get("password") == ADMIN_PASSWORD:
            session.permanent = True # Giữ phiên đăng nhập
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("❌ Sai mật khẩu quản trị!")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


# =============================================================
# ADMIN DASHBOARD
# =============================================================
@app.route("/admin")
@admin_required
def admin_dashboard():
    # Sửa lỗi lấy danh sách tài liệu để hiển thị trong bảng Admin
    pdf_collections = get_pdf_collections()
    
    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor(cursor_factory=RealDictCursor)
    c.execute("SELECT id, question, answer, timestamp, collection_name "
              "FROM history ORDER BY timestamp DESC LIMIT 100")
    history = c.fetchall()
    conn.close()
    
    return render_template("admin.html",
                           pdf_list=list(pdf_collections.values()),
                           history=history)

# Các route khác (admin_refresh, delete_history...) giữ nguyên như code của bạn
@app.route("/admin/refresh", methods=["POST"])
@admin_required
def admin_refresh():
    get_pdf_collections(force_refresh=True)
    flash("✅ Đã refresh danh sách tài liệu!")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete_history/<int:history_id>", methods=["POST"])
@admin_required
def admin_delete_history(history_id):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute("DELETE FROM history WHERE id = %s", (history_id,))
        conn.commit()
        conn.close()
        flash("✅ Đã xóa bản ghi!")
    except Exception as e:
        flash(f"❌ Lỗi xóa: {e}")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete_history_all", methods=["POST"])
@admin_required
def admin_delete_history_all():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        flash("✅ Đã xóa toàn bộ lịch sử!")
    except Exception as e:
        flash(f"❌ Lỗi xóa: {e}")
    return redirect(url_for("admin_dashboard"))

# =============================================================
# DATABASE POSTGRES - Dữ nguyên
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
            "INSERT INTO history (question, answer, timestamp, collection_name) "
            "VALUES (%s, %s, %s, %s)",
            (question, answer, datetime.now().isoformat(), collection_name),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Lỗi lưu history: {e}")

try:
    init_history_db()
except Exception as e:
    print(f"⚠️ Không init được DB: {e}")

# =============================================================
# RAG + GEMINI - Dữ nguyên
# =============================================================
def retrieve_with_metadata(question: str, collection_name: str, k: int = 30):
    pdf_collections = get_pdf_collections()
    try:
        col     = pdf_collections[collection_name]["collection"]
        results = col.query(query_texts=[question], n_results=k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[1.0] * len(documents)])[0]
        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            kw_score = sum(
                0.1 for w in question.lower().split()
                if len(w) > 3 and w in doc.lower()
            )
            chunks.append({
                "content":         doc,
                "source":          meta.get("source", "Không rõ nguồn"),
                "page":            meta.get("page", "?"),
                "relevance_score": (1 - dist) + kw_score,
            })
        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
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

def ask_gemini(question: str, collection_name: str) -> str:
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
    last_error = "Lỗi không xác định"
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=_model_name, contents=prompt
            )
            return format_citations(response.text)
        except Exception as e:
            last_error = str(e)
            if ("429" in last_error or "503" in last_error) and attempt < 2:
                time.sleep((attempt + 1) * 2)
            else:
                break
    return f"⚠️ Lỗi API sau 3 lần thử: {last_error}"

# =============================================================
# ROUTES NGƯỜI DÙNG - Giữ nguyên
# =============================================================
@app.route("/")
def home():
    pdf_collections = get_pdf_collections()
    pdf_files       = get_pdf_files()
    if not pdf_collections:
        return (
            "<h2>Chưa có tài liệu nào.</h2>"
            "<p>Admin vui lòng chạy build_db.py và commit lên Git.</p>"
        )
    pdf_list = [{"id": k, "name": v["name"]} for k, v in pdf_collections.items()]
    return render_template("index.html", pdf_list=pdf_list, pdf_files=pdf_files)

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

@app.route("/download/<path:filename>")
def download_pdf(filename):
    return send_from_directory(
        directory=os.path.abspath(PDF_DIR),
        path=filename,
        as_attachment=True
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)