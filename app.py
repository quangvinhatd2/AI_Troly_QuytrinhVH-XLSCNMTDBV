import os
import re
import time
import logging
import threading
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from datetime import datetime
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, flash, send_from_directory)
from dotenv import load_dotenv
import requests

load_dotenv()

# =============================================================
# LOGGING
# =============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================
# BIẾN MÔI TRƯỜNG
# =============================================================
_raw_keys       = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.replace('\n', ',').split(',') if k.strip()]
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
DATABASE_URL    = os.getenv("DATABASE_URL")
ADMIN_PASSWORD  = os.getenv("ADMIN_PASSWORD")
SECRET_KEY      = os.getenv("SECRET_KEY")
PERSIST_DIR     = "./chroma_db_gemini"
PDF_DIR         = "./pdfs"

missing = []
if not DATABASE_URL:   missing.append("DATABASE_URL")
if not ADMIN_PASSWORD: missing.append("ADMIN_PASSWORD")
if not SECRET_KEY:     missing.append("SECRET_KEY")
if not GROQ_API_KEY:   missing.append("GROQ_API_KEY")
if missing:
    logger.warning(f"Thiếu biến môi trường: {', '.join(missing)}")
    if os.environ.get("RENDER"):
        raise ValueError(f"Production yêu cầu: {', '.join(missing)}")
    ADMIN_PASSWORD = ADMIN_PASSWORD or "admin123"
    SECRET_KEY     = SECRET_KEY     or "change-me-please"

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# =============================================================
# TRẠNG THÁI KHỞI ĐỘNG
# =============================================================
_app_ready      = False
_warmup_status  = "Đang khởi động..."

# =============================================================
# DB POOL
# =============================================================
db_pool = None

def ensure_db_pool():
    global db_pool
    if db_pool is None and DATABASE_URL:
        try:
            db_pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=DATABASE_URL)
            logger.info("✅ DB Pool initialized")
        except Exception as e:
            logger.error(f"Lỗi init DB pool: {e}")
    return db_pool

def get_db_conn():
    p = ensure_db_pool()
    if p is None:
        raise Exception("DB Pool chưa khởi tạo")
    return p.getconn()

def release_db_conn(conn):
    if db_pool:
        db_pool.putconn(conn)

# =============================================================
# LAZY GLOBALS
# =============================================================
_embed_fn        = None
_chroma_client   = None
_pdf_collections = None
_gemini_clients  = []
_gemini_key_idx  = 0
_gemini_model    = None

def get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from sentence_transformers import SentenceTransformer
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
        logger.info("Load SentenceTransformer...")
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _model.max_seq_length = 256

        class _EmbedFn(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return _model.encode(input, convert_to_numpy=True).tolist()

        _embed_fn = _EmbedFn()
        logger.info("✅ SentenceTransformer sẵn sàng.")
    return _embed_fn

def get_display_name(col_name: str) -> str:
    match = re.match(r'^pdf_(.*)_([a-f0-9]{12})$', col_name)
    if match:
        return match.group(1).replace('_', ' ').title()
    if col_name.startswith("pdf_"):
        return col_name[4:].replace('_', ' ').title()
    return col_name.replace('_', ' ').title()

def get_collection_names_only():
    import chromadb
    result = []
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        for col in client.list_collections():
            if col.name.startswith("pdf_"):
                result.append({"id": col.name, "name": get_display_name(col.name)})
    except Exception as e:
        logger.error(f"Lỗi đọc collection names: {e}")
    return result

def get_pdf_collections(force_refresh: bool = False):
    global _chroma_client, _pdf_collections
    if _pdf_collections is None or force_refresh:
        import chromadb
        _chroma_client   = chromadb.PersistentClient(path=PERSIST_DIR)
        embed_fn         = get_embed_fn()
        _pdf_collections = {}
        try:
            for col in _chroma_client.list_collections():
                if col.name.startswith("pdf_"):
                    _pdf_collections[col.name] = {
                        "id":   col.name,
                        "name": get_display_name(col.name),
                        "collection": _chroma_client.get_collection(
                            name=col.name, embedding_function=embed_fn),
                    }
            logger.info(f"✅ Loaded {len(_pdf_collections)} collections.")
        except Exception as e:
            logger.error(f"Lỗi load ChromaDB: {e}")
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
                "size": f"{size/1024/1024:.1f} MB" if size > 1024*1024
                        else f"{size//1024} KB"
            })
    return files

# =============================================================
# ADMIN AUTH
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
        if request.form.get("password") == ADMIN_PASSWORD:
            session.permanent = True
            session["is_admin"] = True
            return redirect(url_for("admin_dashboard"))
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
    pdf_collections = get_pdf_collections()
    conn = get_db_conn()
    c    = conn.cursor(cursor_factory=RealDictCursor)
    c.execute("SELECT id, question, answer, timestamp, collection_name "
              "FROM history ORDER BY timestamp DESC LIMIT 100")
    history = c.fetchall()
    c.close(); release_db_conn(conn)
    return render_template("admin.html",
                           pdf_list=list(pdf_collections.values()),
                           history=history)

@app.route("/admin/refresh", methods=["POST"])
@admin_required
def admin_refresh():
    get_pdf_collections(force_refresh=True)
    flash("✅ Đã refresh danh sách tài liệu!")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/upload", methods=["POST"])
@admin_required
def admin_upload_pdf():
    if 'file' not in request.files:
        flash("❌ Không có file")
        return redirect(url_for("admin_dashboard"))
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        flash("❌ Chỉ chấp nhận file PDF")
        return redirect(url_for("admin_dashboard"))
    os.makedirs(PDF_DIR, exist_ok=True)
    file.save(os.path.join(PDF_DIR, file.filename))
    flash(f"✅ Đã upload {file.filename}. Nhấn 'Rebuild Database' để cập nhật.")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/rebuild", methods=["POST"])
@admin_required
def admin_rebuild():
    import subprocess
    try:
        result = subprocess.run(
            ["python", "build_db.py"],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            get_pdf_collections(force_refresh=True)
            flash("✅ Rebuild thành công!")
        else:
            flash(f"❌ Rebuild thất bại: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        flash("❌ Rebuild timeout sau 300 giây")
    except Exception as e:
        flash(f"❌ Lỗi: {e}")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete_history/<int:history_id>", methods=["POST"])
@admin_required
def admin_delete_history(history_id):
    try:
        conn = get_db_conn(); c = conn.cursor()
        c.execute("DELETE FROM history WHERE id = %s", (history_id,))
        conn.commit(); c.close(); release_db_conn(conn)
        flash("✅ Đã xóa bản ghi!")
    except Exception as e:
        flash(f"❌ Lỗi: {e}")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete_history_all", methods=["POST"])
@admin_required
def admin_delete_history_all():
    try:
        conn = get_db_conn(); c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit(); c.close(); release_db_conn(conn)
        flash("✅ Đã xóa toàn bộ lịch sử!")
    except Exception as e:
        flash(f"❌ Lỗi: {e}")
    return redirect(url_for("admin_dashboard"))

# =============================================================
# DATABASE
# =============================================================
def init_history_db():
    conn = get_db_conn(); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id SERIAL PRIMARY KEY,
                    question TEXT, answer TEXT,
                    timestamp TEXT, collection_name TEXT)''')
    conn.commit(); c.close(); release_db_conn(conn)

def save_question_answer(question, answer, collection_name):
    try:
        conn = get_db_conn(); c = conn.cursor()
        c.execute(
            "INSERT INTO history (question,answer,timestamp,collection_name) "
            "VALUES (%s,%s,%s,%s)",
            (question, answer, datetime.now().isoformat(), collection_name))
        conn.commit(); c.close(); release_db_conn(conn)
    except Exception as e:
        logger.error(f"Lỗi lưu history: {e}")

# =============================================================
# RAG — RETRIEVE
# =============================================================
def retrieve_with_metadata(question: str, collection_name: str, k: int = 15):
    pdf_collections = get_pdf_collections()
    if collection_name not in pdf_collections:
        return []
    try:
        col     = pdf_collections[collection_name]["collection"]
        results = col.query(query_texts=[question], n_results=k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[1.0]*len(documents)])[0]
        keywords  = set(re.findall(r'\b\w{3,}\b', question.lower()))

        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            kw_score  = sum(1 for kw in keywords if kw in doc.lower()) / max(len(keywords), 1)
            relevance = (1 - dist) * 0.65 + kw_score * 0.35
            chunks.append({
                "content":         doc,
                "source":          meta.get("source", "Không rõ nguồn"),
                "page":            meta.get("page", "?"),
                "relevance_score": relevance,
            })

        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        good = [c for c in chunks if c["relevance_score"] > 0.25 and len(c["content"]) > 80]
        return good[:12] if good else chunks[:8]
    except Exception as e:
        logger.error(f"Lỗi ChromaDB: {e}")
        return []

def format_citations(text: str) -> str:
    return re.sub(r'_?\(Nguồn:\s*(.*?)\)_?',
                  r'<small class="citation-source">(Nguồn: \1)</small>', text)

def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[ĐOẠN {i}] Nguồn: {c['source']}, trang {c['page']}\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    )
    return f"""Bạn là trợ lý phân tích tài liệu kỹ thuật thủy điện. Trả lời câu hỏi DỰA HOÀN TOÀN vào tài liệu bên dưới.

## QUY TẮC
1. Chỉ dùng thông tin trong tài liệu. Không thêm kiến thức bên ngoài.
2. Sau mỗi ý PHẢI ghi trích dẫn: (Nguồn: tên_file, trang X)
3. Nếu không có thông tin → trả lời: "Không tìm thấy trong tài liệu."
4. Cấu trúc bắt buộc:
   - Dòng đầu: tóm tắt 1 câu
   - Các mục chính: 1. 2. 3. ...
   - Ý nhỏ trong mục: dấu -
   - Quy trình: theo đúng thứ tự bước
5. Văn phong kỹ thuật, rõ ràng, tiếng Việt.
6. KHÔNG bịa số liệu, tên thiết bị, bước thực hiện.

## TÀI LIỆU
{context}

## CÂU HỎI
{question}

## TRẢ LỜI"""

# =============================================================
# LLM PROVIDERS — Groq chính, Gemini backup
# =============================================================

def call_groq(prompt: str, timeout: int = 45) -> str | None:
    if not GROQ_API_KEY:
        return None

    groq_models = [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "gemma2-9b-it",
    ]

    for model in groq_models:
        try:
            logger.info(f"Groq: thử model {model}")
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       model,
                    "messages":    [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens":  2048,
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and len(content.strip()) > 50:
                    logger.info(f"✅ Groq thành công: {model}")
                    return content
            elif resp.status_code == 429:
                logger.warning(f"Groq {model} rate limit, thử model tiếp theo...")
                continue
            else:
                logger.warning(f"Groq {model}: {resp.status_code} — {resp.text[:200]}")
        except requests.Timeout:
            logger.warning(f"Groq {model} timeout")
        except Exception as e:
            logger.error(f"Groq {model} lỗi: {e}")

    return None


def call_gemini_backup(prompt: str, timeout: int = 30) -> str | None:
    global _gemini_clients, _gemini_key_idx, _gemini_model

    if not GEMINI_API_KEYS:
        return None

    if not _gemini_clients:
        try:
            import google.genai as genai
            for key in GEMINI_API_KEYS:
                try:
                    client    = genai.Client(api_key=key)
                    available = [m.name for m in client.models.list()]
                    if not _gemini_model:
                        for candidate in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                            if any(candidate in m for m in available):
                                _gemini_model = candidate
                                break
                        if not _gemini_model and available:
                            _gemini_model = available[0]
                    _gemini_clients.append(client)
                    logger.info(f"Gemini key thêm thành công: {key[:10]}...")
                except Exception as e:
                    logger.warning(f"Gemini key lỗi {key[:10]}...: {e}")
        except Exception as e:
            logger.error(f"Không load được google.genai: {e}")
            return None

    if not _gemini_clients or not _gemini_model:
        return None

    for _ in range(len(_gemini_clients)):
        client = _gemini_clients[_gemini_key_idx % len(_gemini_clients)]
        _gemini_key_idx += 1
        try:
            logger.info(f"Gemini backup: thử key {_gemini_key_idx % len(_gemini_clients)}")
            response = client.models.generate_content(
                model=_gemini_model, contents=prompt
            )
            content = response.text
            if content and len(content.strip()) > 50:
                logger.info("✅ Gemini backup thành công")
                return content
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                logger.warning(f"Gemini key {_gemini_key_idx} hết quota, thử key tiếp...")
                continue
            logger.error(f"Gemini lỗi: {e}")

    return None


def ask_llm(question: str, collection_name: str) -> str:
    chunks = retrieve_with_metadata(question, collection_name)
    if not chunks:
        return "❌ Không tìm thấy thông tin liên quan trong tài liệu."

    logger.info(f"Dùng {len(chunks)} chunks | câu hỏi: {question[:60]}")
    prompt = build_prompt(question, chunks)

    answer = call_groq(prompt)

    if not answer:
        logger.warning("Groq fail, chuyển sang Gemini backup...")
        answer = call_gemini_backup(prompt)

    if not answer:
        return ("⚠️ Hệ thống AI tạm thời không khả dụng. "
                "Groq và Gemini đều đã hết quota hoặc gặp lỗi. "
                "Vui lòng thử lại sau vài phút.")

    return format_citations(answer)

# =============================================================
# ROUTES NGƯỜI DÙNG
# =============================================================

# --- FIX: Ping endpoint cho UptimeRobot ---
@app.route("/ping")
def ping():
    return "pong", 200

# --- FIX: Endpoint kiểm tra trạng thái warm-up ---
@app.route("/ready")
def ready():
    return jsonify({"ready": _app_ready, "status": _warmup_status})

@app.route("/")
def home():
    # --- FIX: Hiển thị loading screen thay vì trắng khi chưa sẵn sàng ---
    if not _app_ready:
        return """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Đang khởi động...</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', sans-serif;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            min-height: 100vh; background: #f0f4f8; color: #333;
        }
        .card {
            background: white; border-radius: 16px;
            padding: 48px 40px; text-align: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            max-width: 400px; width: 90%;
        }
        .spinner {
            width: 56px; height: 56px;
            border: 5px solid #e2e8f0;
            border-top-color: #3b82f6;
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
            margin: 0 auto 24px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        h2 { font-size: 1.25rem; margin-bottom: 10px; color: #1e293b; }
        p  { font-size: 0.9rem; color: #64748b; line-height: 1.5; }
        .status {
            margin-top: 20px; padding: 10px 16px;
            background: #f1f5f9; border-radius: 8px;
            font-size: 0.82rem; color: #475569;
        }
        .dot { display: inline-block; animation: blink 1.2s infinite; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink { 0%,80%,100%{opacity:0} 40%{opacity:1} }
    </style>
</head>
<body>
    <div class="card">
        <div class="spinner"></div>
        <h2>⚙️ Hệ thống đang khởi động</h2>
        <p>Đang tải mô hình AI và cơ sở dữ liệu tài liệu, vui lòng chờ trong giây lát<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></p>
        <div class="status" id="status-text">Đang khởi động...</div>
    </div>
    <script>
        // Tự động kiểm tra mỗi 3 giây, redirect khi sẵn sàng
        async function checkReady() {
            try {
                const res  = await fetch('/ready');
                const data = await res.json();
                document.getElementById('status-text').textContent = data.status;
                if (data.ready) {
                    window.location.reload();
                } else {
                    setTimeout(checkReady, 3000);
                }
            } catch {
                setTimeout(checkReady, 3000);
            }
        }
        checkReady();
    </script>
</body>
</html>""", 503

    pdf_list  = get_collection_names_only()
    pdf_files = get_pdf_files()
    if not pdf_list:
        return ("<h2>Chưa có tài liệu nào.</h2>"
                "<p>Admin vui lòng upload PDF và rebuild database.</p>")
    return render_template("index.html", pdf_list=pdf_list, pdf_files=pdf_files)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data or "question" not in data or "collection_name" not in data:
            return jsonify({"answer": "Thiếu câu hỏi hoặc tên quy trình"})

        question        = data["question"].strip()
        collection_name = data["collection_name"]

        if not question:
            return jsonify({"answer": "Câu hỏi không được để trống."})

        available = list(get_pdf_collections().keys())
        if collection_name not in available:
            return jsonify({"answer": f"❌ Không tìm thấy tài liệu: {collection_name}"})

        answer = ask_llm(question, collection_name)
        save_question_answer(question, answer, collection_name)
        return jsonify({"answer": answer})

    except Exception as e:
        logger.exception("Lỗi trong /ask")
        return jsonify({"answer": f"Lỗi server: {str(e)}"})

@app.route("/history")
def get_history():
    try:
        conn = get_db_conn()
        c    = conn.cursor(cursor_factory=RealDictCursor)
        c.execute("SELECT id, question, answer, timestamp FROM history "
                  "ORDER BY timestamp DESC LIMIT 20")
        rows = c.fetchall()
        c.close(); release_db_conn(conn)
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        logger.error(f"Lỗi lấy history: {e}")
        return jsonify([])

@app.route("/download/<path:filename>")
def download_pdf(filename):
    return send_from_directory(
        directory=os.path.abspath(PDF_DIR),
        path=filename,
        as_attachment=True
    )

# =============================================================
# WARM-UP — FIX: bỏ time.sleep(5), cập nhật _warmup_status
# =============================================================
def _warmup():
    global _app_ready, _warmup_status
    logger.info("Warm-up: bắt đầu...")
    try:
        # 1. Kiểm tra kết nối DB trước (nhanh nhất)
        _warmup_status = "Đang kiểm tra cơ sở dữ liệu..."
        ensure_db_pool()
        
        # 2. Load Collections (ChromaDB)
        _warmup_status = "Đang kết nối thư viện tài liệu..."
        get_pdf_collections()

        # 3. Load Embedding (Đây là bước nặng nhất, dễ treo nhất)
        _warmup_status = "Đang tải bộ não AI (Embedding)..."
        get_embed_fn()

        _app_ready     = True
        _warmup_status = "Sẵn sàng!"
        logger.info("✅ Warm-up hoàn tất!")
    except Exception as e:
        logger.error(f"❌ Warm-up thất bại: {e}")
        # Ép buộc sẵn sàng dù lỗi để Admin có thể vào sửa/rebuild
        _app_ready = True 
        _warmup_status = f"Khởi động có lỗi: {str(e)[:50]}"

# =============================================================
# KHỞI ĐỘNG
# =============================================================
try:
    ensure_db_pool()
    init_history_db()
except Exception as e:
    logger.error(f"Lỗi khởi tạo DB: {e}")

threading.Thread(target=_warmup, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)