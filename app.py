import os
import re
import time
import logging
import threading
from contextlib import contextmanager
from functools import wraps
from datetime import datetime
from datetime import timedelta

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory
from dotenv import load_dotenv
import requests
import concurrent.futures
from werkzeug.security import check_password_hash

# Embedding local
from sentence_transformers import SentenceTransformer

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
_raw_keys = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEYS = [k.strip() for k in _raw_keys.replace('\n', ',').split(',') if k.strip()]
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "").strip()
FREE_ONLY_MODE = os.getenv("FREE_ONLY_MODE", "true").strip().lower() in ("1", "true", "yes", "on")
ALLOW_OPENROUTER_IN_FREE_MODE = os.getenv("ALLOW_OPENROUTER_IN_FREE_MODE", "false").strip().lower() in ("1", "true", "yes", "on")
PERSIST_DIR = "./chroma_db_gemini"
PDF_DIR = "./pdfs"

missing = []
if not DATABASE_URL: missing.append("DATABASE_URL")
if not ADMIN_PASSWORD: missing.append("ADMIN_PASSWORD")
if not SECRET_KEY: missing.append("SECRET_KEY")
if not GROQ_API_KEY and not OPENROUTER_API_KEY and not GEMINI_API_KEYS:
    missing.append("(GROQ hoặc OPENROUTER hoặc GEMINI) API Key")

if missing:
    logger.warning(f"Thiếu biến môi trường: {', '.join(missing)}")
    if os.environ.get("RENDER"):
        raise ValueError(f"Production yêu cầu các Key để chạy ổn định.")

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = bool(os.environ.get("RENDER"))

if os.environ.get("RENDER") and not ADMIN_PASSWORD_HASH:
    logger.warning("Nên cấu hình ADMIN_PASSWORD_HASH trên production để tránh lưu mật khẩu thô.")
if FREE_ONLY_MODE:
    logger.info("FREE_ONLY_MODE đang bật: chỉ dùng luồng provider miễn phí an toàn.")

# =============================================================
# RATE LIMIT AN TOÀN ĐA LUỒNG
# =============================================================
_last_request = {}
_rate_limit_lock = threading.Lock()
_admin_login_attempts = {}
_admin_login_lock = threading.Lock()
_ADMIN_MAX_ATTEMPTS = 5
_ADMIN_LOCKOUT_SECONDS = 600

def rate_limit(ip):
    with _rate_limit_lock:
        now = time.time()
        if ip in _last_request and now - _last_request[ip] < 2:
            return False
        _last_request[ip] = now
        return True

def _is_locked_admin_ip(ip: str) -> tuple[bool, int]:
    if not ip:
        ip = "unknown"
    with _admin_login_lock:
        state = _admin_login_attempts.get(ip)
        if not state:
            return False, 0
        lock_until = state.get("lock_until", 0)
        now = time.time()
        if lock_until > now:
            return True, int(lock_until - now)
        if lock_until and lock_until <= now:
            _admin_login_attempts.pop(ip, None)
        return False, 0

def _record_admin_login_failure(ip: str):
    if not ip:
        ip = "unknown"
    with _admin_login_lock:
        now = time.time()
        state = _admin_login_attempts.get(ip, {"count": 0, "lock_until": 0})
        if state.get("lock_until", 0) > now:
            return
        state["count"] = state.get("count", 0) + 1
        if state["count"] >= _ADMIN_MAX_ATTEMPTS:
            state["lock_until"] = now + _ADMIN_LOCKOUT_SECONDS
            state["count"] = 0
        _admin_login_attempts[ip] = state

def _clear_admin_login_failures(ip: str):
    if not ip:
        ip = "unknown"
    with _admin_login_lock:
        _admin_login_attempts.pop(ip, None)

def verify_admin_password(raw_password: str) -> bool:
    if not raw_password:
        return False
    if ADMIN_PASSWORD_HASH:
        try:
            return check_password_hash(ADMIN_PASSWORD_HASH, raw_password)
        except Exception as e:
            logger.error(f"Lỗi verify ADMIN_PASSWORD_HASH: {e}")
            return False
    if ADMIN_PASSWORD:
        return raw_password == ADMIN_PASSWORD
    return False

# =============================================================
# TRẠNG THÁI KHỞI ĐỘNG
# =============================================================
_app_ready = False
_warmup_status = "Đang chuẩn bị..."
_warmup_started = False
_warmup_lock = threading.Lock()

# =============================================================
# DB POOL + CONTEXT MANAGER
# =============================================================
db_pool = None

def ensure_db_pool():
    global db_pool
    if db_pool is None and DATABASE_URL:
        try:
            db_pool = psycopg2.pool.ThreadedConnectionPool(
                1, 10,
                dsn=DATABASE_URL,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5
            )
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

@contextmanager
def db_connection():
    conn = get_db_conn()
    try:
        yield conn
    finally:
        release_db_conn(conn)

# =============================================================
# EMBEDDING LOCAL
# =============================================================
_embed_model = None
_chroma_client = None
_pdf_collections = None
_gemini_clients = []
_gemini_key_idx = 0
_gemini_model = None
_gemini_lock = threading.Lock()

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

class LocalEmbedFn(EmbeddingFunction):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"✅ Loaded local embedding model: {model_name}")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input, show_progress_bar=False)
        return embeddings.tolist()

def get_embed_fn():
    global _embed_model
    if _embed_model is None:
        _embed_model = LocalEmbedFn()
    return _embed_model

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
        _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
        embed_fn = get_embed_fn()
        _pdf_collections = {}
        try:
            for col in _chroma_client.list_collections():
                if col.name.startswith("pdf_"):
                    _pdf_collections[col.name] = {
                        "id": col.name,
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
                "size": f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size//1024} KB"
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
    ip = request.remote_addr or "unknown"
    locked, remain = _is_locked_admin_ip(ip)
    if locked and request.method == "POST":
        mins = max(1, remain // 60)
        flash(f"❌ Đăng nhập sai quá nhiều lần. Thử lại sau khoảng {mins} phút.")
        return render_template("admin_login.html")

    if request.method == "POST":
        if verify_admin_password(request.form.get("password", "")):
            session.permanent = True
            session["is_admin"] = True
            _clear_admin_login_failures(ip)
            return redirect(url_for("admin_dashboard"))
        _record_admin_login_failure(ip)
        flash("❌ Sai mật khẩu quản trị!")
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))

@app.route("/admin")
@admin_required
def admin_dashboard():
    pdf_collections = get_pdf_collections()
    with db_connection() as conn:
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute("SELECT id, question, answer, timestamp, collection_name FROM history ORDER BY timestamp DESC LIMIT 100")
        history = c.fetchall()
        c.close()
    return render_template("admin.html", pdf_list=list(pdf_collections.values()), history=history)

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
        with db_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM history WHERE id = %s", (history_id,))
            conn.commit()
            c.close()
        flash("✅ Đã xóa bản ghi!")
    except Exception as e:
        flash(f"❌ Lỗi: {e}")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/delete_history_all", methods=["POST"])
@admin_required
def admin_delete_history_all():
    try:
        with db_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM history")
            conn.commit()
            c.close()
        flash("✅ Đã xóa toàn bộ lịch sử!")
    except Exception as e:
        flash(f"❌ Lỗi: {e}")
    return redirect(url_for("admin_dashboard"))

# =============================================================
# DATABASE INIT
# =============================================================
def init_history_db():
    try:
        with db_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS history (
                            id SERIAL PRIMARY KEY,
                            question TEXT, answer TEXT,
                            timestamp TEXT, collection_name TEXT)''')
            conn.commit()
            c.close()
    except Exception as e:
        logger.error(f"Lỗi init DB: {e}")

def save_question_answer(question, answer, collection_name):
    for attempt in range(3):
        try:
            with db_connection() as conn:
                conn.isolation_level
                c = conn.cursor()
                c.execute(
                    "INSERT INTO history (question,answer,timestamp,collection_name) VALUES (%s,%s,%s,%s)",
                    (question, answer, datetime.now().isoformat(), collection_name)
                )
                conn.commit()
                c.close()
                return
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logger.warning(f"⚠️ DB connection lỗi (lần {attempt+1}): {e} — thử lại...")
            _reset_db_pool()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Lỗi lưu history: {e}")
            return

def _reset_db_pool():
    global db_pool
    try:
        if db_pool:
            db_pool.closeall()
    except Exception:
        pass
    db_pool = None
    ensure_db_pool()
    logger.info("🔄 DB Pool đã được reset")

# =============================================================
# HÀM LÀM SẠCH CÂU TRẢ LỜI (CHUYỂN MARKDOWN -> HTML)
# =============================================================
def clean_answer_to_html(text: str) -> str:
    """
    Chuyển đổi markdown cơ bản thành HTML để hiển thị đẹp, loại bỏ ký tự * gây rối.
    - **bold** -> <strong>bold</strong>
    - Dấu gạch đầu dòng (- hoặc * ở đầu dòng) -> <ul><li>...</li></ul>
    - Số superscript (¹²³...) -> <sup>...</sup>
    """
    if not text:
        return text
    
    # 1. Chuyển **bold** thành <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # 2. Xử lý bullet list (dòng bắt đầu bằng -, *, •)
    lines = text.split('\n')
    new_lines = []
    in_list = False
    for line in lines:
        # Kiểm tra nếu dòng bắt đầu bằng dấu hiệu bullet
        bullet_match = re.match(r'^\s*[\*\•\-]\s+(.+)', line)
        if bullet_match:
            if not in_list:
                new_lines.append('<ul>')
                in_list = True
            content = bullet_match.group(1)
            new_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                new_lines.append('</ul>')
                in_list = False
            new_lines.append(line)
    if in_list:
        new_lines.append('</ul>')
    text = '\n'.join(new_lines)
    
    # 3. Chuyển các số superscript (ví dụ: ¹²³) thành <sup>
    # Các ký tự superscript Unicode: ⁰¹²³⁴⁵⁶⁷⁸⁹
    superscript_map = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
    }
    # Tìm chuỗi liên tiếp các ký tự superscript
    def replace_sup(match):
        chars = match.group(0)
        # Chuyển mỗi ký tự thành số, ghép lại
        nums = ''.join(superscript_map.get(ch, ch) for ch in chars)
        return f'<sup>{nums}</sup>'
    
    text = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹]+', replace_sup, text)
    
    # 4. Xóa các dấu * còn sót lại (không nằm trong cặp **)
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    
    # 5. Thay thế xuống dòng kép bằng <br> hoặc giữ nguyên (HTML sẽ hiểu)
    text = text.replace('\n\n', '<br><br>')
    text = text.replace('\n', '<br>')
    
    return text

# =============================================================
# RAG – RETRIEVE
# =============================================================
def retrieve_with_metadata(question: str, collection_name: str, k: int = 30):
    pdf_collections = get_pdf_collections()
    if collection_name not in pdf_collections:
        return []
    try:
        col = pdf_collections[collection_name]["collection"]
        results = col.query(query_texts=[question], n_results=k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[1.0]*len(documents)])[0]
        keywords = set(re.findall(r'\b\w{3,}\b', question.lower()))

        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            kw_score = sum(1 for kw in keywords if kw in doc.lower()) / max(len(keywords), 1)
            relevance = (1 - dist) * 0.65 + kw_score * 0.35
            chunks.append({
                "content": doc,
                "source": meta.get("source", "Không rõ nguồn").split("/")[-1],
                "page": meta.get("page", "?"),
                "relevance_score": relevance,
            })

        chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

        deduped = []
        seen = set()
        for c in chunks:
            compact = " ".join(c["content"].split())
            sig = (c["source"], c["page"], compact[:240])
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(c)

        # Câu hỏi "liệt kê/tất cả" cần recall cao hơn; câu hỏi thường ưu tiên precision.
        list_keywords = ("liệt kê", "tất cả", "các loại", "những loại", "gồm những", "bao gồm")
        is_list_query = any(kw in question.lower() for kw in list_keywords)
        threshold = 0.2 if is_list_query else 0.3
        min_len = 60 if is_list_query else 80
        good = [c for c in deduped if c["relevance_score"] >= threshold and len(c["content"]) >= min_len]
        return (good[:20] if good else deduped[:20])
    except Exception as e:
        logger.error(f"Lỗi ChromaDB: {e}")
        return []

def format_citations(text: str) -> str:
    return text

def build_prompt(question: str, chunks: list, response_level: int = 3) -> str:
    context = "\n\n".join(
        f"[{i}] Nguồn: {c['source']}, Trang {c['page']}\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    )

    list_keywords = ["liệt kê", "các bảo vệ", "các trường hợp", "danh sách",
                     "tất cả", "những loại", "các loại", "có những gì", "gồm những"]
    is_list_question = any(kw in question.lower() for kw in list_keywords)

    extra = ""
    if is_list_question:
        extra = """
- Câu hỏi yêu cầu liệt kê: PHẢI liệt kê ĐẦY ĐỦ tất cả các mục có trong dữ liệu, không được bỏ sót.
- Nhóm các mục theo chủ đề logic nếu có thể (ví dụ: Bảo vệ điện / Bảo vệ nhiệt độ / Dừng sự cố...).
- Mỗi nhóm dùng tiêu đề **in đậm**, mỗi mục dùng dấu gạch đầu dòng ( - ).
"""

    level_instruction = """
6. **Mức trả lời chi tiết** — Trả lời đầy đủ, có phân nhóm và giải thích ngắn cho từng ý.
"""
    if response_level == 1:
        level_instruction = """
6. **Mức trả lời ngắn gọn** — Trả lời trong 3-5 gạch đầu dòng, ưu tiên ý chính, chỉ giữ trích dẫn quan trọng.
"""
    elif response_level == 2:
        level_instruction = """
6. **Mức trả lời trung bình** — Trả lời có cấu trúc ngắn gọn, nêu đủ thông số cốt lõi và trích dẫn theo từng nhóm ý.
"""

    return f"""Bạn là trợ lý phân tích tài liệu kỹ thuật. Hãy trả lời câu hỏi dựa trên dữ liệu bên dưới, theo phong cách của NotebookLM.

## QUY TẮC ĐỊNH DẠNG (bắt buộc tuân theo):

1. **Chú thích nguồn dạng số** — sau mỗi ý quan trọng, đặt số chú thích nhỏ như: `¹`, `²`, `³` (dùng ký tự superscript Unicode: ¹²³⁴⁵⁶⁷⁸⁹). Số này tương ứng với số thứ tự [N] của đoạn dữ liệu bên dưới.
   - ĐÚNG: "Bảo vệ so lệch dọc (87G) tác động tức thời (0s) để dừng máy¹."
   - SAI: "Bảo vệ so lệch dọc (87G) tác động tức thời (Nguồn: file.pdf, trang 10)."

2. **In đậm** tất cả tên bảo vệ, mã thiết bị, thông số kỹ thuật quan trọng. Ví dụ: **87G**, **64S**, **5,2 MPa**, **115% tốc độ định mức**.

3. **Cấu trúc bài** — Dùng tiêu đề nhóm **in đậm** để phân nhóm các ý liên quan. Không dùng `###` markdown heading, chỉ dùng **bold text** làm tiêu đề nhóm.

4. **Ngôn ngữ** — Viết tự nhiên như chuyên gia giải thích, không cứng nhắc. Giữ nguyên thuật ngữ kỹ thuật tiếng Việt từ tài liệu.

5. **Đầy đủ** — Không tóm tắt sơ sài. Liệt kê đủ thông số, điều kiện, thời gian tác động nếu có trong dữ liệu.
{level_instruction}
{extra}

## DỮ LIỆU THAM KHẢO:
{context}

## CÂU HỎI:
{question}

## TRẢ LỜI:"""

# =============================================================
# LLM PROVIDERS – FALLBACK TỰ ĐỘNG
# =============================================================

import concurrent.futures

# =============================================================
# LLM PROVIDERS – SONG SONG (PARALLEL FALLBACK)
# =============================================================

class PayloadTooLargeError(Exception):
    pass

def reduce_prompt_chunks(original_prompt: str, keep_chunks: int) -> str:
    lines = original_prompt.split('\n')
    new_lines = []
    chunk_counter = 0
    in_data_block = False
    for line in lines:
        if re.match(r'^\s*\[\d+\]', line):
            chunk_counter += 1
            if chunk_counter > keep_chunks:
                in_data_block = True
                continue
            else:
                in_data_block = False
        if not in_data_block:
            new_lines.append(line)
    return '\n'.join(new_lines)

def call_openrouter(prompt: str, timeout: int = 60) -> str | None:
    # Mặc định không dùng OpenRouter trong chế độ free-only để tránh rủi ro tính phí.
    if FREE_ONLY_MODE and not ALLOW_OPENROUTER_IN_FREE_MODE:
        return None
    if not OPENROUTER_API_KEY:
        return None
    try:
        print(f"🟡 [OpenRouter] Bắt đầu gọi...")
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "RAG_NotebookLM"
            },
            json={
                "model": "openrouter/auto",  # Tự chọn model tốt nhất available
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4096
            },
            timeout=timeout
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            if content and len(content.strip()) > 50:
                print(f"✅ [OpenRouter] Thành công")
                return content
        else:
            print(f"❌ [OpenRouter] Lỗi {resp.status_code}")
    except Exception as e:
        print(f"❌ [OpenRouter] Exception: {e}")
    return None

def call_groq(prompt: str, timeout: int = 45) -> str | None:
    if not GROQ_API_KEY:
        return None
    # Chỉ thử 1 model mỗi lần gọi song song để không block lâu
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"]
    for model in models:
        try:
            print(f"🟡 [Groq] Thử {model}...")
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 8192
                },
                timeout=timeout
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                if content and len(content.strip()) > 50:
                    print(f"✅ [Groq] Thành công với {model}")
                    return content
            elif resp.status_code == 413:
                print(f"⚠️ [Groq] {model} báo 413, thử model nhỏ hơn...")
                continue
            elif resp.status_code == 429:
                print(f"⚠️ [Groq] {model} bị rate limit, thử model tiếp...")
                continue
            else:
                print(f"❌ [Groq] Lỗi {resp.status_code}")
        except Exception as e:
            print(f"❌ [Groq] Exception {model}: {e}")
    return None

def call_gemini(prompt: str, timeout: int = 45) -> str | None:
    global _gemini_clients, _gemini_key_idx, _gemini_model
    if not GEMINI_API_KEYS:
        return None
    if not _gemini_clients:
        try:
            import google.genai as genai
            for key in GEMINI_API_KEYS:
                try:
                    client = genai.Client(api_key=key)
                    if not _gemini_model:
                        _gemini_model = "gemini-2.0-flash"
                    _gemini_clients.append(client)
                except Exception:
                    continue
        except Exception:
            return None
    if not _gemini_clients:
        return None
    with _gemini_lock:
        client = _gemini_clients[_gemini_key_idx % len(_gemini_clients)]
        _gemini_key_idx += 1
    try:
        print(f"🟡 [Gemini] Bắt đầu gọi...")
        response = client.models.generate_content(
            model=_gemini_model,
            contents=prompt
        )
        if response.text and len(response.text.strip()) > 50:
            print(f"✅ [Gemini] Thành công")
            return response.text
    except Exception as e:
        print(f"❌ [Gemini] Lỗi: {e}")
    return None

def call_llm_parallel(prompt: str) -> str | None:
    """
    Gọi tất cả provider CÙNG LÚC.
    Provider nào trả kết quả hợp lệ trước → dùng ngay, hủy các provider còn lại.
    Thời gian = provider nhanh nhất thay vì tổng thời gian tất cả.
    """
    provider_fns = []
    if GROQ_API_KEY:
        provider_fns.append(("Groq", call_groq))
    if GEMINI_API_KEYS:
        provider_fns.append(("Gemini", call_gemini))
    if OPENROUTER_API_KEY and (not FREE_ONLY_MODE or ALLOW_OPENROUTER_IN_FREE_MODE):
        provider_fns.append(("OpenRouter", call_openrouter))

    if not provider_fns:
        return None

    # Tạo executor và submit tất cả cùng lúc
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(provider_fns)) as executor:
        future_to_name = {
            executor.submit(fn, prompt): name
            for name, fn in provider_fns
        }

        try:
            for future in concurrent.futures.as_completed(
                future_to_name.keys(), timeout=70
            ):
                name = future_to_name[future]
                try:
                    result = future.result(timeout=5)
                    if result and len(result.strip()) > 50:
                        print(f"🏆 [{name}] Về đích trước!")
                        # Cancel các future còn lại (best effort)
                        for f in future_to_name:
                            if f is not future:
                                f.cancel()
                        return result
                    else:
                        print(f"⚠️ [{name}] Kết quả không hợp lệ, chờ provider khác...")
                except Exception as e:
                    print(f"❌ [{name}] Exception: {e}")
                    continue

        except concurrent.futures.TimeoutError:
            print("⏱️ Timeout 70s — tất cả provider đều không kịp")

    return None

def call_llm_with_fallback(prompt: str, original_chunk_count: int) -> str | None:
    """
    Bước 1: Thử song song với full chunk
    Bước 2: Nếu fail, giảm chunk 60% và thử lại song song
    Bước 3: Nếu vẫn fail, giảm chunk 30% và thử lần cuối
    """
    chunk_levels = [
        original_chunk_count,
        max(6, int(original_chunk_count * 0.6)),
        max(3, int(original_chunk_count * 0.3)),
    ]

    for level_idx, keep_chunks in enumerate(chunk_levels):
        if keep_chunks < original_chunk_count:
            current_prompt = reduce_prompt_chunks(prompt, keep_chunks)
            print(f"🔄 Thử lại với {keep_chunks} chunk (mức {level_idx + 1}/3)")
        else:
            current_prompt = prompt
            print(f"🚀 Gọi song song với {keep_chunks} chunk...")

        result = call_llm_parallel(current_prompt)
        if result:
            return result

        print(f"⚠️ Tất cả provider fail ở {keep_chunks} chunk, thử mức tiếp...")

    return None

def ask_llm(question: str, collection_name: str, response_level: int = 3) -> str:
    question_lc = question.lower()
    is_list_query = any(kw in question_lc for kw in ("liệt kê", "tất cả", "bao gồm", "các loại", "những loại"))
    is_procedure_query = any(kw in question_lc for kw in ("trình tự", "quy trình", "các bước", "thao tác"))

    k_map = {1: 12, 2: 18, 3: 30}
    k = k_map.get(response_level, 30)
    if is_list_query:
        k = min(40, k + 8)
    elif is_procedure_query:
        k = min(36, k + 4)

    chunks = retrieve_with_metadata(question, collection_name, k=k)
    if not chunks:
        return "❌ Không tìm thấy thông tin liên quan trong tài liệu."

    prompt = build_prompt(question, chunks, response_level=response_level)

    # Gọi song song — KHÔNG retry lần 2 (tránh gọi LLM 2 lần)
    answer = call_llm_with_fallback(prompt, len(chunks))

    if not answer:
        # Fallback cuối: trả thông tin thô từ ChromaDB
        context_summary = "\n\n".join(
            f"- {c['source']} (trang {c['page']}): {c['content'][:300]}..."
            for c in chunks[:5]
        )
        return (
            "⚠️ Tạm thời không thể kết nối AI. "
            "Thông tin liên quan từ tài liệu:\n\n"
            f"{context_summary}\n\nVui lòng thử lại sau vài giây."
        )

    return clean_answer_to_html(answer)

# =============================================================
# ROUTES NGƯỜI DÙNG
# =============================================================
@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/ready")
def ready():
    return jsonify({"ready": _app_ready, "status": _warmup_status})

def get_loading_html():
    return """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Đang khởi động...</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; background: #f0f4f8; color: #333; }
        .card { background: white; border-radius: 16px; padding: 48px 40px; text-align: center; box-shadow: 0 4px 24px rgba(0,0,0,0.08); max-width: 400px; width: 90%; }
        .spinner { width: 56px; height: 56px; border: 5px solid #e2e8f0; border-top-color: #3b82f6; border-radius: 50%; animation: spin 0.9s linear infinite; margin: 0 auto 24px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        h2 { font-size: 1.25rem; margin-bottom: 10px; color: #1e293b; }
        p  { font-size: 0.9rem; color: #64748b; line-height: 1.5; }
        .status { margin-top: 20px; padding: 10px 16px; background: #f1f5f9; border-radius: 8px; font-size: 0.82rem; color: #475569; }
    </style>
</head>
<body>
    <div class="card">
        <div class="spinner"></div>
        <h2>⚙️ Hệ thống đang khởi động</h2>
        <p>Vui lòng chờ trong giây lát để AI kết nối tài liệu...</p>
        <div class="status" id="status-text">Đang chuẩn bị...</div>
    </div>
    <script>
        async function checkReady() {
            try {
                const res = await fetch('/ready');
                const data = await res.json();
                document.getElementById('status-text').textContent = data.status;
                if (data.ready) { window.location.reload(); }
                else { setTimeout(checkReady, 3000); }
            } catch { setTimeout(checkReady, 3000); }
        }
        checkReady();
    </script>
</body>
</html>"""

@app.route("/")
def home():
    global _app_ready, _warmup_status, _warmup_started
    if not _app_ready:
        with _warmup_lock:
            if not _warmup_started:
                _warmup_started = True
                _warmup_status = "Kết nối Database và tải model embedding..."
                threading.Thread(target=_warmup, daemon=True).start()
        return get_loading_html(), 503
    return render_template("index.html", pdf_list=get_collection_names_only(), pdf_files=get_pdf_files())

@app.route("/ask", methods=["POST"])
def ask():
    ip = request.remote_addr
    if not rate_limit(ip):
        return jsonify({"answer": "⚠️ Bạn gửi quá nhanh, vui lòng chậm lại!"})

    try:
        data = request.get_json()
        if not data or "question" not in data or "collection_name" not in data:
            return jsonify({"answer": "Thiếu thông tin"})

        response_level = data.get("response_level", 3)
        try:
            response_level = int(response_level)
        except (TypeError, ValueError):
            response_level = 3
        if response_level not in (1, 2, 3):
            response_level = 3

        answer = ask_llm(
            data["question"].strip(),
            data["collection_name"],
            response_level=response_level
        )
        save_question_answer(data["question"].strip(), answer, data["collection_name"])
        return jsonify({"answer": answer})
    except Exception as e:
        logger.exception("Lỗi trong /ask")
        return jsonify({"answer": f"Lỗi server: {str(e)}"})

@app.route("/history")
def get_history():
    try:
        with db_connection() as conn:
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute("SELECT id, question, answer, timestamp FROM history ORDER BY timestamp DESC LIMIT 20")
            rows = c.fetchall()
            c.close()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])

@app.route("/download/<path:filename>")
def download_pdf(filename):
    return send_from_directory(directory=os.path.abspath(PDF_DIR), path=filename, as_attachment=True)

# =============================================================
# WARM-UP
# =============================================================
def _warmup():
    global _app_ready, _warmup_status
    try:
        _warmup_status = "Đang mở CSDL..."
        ensure_db_pool()
        _warmup_status = "Đang tải model embedding local..."
        get_embed_fn()
        _warmup_status = "Đang quét danh mục tài liệu..."
        get_pdf_collections()
        _app_ready = True
        _warmup_status = "Sẵn sàng!"
        logger.info("✅ Warm-up hoàn tất với embedding local!")
    except Exception as e:
        logger.error(f"❌ Warm-up lỗi: {e}")
        _app_ready = False
        _warmup_status = f"Lỗi khởi động: {str(e)[:100]}"

# =============================================================
# KHỞI ĐỘNG
# =============================================================
try:
    ensure_db_pool()
    init_history_db()
except Exception as e:
    logger.warning(f"Không thể kết nối DB lúc khởi động: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)