import os
import sqlite3
import traceback
import uuid
from functools import wraps
from io import BytesIO
from datetime import datetime
from typing import Any

import cv2
import numpy as np

# oneDNN can increase CPU memory usage for some graphs on low-memory systems.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from flask import Flask, abort, redirect, render_template, request, send_file, url_for, session
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from model.gradcam import generate_gradcam_visuals, get_last_conv_layer_name


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
HEATMAP_DIR = os.path.join(STATIC_DIR, "heatmaps")
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")
DB_PATH = os.path.join(BASE_DIR, "database.db")

CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


app = Flask(__name__)
app.secret_key = "your-secret-key-change-this-in-production"
app.config['SESSION_TYPE'] = 'filesystem'

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)


# =========================
# ✅ IMAGE VALIDATION
# =========================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_xray_like(image_rgb: np.ndarray) -> bool:
    if image_rgb is None or image_rgb.size == 0:
        return False

    h, w, c = image_rgb.shape
    if c != 3 or min(h, w) < 80:
        return False

    # X-rays are near-monochrome; highly colorful images are unlikely to be valid studies.
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    colorfulness = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    if colorfulness > 18:
        return False

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))

    if std < 18:
        return False
    if mean < 30 or mean > 220:
        return False

    median_val = float(np.median(gray))
    low = int(max(0, 0.66 * median_val))
    high = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(gray, low, high)
    edge_density = float(np.mean(edges > 0))
    if edge_density < 0.01 or edge_density > 0.35:
        return False

    return True


# =========================
# ✅ DB INIT
# =========================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            patient_name TEXT,
            age INTEGER,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)


init_db()


# =========================
# ✅ MODEL LOAD
# =========================
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)


def detect_input_mode(model: tf.keras.Model) -> str:
    """Infer expected input scaling from the loaded model graph."""
    for layer in model.layers[:12]:
        name = layer.name.lower()
        if "preprocess" in name:
            return "raw255"
        if isinstance(layer, tf.keras.layers.Rescaling):
            return "raw255"
    return "unit"


INPUT_MODE = detect_input_mode(MODEL)


# =========================
# ✅ PREPROCESS
# =========================
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Uploaded file could not be read as an image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_ready = img_resized.astype(np.float32)

    # Some saved models include preprocessing layers; others expect 0-1 input.
    if INPUT_MODE == "unit":
        img_ready = img_ready / 255.0

    img_exp = np.expand_dims(img_ready, axis=0)
    return img_exp, img


# =========================
# ✅ PROBABILITY FIX
# =========================
def normalize_probabilities(raw_output: np.ndarray) -> np.ndarray:
    scores = np.array(raw_output, dtype=np.float32).flatten()
    if scores.size != len(CLASS_NAMES):
        raise ValueError("Unexpected model output shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced invalid scores")

    total = float(np.sum(scores))
    is_probability_vector = (
        np.all(scores >= 0)
        and np.all(scores <= 1.0 + 1e-3)
        and abs(total - 1.0) < 0.05
    )
    if is_probability_vector:
        probs = scores / max(total, 1e-8)
    else:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores)

    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / np.sum(probs)
    return probs


# =========================
# ✅ PREDICTION
# =========================
def predict_image(image_batch):
    raw = MODEL.predict(image_batch, verbose=0)[0]
    probs = normalize_probabilities(raw)

    idx = np.argmax(probs)
    prediction = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    if len(probs) > 1:
        second = float(np.partition(probs, -2)[-2])
    else:
        second = 0.0

    return prediction, confidence, probs


def get_risk_level(probability: float) -> dict[str, str]:
    if probability < 0.3:
        return {"level": "Low", "color": "green"}
    elif probability < 0.7:
        return {"level": "Medium", "color": "yellow"}
    else:
        return {"level": "High", "color": "red"}


def recommend_doctor(disease: str, probability: float) -> dict[str, str]:
    if probability < 0.3:
        urgency = "Routine Checkup"
    elif probability < 0.7:
        urgency = "Consult Doctor Soon"
    else:
        urgency = "Immediate Attention Required"

    disease_key = disease.strip().lower()
    if disease_key in ["pneumonia", "tuberculosis", "covid", "covid-19", "lung cancer"]:
        specialist = "Pulmonologist"
    else:
        specialist = "Radiologist"

    return {
        "specialist": specialist,
        "urgency": urgency,
    }


def build_disease_predictions(probs: np.ndarray) -> dict[str, dict[str, Any]]:
    disease_predictions: dict[str, dict[str, Any]] = {}

    # Optional enhancement: return diseases sorted by highest probability.
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        disease_label = CLASS_NAMES[int(i)].title()
        probability = float(probs[i])
        risk = get_risk_level(probability)

        disease_predictions[disease_label] = {
            "probability": round(probability, 4),
            "risk_level": risk["level"],
            "color": risk["color"],
            "doctor_recommendation": recommend_doctor(disease_label, probability),
        }

    return disease_predictions


# =========================
# ✅ USER MANAGEMENT
# =========================
def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)"""
    try:
        password_hash = generate_password_hash(password)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, email, password_hash, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
        return True, "Registration successful"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists"
        elif "email" in str(e):
            return False, "Email already exists"
        return False, "Registration failed"
    except Exception as e:
        return False, str(e)


def verify_user(username: str, password: str) -> dict[str, Any] | None:
    """Verify user credentials. Returns user dict if successful, None otherwise"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT id, username, email FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    
    if not user:
        return None
    
    # Get password hash separately
    with sqlite3.connect(DB_PATH) as conn:
        result = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    
    if result and check_password_hash(result[0], password):
        return dict(user) if isinstance(user, sqlite3.Row) else user
    
    return None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Get user info by ID"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute(
            "SELECT id, username, email FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(user) if user else None


# =========================
# ✅ PREDICTION HISTORY (Updated for user_id)
# =========================
def save_prediction_record(user_id: int, patient_name: str | None, age: int | None, filename: str, prediction: str, confidence: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (user_id, patient_name, age, filename, prediction, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                patient_name,
                age,
                filename,
                prediction,
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )


def fetch_history(user_id: int, limit: int = 25) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, patient_name, age, filename, prediction, confidence, timestamp
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_record(record_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, patient_name, age, filename, prediction, confidence, timestamp, user_id
            FROM prediction_history
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_record(record_id: int, user_id: int) -> bool:
    record = fetch_record(record_id)
    if not record or record.get("user_id") != user_id:
        return False

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM prediction_history WHERE id = ?",
            (record_id,),
        )

    filename = record.get("filename")
    if filename:
        upload_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass

    return True


def build_report_pdf(record: dict[str, Any]) -> BytesIO:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, height - 45 * mm, width, 45 * mm, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(18 * mm, height - 18 * mm, "Chest X-ray AI Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(18 * mm, height - 26 * mm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 60 * mm
    line_h = 9 * mm
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 12)
    rows = [
        ("Record ID", str(record.get("id", "-"))),
        ("Patient Name", str(record.get("patient_name") or "Unknown")),
        ("Age", str(record.get("age") if record.get("age") is not None else "-")),
        ("Prediction", str(record.get("prediction", "-"))),
        ("Confidence", f"{float(record.get('confidence', 0.0)):.2f}%"),
        ("Timestamp", str(record.get("timestamp", "-"))),
    ]
    for key, value in rows:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(18 * mm, y, f"{key}:")
        pdf.setFont("Helvetica", 12)
        pdf.drawString(55 * mm, y, value)
        y -= line_h

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColor(colors.HexColor("#475569"))
    pdf.drawString(18 * mm, 14 * mm, "For screening support only. Final diagnosis must be confirmed by a radiologist.")
    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return buffer


# =========================
# ✅ ROUTES
# =========================

# Helper function to check if user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/", methods=["GET"])
def welcome():
    """Welcome/Landing page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template("welcome.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User signup page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            error = "All fields are required"
        elif password != confirm_password:
            error = "Passwords do not match"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        else:
            success, message = register_user(username, email, password)
            if success:
                return redirect(url_for('login'))
            else:
                error = message

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = verify_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/logout", methods=["GET"])
def logout():
    """User logout"""
    session.clear()
    return redirect(url_for('welcome'))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    prediction = None
    confidence = None
    probability_data = []
    disease_predictions: dict[str, dict[str, Any]] = {}
    top_two: list[dict[str, Any]] = []
    original_image = None
    heatmap_image = None
    overlay_image = None
    error_message = None
    patient_name = ""
    patient_age = ""
    user_id = session.get('user_id')

    if request.method == "POST":
        try:
            if "file" not in request.files:
                raise ValueError("Please select an image file")

            file = request.files["file"]
            patient_name = (request.form.get("patient_name") or "").strip()
            patient_age = (request.form.get("age") or "").strip()

            if file.filename == "":
                raise ValueError("No file selected")

            if not allowed_file(file.filename):
                raise ValueError("Unsupported file format. Use PNG, JPG, JPEG, BMP, or WEBP.")

            if file.mimetype and not file.mimetype.startswith("image/"):
                raise ValueError("Uploaded file is not an image")

            filename = secure_filename(file.filename)
            unique_name = str(uuid.uuid4()) + "_" + filename
            path = os.path.join(UPLOAD_DIR, unique_name)
            file.save(path)

            img_batch, original = preprocess_image(path)

            # 🔥 VALIDATION
            if not is_xray_like(original):
                os.remove(path)
                raise ValueError("Not a valid chest X-ray image")

            prediction, confidence, probs = predict_image(img_batch)

            sorted_idx = np.argsort(probs)[::-1]
            top_two = [
                {"label": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100.0, 2)}
                for i in sorted_idx[:2]
            ]

            probability_data = [
                {"label": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100.0, 2)}
                for i in range(len(CLASS_NAMES))
            ]
            disease_predictions = build_disease_predictions(probs)

            heatmap_name = f"heatmap_{uuid.uuid4().hex}.jpg"
            overlay_name = f"overlay_{uuid.uuid4().hex}.jpg"

            heatmap_path = os.path.join(HEATMAP_DIR, heatmap_name)
            overlay_path = os.path.join(HEATMAP_DIR, overlay_name)

            generate_gradcam_visuals(
                model=MODEL,
                image_array=img_batch,
                original_rgb=original,
                heatmap_output_path=heatmap_path,
                overlay_output_path=overlay_path,
                class_index=int(np.argmax(probs)),
                last_conv_layer_name=get_last_conv_layer_name(MODEL),
            )

            original_image = url_for("static", filename="uploads/" + unique_name)
            heatmap_image = url_for("static", filename="heatmaps/" + heatmap_name)
            overlay_image = url_for("static", filename="heatmaps/" + overlay_name)

            confidence = round(confidence * 100, 2)
            age_value = int(patient_age) if patient_age.isdigit() else None
            save_prediction_record(user_id, patient_name, age_value, unique_name, prediction, confidence)

        except Exception as e:
            traceback.print_exc()
            error_message = str(e)

    history_rows = fetch_history(user_id, limit=30)
    latest_record_id = history_rows[0]["id"] if history_rows else None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        probability_data=probability_data,
        disease_predictions=disease_predictions,
        top_two=top_two,
        original_image=original_image,
        heatmap_image=heatmap_image,
        overlay_image=overlay_image,
        error_message=error_message,
        patient_name=patient_name,
        patient_age=patient_age,
        history_rows=history_rows,
        latest_record_id=latest_record_id,
    )


@app.route("/report/<int:record_id>")
@login_required
def download_report(record_id: int):
    user_id = session.get('user_id')
    record = fetch_record(record_id)
    if not record or record.get('user_id') != user_id:
        abort(404, description="Record not found")

    pdf_bytes = build_report_pdf(record)
    return send_file(
        pdf_bytes,
        as_attachment=True,
        download_name=f"prediction_report_{record_id}.pdf",
        mimetype="application/pdf",
    )


@app.route("/delete-report/<int:record_id>", methods=["POST"])
@login_required
def delete_report(record_id: int):
    user_id = session.get('user_id')
    if not delete_record(record_id, user_id):
        abort(404, description="Record not found")

    return redirect(url_for("dashboard", _anchor="history-section"))


# Legacy support: redirect /index to /dashboard
@app.route("/index", methods=["GET", "POST"])
def index():
    return redirect(url_for('dashboard'))


# =========================
# ✅ RUN
# =========================
if __name__ == "__main__":
    # Disable auto-reloader to avoid a second process importing TensorFlow again.
    app.run(debug=True, use_reloader=False)