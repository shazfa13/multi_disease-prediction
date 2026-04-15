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


MODEL_CACHE = None
INPUT_MODE_CACHE = None


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
def get_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is not installed in the active Python environment. "
            "Activate the project .venv and run python app.py from there."
        ) from exc

    return tf


def build_model():
    """Rebuild the prediction model without relying on legacy HDF5 deserialization."""
    tf = get_tensorflow()
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Lambda(
        tf.keras.applications.resnet.preprocess_input,
        name="preprocess_input",
    )(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Lambda(lambda t: t, name="last_conv_map")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="chest_xray_resnet50")


def load_prediction_model(model_path: str):
    """Load the trained model artifact with strict fallback behavior."""
    tf = get_tensorflow()

    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except Exception as load_exc:
        # Fallback only for weights-only files; never use partial/mismatched loading
        # because it can silently produce meaningless overconfident predictions.
        model = build_model()
        try:
            model.load_weights(model_path)
            return model
        except Exception as weights_exc:
            raise RuntimeError(
                "Unable to load model.h5 as a full model or strict weights file. "
                "Please re-export or retrain model/model.h5."
            ) from weights_exc


def get_prediction_model():
    global MODEL_CACHE, INPUT_MODE_CACHE

    if MODEL_CACHE is None:
        MODEL_CACHE = load_prediction_model(MODEL_PATH)
        INPUT_MODE_CACHE = detect_input_mode(MODEL_CACHE)

    return MODEL_CACHE


def get_input_mode() -> str:
    if INPUT_MODE_CACHE is None:
        get_prediction_model()
    return INPUT_MODE_CACHE or "unit"


def detect_input_mode(model) -> str:
    """Infer expected input scaling from the loaded model graph."""
    tf = get_tensorflow()
    for layer in model.layers[:20]:
        name = layer.name.lower()
        if "preprocess" in name:
            return "raw255"
        if isinstance(layer, tf.keras.layers.Rescaling):
            return "raw255"
        if isinstance(layer, tf.keras.layers.Lambda):
            try:
                fn_text = str(layer.get_config().get("function", "")).lower()
                if "preprocess_input" in fn_text:
                    return "raw255"
            except Exception:
                pass
    return "unit"


# =========================
# ✅ PREPROCESS
# =========================
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Uploaded file could not be read as an image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224)).astype(np.float32)
    raw_batch = np.expand_dims(img_resized, axis=0)
    unit_batch = np.expand_dims(img_resized / 255.0, axis=0)

    if get_input_mode() == "unit":
        return unit_batch, raw_batch, img
    return raw_batch, unit_batch, img


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
def _prediction_entropy(probs: np.ndarray) -> float:
    safe = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def choose_probabilities(primary_probs: np.ndarray, secondary_probs: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Choose the most reliable probability vector between two preprocessing modes."""
    if secondary_probs is None:
        return primary_probs, False

    p1_max = float(np.max(primary_probs))
    p2_max = float(np.max(secondary_probs))
    p1_entropy = _prediction_entropy(primary_probs)
    p2_entropy = _prediction_entropy(secondary_probs)

    primary_saturated = p1_max >= 0.999
    secondary_healthier = (p2_max <= 0.995) or (p2_entropy > p1_entropy + 0.15)
    if primary_saturated and secondary_healthier:
        return secondary_probs, True

    return primary_probs, False


def predict_image(model, image_batch, fallback_batch=None):
    raw_primary = model.predict(image_batch, verbose=0)[0]
    primary_probs = normalize_probabilities(raw_primary)

    secondary_probs = None
    if fallback_batch is not None:
        raw_secondary = model.predict(fallback_batch, verbose=0)[0]
        secondary_probs = normalize_probabilities(raw_secondary)

    probs, used_fallback = choose_probabilities(primary_probs, secondary_probs)
    chosen_batch = fallback_batch if used_fallback else image_batch

    idx = np.argmax(probs)
    prediction = CLASS_NAMES[idx]
    confidence = float(probs[idx])

    return prediction, confidence, probs, chosen_batch


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


def build_history_analytics(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare per-patient chart points from prediction history."""
    points: list[dict[str, Any]] = []
    for row in reversed(history_rows):
        patient_name = (row.get("patient_name") or "Unknown").strip() or "Unknown"
        timestamp = str(row.get("timestamp") or "")
        short_time = timestamp[5:16] if len(timestamp) >= 16 else timestamp
        points.append(
            {
                "id": row.get("id"),
                "label": f"{patient_name} ({short_time})" if short_time else patient_name,
                "patient_name": patient_name,
                "prediction": row.get("prediction") or "-",
                "confidence": round(float(row.get("confidence") or 0.0), 2),
                "timestamp": timestamp,
            }
        )
    return points


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
    patient_name = ""
    patient_age = ""
    user_id = session.get('user_id')
    error_message = session.pop("analysis_error", None)
    form_state = session.pop("analysis_form_state", None)
    if isinstance(form_state, dict):
        patient_name = form_state.get("patient_name", "")
        patient_age = form_state.get("patient_age", "")

    history_rows = fetch_history(user_id, limit=30)
    history_analytics_data = build_history_analytics(history_rows)
    latest_record_id = history_rows[0]["id"] if history_rows else None

    return render_template(
        "index.html",
        probability_data=[],
        error_message=error_message,
        patient_name=patient_name,
        patient_age=patient_age,
        history_rows=history_rows,
        history_analytics_data=history_analytics_data,
        latest_record_id=latest_record_id,
    )


@app.route("/analyze", methods=["POST"])
@login_required
def analyze_xray():
    user_id = session.get('user_id')
    patient_name = (request.form.get("patient_name") or "").strip()
    patient_age = (request.form.get("age") or "").strip()
    session["analysis_form_state"] = {
        "patient_name": patient_name,
        "patient_age": patient_age,
    }

    try:
        if "file" not in request.files:
            raise ValueError("Please select an image file")

        file = request.files["file"]
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

        model = get_prediction_model()
        img_batch, fallback_batch, original = preprocess_image(path)

        if not is_xray_like(original):
            os.remove(path)
            raise ValueError("Not a valid chest X-ray image")

        prediction, confidence, probs, gradcam_batch = predict_image(model, img_batch, fallback_batch)

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
            model=model,
            image_array=gradcam_batch,
            original_rgb=original,
            heatmap_output_path=heatmap_path,
            overlay_output_path=overlay_path,
            class_index=int(np.argmax(probs)),
            last_conv_layer_name=get_last_conv_layer_name(model),
        )

        original_image = url_for("static", filename="uploads/" + unique_name)
        heatmap_image = url_for("static", filename="heatmaps/" + heatmap_name)
        overlay_image = url_for("static", filename="heatmaps/" + overlay_name)

        confidence_pct = round(confidence * 100, 2)
        age_value = int(patient_age) if patient_age.isdigit() else None
        save_prediction_record(user_id, patient_name, age_value, unique_name, prediction, confidence_pct)

        session["latest_analysis"] = {
            "prediction": prediction,
            "confidence": confidence_pct,
            "probability_data": probability_data,
            "disease_predictions": disease_predictions,
            "top_two": top_two,
            "original_image": original_image,
            "heatmap_image": heatmap_image,
            "overlay_image": overlay_image,
            "patient_name": patient_name,
            "patient_age": patient_age,
        }
        session.pop("analysis_error", None)
        return redirect(url_for("analysis"))

    except Exception as e:
        traceback.print_exc()
        session.pop("latest_analysis", None)
        session["analysis_error"] = str(e)
        return redirect(url_for("dashboard"))


@app.route("/analysis", methods=["GET"])
@login_required
def analysis():
    analysis_data = session.get("latest_analysis")
    if not analysis_data:
        return redirect(url_for("dashboard"))

    user_id = session.get('user_id')
    history_rows = fetch_history(user_id, limit=30)
    history_analytics_data = build_history_analytics(history_rows)
    latest_record_id = history_rows[0]["id"] if history_rows else None

    return render_template(
        "analysis.html",
        prediction=analysis_data.get("prediction"),
        confidence=analysis_data.get("confidence"),
        probability_data=analysis_data.get("probability_data", []),
        disease_predictions=analysis_data.get("disease_predictions", {}),
        top_two=analysis_data.get("top_two", []),
        original_image=analysis_data.get("original_image"),
        heatmap_image=analysis_data.get("heatmap_image"),
        overlay_image=analysis_data.get("overlay_image"),
        patient_name=analysis_data.get("patient_name", ""),
        patient_age=analysis_data.get("patient_age", ""),
        history_rows=history_rows,
        history_analytics_data=history_analytics_data,
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