# Chest X-ray Multi-Disease Prediction Web App

A Flask-based web application for chest X-ray screening with:
- Multi-class disease prediction (COVID, NORMAL, PNEUMONIA, TUBERCULOSIS)
- Grad-CAM explainability (heatmap + overlay)
- User authentication (signup/login/logout)
- Patient-wise prediction history
- Analytics dashboard from historical records
- PDF report generation per prediction

This project is designed as an AI-assisted screening tool, not a replacement for clinical diagnosis.

## 1. Features

### Core AI Features
- Predicts one class from 4 chest X-ray categories.
- Returns class confidence values.
- Generates:
  - Heatmap image
  - Overlay image (heatmap on original X-ray)
- Includes fallback inference safeguards to reduce overconfident saturated outputs.

### App Features
- User registration and login using SQLite.
- Dashboard for patient metadata and image upload.
- Dedicated Analysis page for results.
- Prediction history table with delete and PDF download actions.
- Analytics section with:
  - Per-patient confidence trend across history
  - Disease-wise counts across history

### Data Persistence
- User and prediction records are stored in `database.db`.
- Uploaded images are stored in `static/uploads/`.
- Grad-CAM artifacts are stored in `static/heatmaps/`.

## 2. Tech Stack

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib (training utility)
- ReportLab (PDF generation)
- SQLite
- Chart.js (frontend analytics)

## 3. Project Structure

```text
multi_desease_prediction/
|-- app.py
|-- requirements.txt
|-- database.db
|-- model/
|   |-- __init__.py
|   |-- gradcam.py
|   |-- model.h5
|   `-- train.py
|-- static/
|   |-- css/
|   |   `-- styles.css
|   |-- js/
|   |   `-- app.js
|   |-- uploads/
|   `-- heatmaps/
`-- templates/
    |-- welcome.html
    |-- signup.html
    |-- login.html
    |-- index.html
    |-- analysis.html
    |-- history.html
    `-- analytics.html
```

## 4. Prerequisites

- Windows, Linux, or macOS
- Python 3.10+ recommended
- pip
- (Optional) SQLite CLI (`sqlite3`) for inspecting DB manually

## 5. Installation

### 5.1 Clone/Open Project

```bash
git clone <your-repo-url>
cd multi_desease_prediction
```

### 5.2 Create and Activate Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5.3 Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the Application

```bash
python app.py
```

Then open:
- `http://127.0.0.1:5000`

## 7. Application Workflow

1. Open welcome page.
2. Sign up or log in.
3. On Dashboard:
   - Enter patient name and age
   - Upload chest X-ray image
   - Click **Analyze X-ray**
4. App redirects to Analysis page with:
   - Predicted class
   - Confidence values
   - Grad-CAM heatmap and overlay
   - Recommendation/risk details
5. Use Analytics tab to view historical trends and disease counts.
6. Download report PDF from Report section.

## 8. Routes Overview

- `GET /` -> Welcome page
- `GET|POST /signup` -> Register user
- `GET|POST /login` -> Login
- `GET /logout` -> Logout
- `GET|POST /dashboard` -> Dashboard + upload form view
- `POST /analyze` -> Run inference and redirect to analysis
- `GET /analysis` -> Analysis results page
- `GET /report/<record_id>` -> Download PDF report
- `POST /delete-report/<record_id>` -> Delete prediction record
- `GET|POST /index` -> Legacy redirect to dashboard

## 9. Model and Inference Notes

- Model file: `model/model.h5`
- Classes:
  - COVID
  - NORMAL
  - PNEUMONIA
  - TUBERCULOSIS
- Input resolution: `224 x 224`
- Inference includes:
  - Probability normalization
  - Input scaling detection
  - Fallback scoring path for suspiciously saturated outputs
- Grad-CAM uses last convolutional activation map and writes output images to `static/heatmaps/`.

## 10. Training Utility

You can use `model/train.py` to train (or re-train) the classifier.

Example:

```bash
python model/train.py --data-dir dataset --model-path model/model.h5 --epochs 12 --batch-size 16
```

Notes:
- If dataset path is missing or empty, the script can save an untrained placeholder model.
- Validation generator is separated from augmentation pipeline.

## 11. Database Schema

### users
- `id` (PK)
- `username` (unique)
- `email` (unique)
- `password_hash`
- `created_at`

### prediction_history
- `id` (PK)
- `user_id` (FK -> users.id)
- `patient_name`
- `age`
- `filename`
- `prediction`
- `confidence`
- `timestamp`

## 12. Configuration and Important Files

- Main app entry: `app.py`
- Grad-CAM utilities: `model/gradcam.py`
- Frontend interactions/charts: `static/js/app.js`
- Styling: `static/css/styles.css`
- Templates: `templates/*.html`

## 13. Troubleshooting

### A) TensorFlow not found
Symptom: runtime error saying TensorFlow is not installed.

Fix:
1. Activate the project virtual environment.
2. Install dependencies again:

```bash
pip install -r requirements.txt
```

### B) App starts but predictions look wrong or overconfident
Possible causes:
- Incompatible `model.h5`
- Mismatch between training preprocessing and inference expectations

Fix:
1. Ensure `model/model.h5` is the correct trained artifact.
2. Re-train/export model if needed.
3. Test with known validation images.

### C) `python app.py` exits immediately
Fix checklist:
- Run inside project venv.
- Check terminal logs for import/runtime exceptions.
- Verify dependencies are installed.

### D) Uploaded file rejected
Fix checklist:
- Use supported image formats: PNG, JPG, JPEG, BMP, WEBP.
- Ensure image is a valid chest X-ray-like image (app includes heuristic validation).

## 14. Security Notes (Important)

For production deployment, update these areas first:
- Replace hardcoded Flask `secret_key` with environment variable.
- Disable debug mode.
- Use a production WSGI server (gunicorn/waitress/uwsgi).
- Add CSRF protection for forms.
- Add stricter file upload validation and scanning.
- Consider moving from SQLite to PostgreSQL/MySQL for multi-user scale.

## 15. Disclaimer

This software is for educational/research screening support only.
It is not a certified medical device and must not be used as the sole basis for diagnosis or treatment decisions.
Always involve qualified radiologists/physicians for final interpretation.

## 16. Quick Start (Windows)

```powershell
cd D:\chest-xray-streamlit\multi_desease_prediction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.
