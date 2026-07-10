"""
app.py
======
Aplikasi web Flask untuk klasifikasi jenis makanan Indonesia
menggunakan model VGG16 hasil transfer learning.

Fitur:
- Halaman utama: upload gambar + preview
- Prediksi jenis makanan beserta confidence score (dan top-3 prediksi)
- Halaman About berisi penjelasan project, arsitektur model, dan dataset
- Penanganan error (file tidak valid, model belum tersedia, dll)

Menjalankan secara lokal:
    python app.py

Menjalankan dengan gunicorn (production, sesuai Procfile):
    gunicorn app:app
"""

import os
import uuid
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
)
from werkzeug.utils import secure_filename

import config
import predict as predict_module

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def model_is_ready():
    """Cek apakah model & class_names sudah tersedia (hasil training)."""
    model_exists = os.path.exists(config.MODEL_PATH) or os.path.exists(config.FINAL_MODEL_PATH)
    classes_exist = os.path.exists(config.CLASS_NAMES_PATH)
    return model_exists and classes_exist


@app.context_processor
def inject_globals():
    return {
        "app_name": "Klasifikasi Makanan Indonesia",
        "current_year": datetime.now().year,
        "model_ready": model_is_ready(),
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if not model_is_ready():
        flash(
            "Model belum tersedia. Silakan jalankan 'python train.py' "
            "terlebih dahulu untuk melatih model.",
            "danger",
        )
        return redirect(url_for("index"))

    if "file" not in request.files:
        flash("Tidak ada file yang diupload.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("Silakan pilih file gambar terlebih dahulu.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(
            "Format file tidak didukung. Gunakan file PNG, JPG, JPEG, atau WEBP.",
            "danger",
        )
        return redirect(url_for("index"))

    try:
        # Buat nama file unik agar tidak bentrok antar upload
        original_name = secure_filename(file.filename)
        ext = original_name.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        result = predict_module.predict_image(save_path)

        # Format nama kelas & top predictions agar enak dibaca di UI
        display_result = {
            "predicted_class": predict_module.format_class_name(result["predicted_class"]),
            "confidence": round(result["confidence"] * 100, 2),
            "top_predictions": [
                {
                    "class": predict_module.format_class_name(p["class"]),
                    "confidence": round(p["confidence"] * 100, 2),
                }
                for p in result["top_predictions"]
            ],
        }

        image_url = url_for("static", filename=f"uploads/{unique_name}")

        return render_template(
            "index.html",
            result=display_result,
            image_url=image_url,
        )

    except FileNotFoundError as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Terjadi kesalahan saat memproses gambar: {e}", "danger")
        return redirect(url_for("index"))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON API (opsional) untuk integrasi eksternal / testing."""
    if not model_is_ready():
        return jsonify({"error": "Model belum tersedia."}), 503

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload."}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "File tidak valid."}), 400

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        result = predict_module.predict_image(save_path)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404


@app.errorhandler(413)
def request_entity_too_large(error):
    flash("Ukuran file terlalu besar. Maksimal 5 MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_server_error(error):
    flash("Terjadi kesalahan pada server.", "danger")
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
