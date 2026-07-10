"""
predict.py
==========
Modul untuk melakukan prediksi/inferensi klasifikasi jenis makanan
Indonesia dari sebuah file gambar, menggunakan model VGG16 hasil
transfer learning yang telah disimpan pada tahap training.

Modul ini didesain untuk dipakai oleh app.py (Flask), namun juga bisa
dijalankan langsung dari command line untuk keperluan testing:

    python predict.py path/ke/gambar.jpg
"""

import os
import json
import sys

import numpy as np

import config

# TensorFlow di-import secara lazy (di dalam fungsi) pada beberapa bagian
# agar modul ini tetap bisa di-import dengan cepat tanpa memuat seluruh
# TensorFlow saat tidak diperlukan (misalnya saat testing routing Flask).
_model = None
_class_names = None


def _resolve_model_path():
    """Menentukan path model yang akan dipakai: prioritaskan best_model,
    fallback ke final_model jika best_model tidak ditemukan."""
    if os.path.exists(config.MODEL_PATH):
        return config.MODEL_PATH
    if os.path.exists(config.FINAL_MODEL_PATH):
        return config.FINAL_MODEL_PATH
    raise FileNotFoundError(
        "Model belum ditemukan. Jalankan 'python train.py' terlebih dahulu "
        f"untuk menghasilkan model di '{config.MODEL_PATH}' atau "
        f"'{config.FINAL_MODEL_PATH}'."
    )


def load_class_names():
    global _class_names
    if _class_names is not None:
        return _class_names

    if not os.path.exists(config.CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"File daftar kelas tidak ditemukan di '{config.CLASS_NAMES_PATH}'. "
            "Jalankan 'python train.py' terlebih dahulu."
        )

    with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        _class_names = json.load(f)
    return _class_names


def load_trained_model():
    """Load model Keras sekali saja (singleton) supaya efisien saat
    dipanggil berulang kali dari route Flask."""
    global _model
    if _model is not None:
        return _model

    from tensorflow.keras.models import load_model

    model_path = _resolve_model_path()
    print(f">> Memuat model dari: {model_path}")
    _model = load_model(model_path)
    return _model


def preprocess_image(image_path):
    """
    Preprocessing gambar sesuai kebutuhan VGG16:
    1. Load gambar & resize ke 224x224
    2. Konversi ke array numpy
    3. Tambahkan dimensi batch
    4. Terapkan preprocess_input khusus VGG16 (BGR & mean-centering)
    """
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.vgg16 import preprocess_input

    img = keras_image.load_img(image_path, target_size=config.IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(image_path, top_k=None):
    """
    Melakukan prediksi kelas makanan dari sebuah file gambar.

    Returns
    -------
    dict berisi:
        - predicted_class : str, nama kelas dengan confidence tertinggi
        - confidence      : float, confidence score kelas teratas (0-1)
        - top_predictions : list of dict [{class, confidence}, ...]
                            diurutkan dari confidence tertinggi
    """
    if top_k is None:
        top_k = config.TOP_K_PREDICTIONS

    model = load_trained_model()
    class_names = load_class_names()

    processed = preprocess_image(image_path)
    predictions = model.predict(processed, verbose=0)[0]  # shape: (num_classes,)

    # Urutkan index berdasarkan confidence tertinggi
    top_indices = predictions.argsort()[::-1][:top_k]

    top_predictions = [
        {
            "class": class_names[idx],
            "confidence": float(predictions[idx]),
        }
        for idx in top_indices
    ]

    result = {
        "predicted_class": top_predictions[0]["class"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }
    return result


def format_class_name(raw_name):
    """Ubah nama folder kelas (mis: 'nasi_goreng') menjadi label yang
    lebih enak dibaca (mis: 'Nasi Goreng')."""
    return raw_name.replace("_", " ").replace("-", " ").title()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan: python predict.py path/ke/gambar.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"[ERROR] File tidak ditemukan: {img_path}")
        sys.exit(1)

    result = predict_image(img_path)
    print("\nHasil Prediksi:")
    print(f"  Kelas Prediksi : {format_class_name(result['predicted_class'])}")
    print(f"  Confidence     : {result['confidence'] * 100:.2f}%")
    print("  Top Predictions:")
    for p in result["top_predictions"]:
        print(f"    - {format_class_name(p['class'])}: {p['confidence'] * 100:.2f}%")
