"""
config.py
=========
Konfigurasi terpusat untuk seluruh project:
- Konfigurasi dataset (Kaggle via kagglehub)
- Konfigurasi preprocessing gambar
- Konfigurasi arsitektur & training model VGG16
- Konfigurasi path Flask (upload, model, dsb)

Semua modul (train.py, predict.py, app.py) mengimpor nilai dari file ini
supaya tidak ada duplikasi/incosistency konfigurasi antar file.
"""

import os

# ============================================================
# BASE PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# DATASET (KAGGLE)
# ============================================================
KAGGLE_DATASET_SLUG = "yourboys/indonesian-food"

# Direktori tempat subset dataset hasil sampling akan disimpan
# (dibuat otomatis oleh train.py, tidak perlu dibuat manual)
SUBSET_DATASET_DIR = os.path.join(BASE_DIR, "dataset_subset")

# Total jumlah gambar yang diambil dari keseluruhan dataset asli
# untuk mempercepat proses training (sesuai requirement: 3000-5000 gambar)
SUBSET_TOTAL_IMAGES = 4000

# Minimal & maksimal gambar per kelas ketika melakukan sampling subset
MIN_IMAGES_PER_CLASS = 30
MAX_IMAGES_PER_CLASS = 400

# Proporsi data validasi (diambil dari subset dataset)
VALIDATION_SPLIT = 0.2

# Random seed supaya hasil sampling & split reproducible
RANDOM_SEED = 42

# Ekstensi file gambar yang dikenali sebagai data valid
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ============================================================
# PREPROCESSING GAMBAR
# ============================================================
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE = 32
EPOCHS = 10                 # Maksimal 10 epoch sesuai requirement
LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5

# Jumlah layer terakhir VGG16 (dihitung dari belakang) yang akan
# di-unfreeze untuk fine-tuning ringan (blok konvolusi ke-5 VGG16)
FINE_TUNE_AT_LAYER = "block5_conv1"

# EarlyStopping & ModelCheckpoint
EARLY_STOPPING_PATIENCE = 3
MONITOR_METRIC = "val_accuracy"

# ============================================================
# PATH MODEL & ARTEFAK
# ============================================================
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")
TRAINING_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# FLASK APP
# ============================================================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

# Jumlah top-N prediksi yang ditampilkan di halaman hasil
TOP_K_PREDICTIONS = 3

SECRET_KEY = os.environ.get("SECRET_KEY", "ganti-secret-key-di-production")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
