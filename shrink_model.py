"""
shrink_model.py
================
Script sekali-pakai untuk memperkecil ukuran file model (.h5) dengan
menghapus state optimizer yang tidak diperlukan untuk inference/prediksi.
Jalankan ini SEKALI setelah training selesai, sebelum push ke GitHub.
"""

from tensorflow.keras.models import load_model
import config
import os

print(">> Memuat model...")
model = load_model(config.MODEL_PATH)

print(">> Menyimpan ulang tanpa optimizer state...")
model.save(config.MODEL_PATH, include_optimizer=False)
model.save(config.FINAL_MODEL_PATH, include_optimizer=False)

size_mb = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)
print(f">> Selesai. Ukuran model sekarang: {size_mb:.2f} MB")