"""
train.py
========
Script training model klasifikasi makanan Indonesia menggunakan
Transfer Learning VGG16 (pre-trained ImageNet) + fine-tuning ringan.

Alur proses:
1. Download dataset dari Kaggle menggunakan kagglehub
   (yourboys/indonesian-food)
2. Deteksi otomatis folder kelas di dalam dataset (struktur folder bisa
   bervariasi, termasuk split train/test/val terpisah, sehingga
   dilakukan pencarian folder yang berisi sub-folder kelas gambar
   dengan TOTAL GAMBAR terbanyak)
3. Ambil subset dataset (±3000-5000 gambar) secara proporsional per kelas
   agar training lebih cepat
4. Bangun ImageDataGenerator dengan augmentasi ringan + preprocess_input
   VGG16, split train/validation
5. Bangun model VGG16 (include_top=False) + custom classification head
6. Tahap 1: Training head saja (base VGG16 di-freeze)
7. Tahap 2: Fine-tuning ringan (unfreeze block5 VGG16) dengan learning
   rate lebih kecil
8. Callback EarlyStopping + ModelCheckpoint menyimpan model terbaik
9. Simpan model final, daftar nama kelas (class_names.json), dan
   riwayat training (grafik + json)

Cara menjalankan:
    python train.py
"""

import os
import json
import random
import shutil
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


# ============================================================
# 1. DOWNLOAD DATASET DARI KAGGLE
# ============================================================
def download_dataset():
    """Download dataset 'yourboys/indonesian-food' menggunakan kagglehub."""
    import kagglehub

    print(">> Mengunduh dataset dari Kaggle (yourboys/indonesian-food)...")
    path = kagglehub.dataset_download(config.KAGGLE_DATASET_SLUG)
    print(">> Path to dataset files:", path)
    return path


def find_class_folder_root(dataset_path):
    """
    Struktur dataset Kaggle terkadang memiliki folder pembungkus tambahan
    dan/atau split train/test/val terpisah (misal:
    dataset_path/dataset/train/<kelas>/*.jpg dan
    dataset_path/dataset/test/<kelas>/*.jpg).

    Fungsi ini menelusuri seluruh direktori secara rekursif dan memilih
    folder yang:
    1. Memiliki sub-folder kelas berisi gambar (folder valid)
    2. Di antara folder valid, dipilih yang memiliki TOTAL GAMBAR
       terbanyak (bukan sekadar jumlah kelas terbanyak) - ini penting
       supaya folder 'train' (biasanya lebih besar) diprioritaskan
       dibanding folder 'test'/'val' yang jumlah datanya lebih sedikit.
    """
    candidates = []  # list of (root, valid_class_dirs, total_images)

    for root, dirs, files in os.walk(dataset_path):
        if not dirs:
            continue

        valid_class_dirs = 0
        total_images = 0
        for d in dirs:
            sub_dir = os.path.join(root, d)
            if not os.path.isdir(sub_dir):
                continue
            image_count = sum(
                1
                for f in os.listdir(sub_dir)
                if f.lower().endswith(config.VALID_IMAGE_EXTENSIONS)
                and os.path.isfile(os.path.join(sub_dir, f))
            )
            if image_count > 0:
                valid_class_dirs += 1
                total_images += image_count

        if valid_class_dirs > 0:
            candidates.append((root, valid_class_dirs, total_images))

    if not candidates:
        raise RuntimeError(
            "Tidak ditemukan folder kelas berisi gambar di dalam dataset. "
            "Silakan periksa struktur dataset secara manual."
        )

    # Tampilkan semua kandidat yang terdeteksi supaya transparan di log
    print(">> Kandidat folder kelas yang terdeteksi:")
    for root, n_classes, n_images in sorted(candidates, key=lambda c: -c[2]):
        print(f"   - {root}  ({n_classes} kelas, {n_images} gambar)")

    # Prioritas: jumlah TOTAL GAMBAR terbanyak (bukan sekadar jumlah kelas)
    best_candidate, best_num_classes, best_num_images = max(
        candidates, key=lambda c: c[2]
    )

    print(
        f">> Dipilih folder: {best_candidate} "
        f"({best_num_classes} kelas, {best_num_images} gambar)"
    )
    return best_candidate


# ============================================================
# 2. BUAT SUBSET DATASET (3000-5000 GAMBAR)
# ============================================================
def build_subset_dataset(class_root):
    """
    Membuat subset dataset dengan mengambil sampel gambar secara
    proporsional dari tiap kelas, lalu menyalinnya (copy) ke folder
    config.SUBSET_DATASET_DIR/<nama_kelas>/.
    """
    if os.path.exists(config.SUBSET_DATASET_DIR):
        print(">> Subset dataset sudah ada, menghapus subset lama...")
        shutil.rmtree(config.SUBSET_DATASET_DIR)
    os.makedirs(config.SUBSET_DATASET_DIR, exist_ok=True)

    class_names = sorted(
        d for d in os.listdir(class_root)
        if os.path.isdir(os.path.join(class_root, d))
    )
    # Hanya pertahankan kelas yang benar-benar memiliki gambar
    class_names = [
        c for c in class_names
        if any(
            f.lower().endswith(config.VALID_IMAGE_EXTENSIONS)
            for f in os.listdir(os.path.join(class_root, c))
        )
    ]

    num_classes = len(class_names)
    if num_classes == 0:
        raise RuntimeError("Tidak ada kelas yang valid ditemukan pada dataset.")

    target_per_class = max(
        config.MIN_IMAGES_PER_CLASS,
        min(config.MAX_IMAGES_PER_CLASS, config.SUBSET_TOTAL_IMAGES // num_classes),
    )

    print(f">> Jumlah kelas terdeteksi: {num_classes}")
    print(f">> Target gambar per kelas (subset): {target_per_class}")

    total_copied = 0
    final_class_names = []
    for cls in class_names:
        src_dir = os.path.join(class_root, cls)
        images = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith(config.VALID_IMAGE_EXTENSIONS)
        ]
        if len(images) < 5:
            # kelas dengan gambar terlalu sedikit dilewati agar training stabil
            print(f"   - Lewati kelas '{cls}' (hanya {len(images)} gambar)")
            continue

        random.shuffle(images)
        selected = images[:target_per_class]

        dst_dir = os.path.join(config.SUBSET_DATASET_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for img_name in selected:
            shutil.copy2(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))

        total_copied += len(selected)
        final_class_names.append(cls)
        print(f"   - {cls}: {len(selected)} gambar")

    print(f">> Total gambar dalam subset: {total_copied}")
    return config.SUBSET_DATASET_DIR, final_class_names


# ============================================================
# 3. DATA GENERATOR (AUGMENTASI + PREPROCESS_INPUT VGG16)
# ============================================================
def build_data_generators(subset_dir):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        validation_split=config.VALIDATION_SPLIT,
    )

    # Generator validasi hanya perlu preprocess_input, tanpa augmentasi
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=config.VALIDATION_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        subset_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=config.RANDOM_SEED,
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        subset_dir,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=config.RANDOM_SEED,
        shuffle=False,
    )

    return train_generator, val_generator


# ============================================================
# 4. BANGUN MODEL VGG16 + CUSTOM HEAD
# ============================================================
def build_model(num_classes):
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=config.INPUT_SHAPE,
    )
    base_model.trainable = False  # freeze seluruh base model pada tahap 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def unfreeze_for_fine_tuning(base_model):
    """Unfreeze layer mulai dari FINE_TUNE_AT_LAYER (block5) untuk fine-tuning ringan."""
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == config.FINE_TUNE_AT_LAYER:
            set_trainable = True
        layer.trainable = set_trainable
    return base_model


# ============================================================
# 5. TRAINING
# ============================================================
def get_callbacks():
    return [
        EarlyStopping(
            monitor=config.MONITOR_METRIC,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor=config.MONITOR_METRIC,
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=config.MONITOR_METRIC,
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def main():
    print("=" * 60)
    print("TRAINING - Klasifikasi Makanan Indonesia (VGG16 Transfer Learning)")
    print("=" * 60)

    # 1) Download dataset
    dataset_path = download_dataset()

    # 2) Cari folder kelas
    class_root = find_class_folder_root(dataset_path)

    # 3) Buat subset dataset
    subset_dir, class_names = build_subset_dataset(class_root)
    num_classes = len(class_names)

    # 4) Data generator
    train_generator, val_generator = build_data_generators(subset_dir)

    # Simpan mapping index -> nama kelas berdasarkan urutan generator (bukan urutan manual)
    # supaya konsisten dengan label yang dipakai saat training
    class_indices = train_generator.class_indices  # {nama_kelas: index}
    idx_to_class = {v: k for k, v in class_indices.items()}
    ordered_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(ordered_class_names, f, ensure_ascii=False, indent=2)
    print(f">> Daftar kelas disimpan di: {config.CLASS_NAMES_PATH}")

    # 5) Bangun model
    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = get_callbacks()

    # ---------------------------------------------------
    # TAHAP 1: Training head (base model freeze)
    # ---------------------------------------------------
    print("\n>> TAHAP 1: Training classification head (VGG16 base freeze)...")
    history_1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=max(1, config.EPOCHS // 2),
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------------------------------------------
    # TAHAP 2: Fine-tuning ringan (unfreeze block5)
    # ---------------------------------------------------
    print("\n>> TAHAP 2: Fine-tuning ringan (unfreeze block5 VGG16)...")
    base_model = unfreeze_for_fine_tuning(base_model)
    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    remaining_epochs = max(1, config.EPOCHS - len(history_1.history["loss"]))
    history_2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=remaining_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------------------------------------------
    # SIMPAN MODEL FINAL & RIWAYAT TRAINING
    # ---------------------------------------------------
    model.save(config.FINAL_MODEL_PATH)
    print(f">> Model final disimpan di: {config.FINAL_MODEL_PATH}")
    print(f">> Model terbaik (checkpoint) disimpan di: {config.MODEL_PATH}")

    combined_history = {
        "accuracy": history_1.history.get("accuracy", []) + history_2.history.get("accuracy", []),
        "val_accuracy": history_1.history.get("val_accuracy", []) + history_2.history.get("val_accuracy", []),
        "loss": history_1.history.get("loss", []) + history_2.history.get("loss", []),
        "val_loss": history_1.history.get("val_loss", []) + history_2.history.get("val_loss", []),
    }
    with open(config.TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(combined_history["accuracy"], label="Train Accuracy")
        plt.plot(combined_history["val_accuracy"], label="Val Accuracy")
        plt.title("Akurasi Training")
        plt.xlabel("Epoch")
        plt.ylabel("Akurasi")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(combined_history["loss"], label="Train Loss")
        plt.plot(combined_history["val_loss"], label="Val Loss")
        plt.title("Loss Training")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(config.TRAINING_PLOT_PATH)
        print(f">> Grafik training disimpan di: {config.TRAINING_PLOT_PATH}")
    except ImportError:
        print(">> matplotlib tidak tersedia, grafik training dilewati.")

    print("\n>> TRAINING SELESAI.")
    print(f">> Jumlah kelas: {num_classes}")
    print(f">> Kelas: {ordered_class_names}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Training gagal: {e}", file=sys.stderr)
        raise