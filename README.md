# 🍛 NusantaraVision — Klasifikasi Jenis Makanan Indonesia (VGG16 Transfer Learning)

Aplikasi web untuk mengklasifikasikan jenis makanan Indonesia dari sebuah foto,
menggunakan **Transfer Learning VGG16** (pre-trained ImageNet + fine-tuning ringan),
dibungkus dalam aplikasi **Flask** dengan tampilan **Bootstrap 5** yang bersih dan modern.

---

## ✨ Fitur

- 📤 Upload gambar makanan (drag & drop atau klik pilih file)
- 🖼️ Preview gambar sebelum diprediksi
- 🤖 Prediksi jenis makanan menggunakan model VGG16 hasil transfer learning
- 📊 Confidence score + Top-3 prediksi
- ℹ️ Halaman **About** berisi detail arsitektur model & dataset
- 📱 Tampilan responsif (mobile-friendly) dengan Bootstrap 5
- 🚀 Siap deploy ke **Railway** (juga kompatibel dengan Render)

---

## 🗂️ Struktur Project

```
food_classifier/
├── app.py                  # Aplikasi Flask utama (routing, upload, render halaman)
├── train.py                 # Script training model (download dataset, augmentasi, transfer learning)
├── predict.py                # Modul inferensi/prediksi gambar
├── config.py                 # Konfigurasi terpusat (path, hyperparameter, dsb)
├── requirements.txt           # Daftar dependency Python
├── Procfile                   # Perintah start untuk Railway/Render/Heroku
├── runtime.txt                 # Versi Python untuk platform deploy
├── render.yaml                  # Konfigurasi deploy Render (opsional)
├── railway.json                  # Konfigurasi deploy Railway (opsional)
├── .gitignore
├── README.md
├── templates/
│   ├── base.html               # Layout dasar (navbar, footer)
│   ├── index.html                # Halaman upload + hasil prediksi
│   ├── about.html                 # Halaman tentang project
│   └── 404.html                    # Halaman error
├── static/
│   ├── css/style.css               # Styling custom (tema clean & modern)
│   ├── js/main.js                   # Interaksi upload, preview, drag & drop
│   └── uploads/                      # Folder penyimpanan gambar yang diupload user
├── model/
│   ├── best_model.h5                 # (dihasilkan setelah training) model terbaik
│   ├── final_model.h5                 # (dihasilkan setelah training) model akhir
│   └── class_names.json                # (dihasilkan setelah training) daftar label kelas
└── notebooks/                           # (opsional) tempat eksperimen/EDA
```

---

## ⚙️ Instalasi & Setup Lokal

### 1. Clone / salin project & buat virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependency

```bash
pip install -r requirements.txt
```

> 💡 Jika ingin training menggunakan GPU, install `tensorflow` versi penuh
> (bukan `tensorflow-cpu`) sesuai dengan driver CUDA/cuDNN di komputer Anda.

### 3. Setup akun & API key Kaggle (untuk kagglehub)

`kagglehub` membutuhkan kredensial Kaggle API. Cara termudah:

1. Buat akun di [kaggle.com](https://www.kaggle.com)
2. Masuk ke **Account Settings → API → Create New Token**, lalu file `kaggle.json`
   akan terunduh
3. Letakkan file tersebut di:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
4. Atau set environment variable `KAGGLE_USERNAME` dan `KAGGLE_KEY`

### 4. Jalankan training model

```bash
python train.py
```

Script `train.py` akan otomatis:
1. Mengunduh dataset `yourboys/indonesian-food` via `kagglehub`
2. Mendeteksi folder kelas di dalam dataset
3. Mengambil subset ±3.000–5.000 gambar secara proporsional per kelas
4. Melatih model VGG16 (freeze base → fine-tuning ringan block5) maksimal 10 epoch
   dengan EarlyStopping & ModelCheckpoint
5. Menyimpan model terbaik ke `model/best_model.h5` dan daftar kelas ke
   `model/class_names.json`

### 5. Jalankan aplikasi web

```bash
python app.py
```

Buka browser ke `http://localhost:5000`

---

## 🧠 Detail Model

| Komponen | Keterangan |
|---|---|
| Base Model | VGG16 (`weights="imagenet"`, `include_top=False`) |
| Input Shape | 224 × 224 × 3 |
| Head | GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(128, ReLU) → Dropout(0.3) → Dense(N, Softmax) |
| Tahap 1 | Training head saja, base model di-freeze |
| Tahap 2 | Fine-tuning ringan: unfreeze mulai `block5_conv1` dengan learning rate lebih kecil |
| Optimizer | Adam (lr 1e-4 → 1e-5 saat fine-tuning) |
| Loss | Categorical Crossentropy |
| Callback | EarlyStopping (`val_accuracy`, patience=3), ModelCheckpoint (save best), ReduceLROnPlateau |
| Epoch maksimal | 10 (gabungan tahap 1 + tahap 2) |

### Preprocessing & Augmentasi

- Resize gambar ke 224×224
- `preprocess_input` bawaan `tensorflow.keras.applications.vgg16`
- Augmentasi (hanya pada data training): rotasi ±20°, shift 15%, shear, zoom,
  flip horizontal, variasi brightness

---

## 🌐 Deploy ke Railway

1. Push project ini ke repository GitHub
2. Buka [railway.app](https://railway.app) → **New Project → Deploy from GitHub Repo**
3. Railway akan otomatis mendeteksi `Procfile` / `railway.json` dan menjalankan:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
   ```
4. Set environment variable berikut di dashboard Railway (tab **Variables**):
   - `SECRET_KEY` → string acak untuk keamanan sesi Flask
   - `FLASK_DEBUG` → `0`
5. **Penting:** pastikan folder `model/` (berisi `best_model.h5` /
   `final_model.h5` dan `class_names.json`) ikut ter-commit ke repository,
   karena Railway tidak menjalankan `train.py` secara otomatis saat build.
   Jalankan training di lokal terlebih dahulu, lalu commit hasil modelnya.
6. Klik **Deploy** — aplikasi akan tersedia di URL publik yang diberikan Railway.

> Project ini juga menyertakan `render.yaml` sehingga bisa langsung
> di-deploy pula ke [Render](https://render.com) apabila diperlukan.

---

## 🔌 API Endpoint (opsional)

Selain form upload di halaman utama, tersedia juga endpoint JSON API sederhana:

```
POST /api/predict
Content-Type: multipart/form-data
Body: file=<gambar>
```

Contoh dengan `curl`:

```bash
curl -X POST -F "file=@contoh_makanan.jpg" http://localhost:5000/api/predict
```

Response:

```json
{
  "predicted_class": "rendang",
  "confidence": 0.9421,
  "top_predictions": [
    {"class": "rendang", "confidence": 0.9421},
    {"class": "semur_daging", "confidence": 0.0312},
    {"class": "gulai", "confidence": 0.0134}
  ]
}
```

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
|---|---|
| `Model belum tersedia` saat prediksi | Jalankan `python train.py` terlebih dahulu |
| Error saat `kagglehub.dataset_download` | Pastikan `kaggle.json` sudah terpasang dengan benar |
| Training sangat lambat / OOM | Kurangi `SUBSET_TOTAL_IMAGES` atau `BATCH_SIZE` di `config.py` |
| Deploy Railway gagal build (memori) | Gunakan `tensorflow-cpu` (sudah default di `requirements.txt`) |
| Upload gagal (413) | Ukuran file melebihi 5MB, kompres gambar terlebih dahulu |

---

## 📄 Lisensi

Project ini dibuat untuk tujuan edukasi/pembelajaran penerapan Transfer Learning
pada klasifikasi citra makanan Indonesia.
