/**
 * main.js
 * Interaksi front-end untuk halaman upload:
 * - Preview gambar sebelum submit
 * - Drag & drop file ke dropzone
 * - Validasi ukuran & tipe file di sisi client (validasi utama tetap di server)
 * - Loading state pada tombol submit saat form dikirim
 */

document.addEventListener("DOMContentLoaded", function () {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzoneContent = document.getElementById("dropzoneContent");
    const previewContent = document.getElementById("previewContent");
    const previewImage = document.getElementById("previewImage");
    const previewFileName = document.getElementById("previewFileName");
    const uploadForm = document.getElementById("uploadForm");
    const submitBtn = document.getElementById("submitBtn");

    if (!dropzone || !fileInput) return;

    const MAX_SIZE_BYTES = 5 * 1024 * 1024; // 5 MB
    const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];

    function showPreview(file) {
        if (!ALLOWED_TYPES.includes(file.type)) {
            alert("Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau WEBP.");
            fileInput.value = "";
            return;
        }
        if (file.size > MAX_SIZE_BYTES) {
            alert("Ukuran file terlalu besar. Maksimal 5MB.");
            fileInput.value = "";
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewFileName.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
            dropzoneContent.classList.add("d-none");
            previewContent.classList.remove("d-none");
        };
        reader.readAsDataURL(file);
    }

    // Klik pada dropzone -> buka file picker (handled by <label for>),
    // tapi tetap listen perubahan file input untuk preview.
    fileInput.addEventListener("change", function () {
        if (fileInput.files && fileInput.files[0]) {
            showPreview(fileInput.files[0]);
        }
    });

    // Drag & drop events
    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", function (e) {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            fileInput.files = files;
            showPreview(files[0]);
        }
    });

    // Loading state saat form disubmit
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener("submit", function () {
            if (!fileInput.files || fileInput.files.length === 0) {
                return;
            }
            submitBtn.disabled = true;
            submitBtn.innerHTML =
                '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Menganalisis Gambar...';
        });
    }
});
