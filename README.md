# Benchmarking Strategi Kuantisasi Model VGG16 pada Klasifikasi Citra CIFAR-10

Repositori ini memuat seluruh artefak kode sumber, bobot model terlatih (*checkpoints*), log eksperimen deterministik, serta visualisasi evaluasi dari penelitian benchmarking teknik kompresi model berbasis **Quantization (INT8)** pada arsitektur **VGG16** untuk tugas klasifikasi citra pada dataset **CIFAR-10**. 

Eksperimen ini secara komprehensif membandingkan tiga paradigma reduksi presisi numerik:
1. **Post-Training Static Quantization (PTSQ)**
2. **Post-Training Dynamic Quantization (PTDQ)**
3. **Quantization-Aware Training (QAT)**

Evaluasi dilakukan lintas kerangka kerja (*framework*) menggunakan pustaka **`torch.ao`** (PyTorch) dan format portabilitas **ONNX (ONNX Runtime)**, dengan fokus pengujian latensi inferensi secara ketat pada lingkungan **CPU** guna mensimulasikan karakteristik perangkat *edge computing* yang terbatas sumber daya.

---

## 📁 Struktur Direktori dan Manajemen Berkas

Untuk menjaga efisiensi ukuran repositori GitHub, berkas model biner berukuran besar (>100 MB) dan dataset mentah disimpan di Google Drive. Repositori ini menyediakan kode program, catatan log eksperimen, dan tautan akses langsung ke penyimpanan eksternal tersebut.

```text
├── experiment_logs/
│   ├── confusion_matrices/
│   ├── qat_loss_curves/
│   ├── Analisis Degradasi Akurasi dari Quantization/
│   ├── cifar10_quantization_benchmark.csv
│   └── cifar10_quantization_summary.csv
├── Quantization Effect in Image Classification (torch.ao).ipynb
├── Quantization Effect in Image Classification (ONNX).ipynb
└── README.md
```

### Rincian Berkas dalam Repositori GitHub

| Berkas / Direktori | Deskripsi |
| :--- | :--- |
| **`experiment_logs/`** | Menyimpan seluruh catatan data mentah (*raw logs*) dan agregasi eksperimen multi-seed (format `.csv` dan `.json`), grafik konvergensi loss QAT, visualisasi *Confusion Matrix* resolusi tinggi, serta tabulasi analisis degradasi akurasi per kelas. |
| **`Quantization Effect in Image Classification (torch.ao).ipynb`** | Notebook Jupyter yang memuat alur kerja lengkap pelatihan baseline FP32, integrasi modul `torch.ao`, kalibrasi observer, fine-tuning QAT (10 epoch), analisis multi-seed, serta benchmarking latensi CPU PyTorch. |
| **`Quantization Effect in Image Classification (ONNX).ipynb`** | Notebook Jupyter yang memuat alur kerja konversi model ke ONNX, proses *symbolic shape inference*, kalibrasi QDQ static quantization, dynamic quantization, serta benchmarking efisiensi menggunakan ONNX Runtime CPU. |

---

## 💾 Akses Dataset dan Model Terlatih (Google Drive)

Mengingat ukuran berkas model konvolusional VGG16 (FP32) dan arsip dataset melebihi batas penyimpanan GitHub (>100 MB), seluruh berkas biner tersebut disimpan di Google Drive dan dapat diunduh melalui tautan berikut:

| Direktori / Artefak Data | Deskripsi | Tautan Akses Google Drive |
| :--- | :--- | :---: |
| **CIFAR10 Dataset** | Arsip data citra CIFAR-10 (60.000 citra beresolusi $32 \times 32$, 10 kelas) yang diunduh melalui modul `torchvision.datasets`. | [📂 Akses Folder Dataset](https://drive.google.com/drive/folders/13x2PZnp1b_JBZdCdyXk2rvaoOsdwtMWQ?usp=sharing) |
| **VGG16 Model Saved (Linear Logit)** | Kumpulan bobot model *checkpoint* VGG16 FP32 per epoch (1–20) dengan *output layer* linier (*raw logits*) sebagai **baseline utama** yang konvergen. | [📂 Akses Folder Model Linear Logit](https://drive.google.com/drive/folders/1s5vLqQ9wVtelTqDEcpnUEd-XcD8TtgZG?usp=sharing) |
| **VGG16 Model Saved (Softmax)** | Kumpulan bobot model VGG16 FP32 per epoch (1–20) dengan fungsi aktivasi Softmax eksplisit (untuk analisis fenomena *vanishing gradient*). | [📂 Akses Folder Model Softmax](https://drive.google.com/drive/folders/1yQQttJaC1gUDw2DN2wXWlh2LwBjeMike?usp=sharing) |
| **VGG16 Model Saved (Quantization)** | Seluruh artefak model hasil konversi kuantisasi INT8 (`.pth` dan `.onnx`) mencakup metode PTSQ, PTDQ, dan QAT untuk kedua framework. | [📂 Akses Folder Model Terkuantisasi](https://drive.google.com/drive/folders/1zIhonZuC8y80Pm2HxlVPbrijHtyodrPz?usp=sharing) |

---

## 🔬 Sorotan Metodologi & Temuan Kunci

* **Protokol Multi-Seed Deterministik:** Seluruh eksperimen dieksekusi secara independen melintasi 5 *random initialization seeds* (`42, 123, 456, 1337, 2024`) untuk menjamin signifikansi statistik dan reproduktibilitas penuh.
* **Benchmarking CPU Murni:** Pengukuran latensi inferensi dijalankan secara eksklusif pada CPU standar Google Colab (Intel Xeon @ 2.20 GHz, *single-threaded*, `batch_size = 1`, 100 *warm-up runs*) guna mengeliminasi bias akselerasi GPU.
* **Konfigurasi Pareto Optimal:** Berdasarkan skor efisiensi multi-kriteria (*Efficiency Score* $E$), **ONNX-based PTSQ** muncul sebagai konfigurasi paling unggul ($E = 14.88$) dengan reduksi ukuran penyimpanan sebesar **74.94%** (dari 128.42 MB ke 32.17 MB), latensi inferensi mencapai **5.11 ± 0.06 ms** (**4.55x speedup** terhadap PyTorch baseline), dan retensi akurasi rata-rata yang kokoh (**83.86% ± 1.66%**, $p > 0.05$).

---

## 🚀 Panduan Menjalankan Eksperimen

1. **Kloning Repositori:**
   ```bash
   git clone https://github.com/rehgend/Quantization-VGG16.git
   cd Quantization-VGG16
   ```
2. **Pemasangan Dependensi:**
   ```bash
   pip install torch torchvision torchinfo onnx onnxruntime onnxscript scikit-learn seaborn matplotlib pandas
   ```
3. **Eksekusi Eksperimen PyTorch (`torch.ao`):**
   Buka dan jalankan seluruh sel pada notebook `Quantization Effect in Image Classification (torch.ao).ipynb` untuk mereproduksi baseline FP32, model terkuantisasi `torch.ao`, dan pencatatan log.
4. **Eksekusi Eksperimen ONNX Runtime:**
   Jalankan notebook `Quantization Effect in Image Classification (ONNX).ipynb` untuk mengekspor model PyTorch yang telah terlatih ke dalam graf ONNX, menjalankan inferensi bentuk simbolik, dan mengeksekusi kuantisasi statis/dinamis.

---

## 📜 Lisensi & Sitasi
Repositori ini dilisensikan di bawah lisensi terbuka untuk tujuan akademik dan reproduksi penelitian. Jika Anda menggunakan kode, log, atau model dari penelitian ini, silakan kutip artikel terkait:

```bibtex
@article{rehgend2026vgg16quantization,
  title={Impact of Quantization Techniques on Visual Geometry Group (VGG) Inference for The CIFAR-10 Dataset},
  author={Rahman Indra Kesuma, I Wayan Wiprayoga Wisesa, Meida Cahyo Untoro, Arief Ichwani},
  journal={Journal of Applied Informatics and Computing},
  year={2026},
  url={......}
}
