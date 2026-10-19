# Benchmarking Strategi Kuantisasi Model VGG16 pada Klasifikasi Citra CIFAR-10

Repositori ini memuat seluruh artefak kode sumber, bobot model terlatih (*checkpoints*), log eksperimen deterministik, serta visualisasi evaluasi dari penelitian benchmarking teknik kompresi model berbasis **Quantization (INT8)** pada arsitektur **VGG16** untuk tugas klasifikasi citra pada dataset **CIFAR-10**. 

Eksperimen ini secara komprehensif membandingkan tiga paradigma reduksi presisi numerik:
1. **Post-Training Static Quantization (PTSQ)**
2. **Post-Training Dynamic Quantization (PTDQ)**
3. **Quantization-Aware Training (QAT)**

Evaluasi dilakukan lintas kerangka kerja (*framework*) menggunakan pustaka **`torch.ao`** (PyTorch) dan format portabilitas **ONNX (ONNX Runtime)**, dengan fokus pengujian latensi inferensi secara ketat pada lingkungan **CPU** guna mensimulasikan karakteristik perangkat *edge computing* yang terbatas sumber daya.

---

## 📁 Struktur Direktori dan Manajemen Berkas

Repositori ini diorganisasikan ke dalam direktori-direktori berikut untuk mempermudah reproduktibilitas penelitian:

```text
├── CIFAR10 Dataset/
├── experiment_logs/
│   ├── confusion_matrices/
│   ├── qat_loss_curves/
│   ├── Analisis Degradasi Akurasi dari Quantization/
│   ├── cifar10_quantization_benchmark.csv
│   └── cifar10_quantization_summary.csv
├── VGG16 Model Saved (Linear Logit)/
├── VGG16 Model Saved (Softmax)/
├── VGG16 Model Saved (Quantization)/
├── Quantization Effect in Image Classification (torch.ao).ipynb
├── Quantization Effect in Image Classification (ONNX).ipynb
└── README.md
```

### Rincian Modul dan Direktori

| Direktori / Berkas | Deskripsi |
| :--- | :--- |
| **`CIFAR10 Dataset/`** | Menyimpan arsip data citra CIFAR-10 (60.000 citra beresolusi $32 \times 32$ piksel, 10 kelas) yang diunduh dan dipartisi secara otomatis melalui modul `torchvision.datasets`. |
| **`experiment_logs/`** | Menyimpan seluruh catatan data mentah (*raw logs*) dan agregasi eksperimen multi-seed (format `.csv` dan `.json`), termasuk grafik konvergensi loss QAT, visualisasi *Confusion Matrix* (resolusi tinggi), serta tabulasi analisis degradasi akurasi per kelas. |
| **`VGG16 Model Saved (Linear Logit)/`** | Menyimpan *checkpoint* bobot model VGG16 FP32 per epoch (1–20) yang dilatih menggunakan konfigurasi *output layer* linier (*raw logits*). Model ini berfungsi sebagai **baseline utama** yang stabil dan konvergen. |
| **`VGG16 Model Saved (Softmax)/`** | Menyimpan *checkpoint* bobot model VGG16 FP32 per epoch (1–20) dengan fungsi aktivasi eksplisit Softmax pada *output layer*, yang digunakan untuk menganalisis fenomena *vanishing gradient* dan saturasi matematis terhadap *Cross-Entropy Loss*. |
| **`VGG16 Model Saved (Quantization)/`** | Berisi seluruh artefak model hasil konversi kuantisasi INT8, baik berformat PyTorch (`.pth`) maupun ONNX (`.onnx`), mencakup metode PTSQ, PTDQ, dan QAT. |
| **`Quantization Effect in Image Classification (torch.ao).ipynb`** | Notebook Jupyter yang memuat alur kerja lengkap pelatihan baseline FP32, integrasi modul `torch.ao`, kalibrasi observer, fine-tuning QAT (10 epoch), serta benchmarking latensi CPU PyTorch. |
| **`Quantization Effect in Image Classification (ONNX).ipynb`** | Notebook Jupyter yang memuat alur kerja konversi model ke ONNX, proses *symbolic shape inference*, kalibrasi QDQ static quantization, dynamic quantization, serta benchmarking efisiensi menggunakan ONNX Runtime CPU. |

---

## 🔬 Sorotan Metodologi & Temuan Kunci

* **Protokol Multi-Seed Deterministik:** Seluruh eksperimen dieksekusi secara independen melintasi 5 *random initialization seeds* (`42, 123, 456, 1337, 2024`) untuk menjamin signifikansi statistik dan reproduktibilitas penuh.
* **Benchmarking CPU Murni:** Pengukuran latensi inferensi dijalankan secara eksklusif pada CPU standar Google Colab (Intel Xeon @ 2.20 GHz, *single-threaded*, `batch_size = 1`, 100 *warm-up runs*) guna mengeliminasi bias akselerasi GPU.
* **Konfigurasi Pareto Optimal:** Berdasarkan skor efisiensi multi-kriteria (*Efficiency Score* $E$), **ONNX-based PTSQ** muncul sebagai konfigurasi paling unggul ($E = 14.88$) dengan reduksi ukuran penyimpanan sebesar **74.94%** (dari 128.42 MB ke 32.17 MB), latensi inferensi mencapai **5.11 ± 0.06 ms** (**4.55x speedup** terhadap PyTorch baseline), dan retensi akurasi rata-rata yang kokoh (**83.86% ± 1.66%**, $p > 0.05$).

---

## 🚀 Panduan Menjalankan Eksperimen

1. **Persiapan Lingkungan:**
   Pastikan dependensi utama terpasang sesuai dengan spesifikasi lingkungan eksperimen:
   ```bash
   pip install torch torchvision torchinfo onnx onnxruntime onnxscript scikit-learn seaborn matplotlib pandas
   ```
2. **Eksekusi Eksperimen PyTorch (`torch.ao`):**
   Buka dan jalankan seluruh sel pada notebook `Quantization Effect in Image Classification (torch.ao).ipynb` untuk mereproduksi baseline FP32, model terkuantisasi `torch.ao`, dan pencatatan log.
3. **Eksekusi Eksperimen ONNX Runtime:**
   Jalankan notebook `Quantization Effect in Image Classification (ONNX).ipynb` untuk mengekspor model PyTorch yang telah terlatih ke dalam graf ONNX, menjalankan inferensi bentuk simbolik, dan mengeksekusi kuantisasi statis/dinamis.

---

## 📜 Lisensi & Sitasi
Repositori ini ditujukan untuk kebutuhan akademik dan penelitian terbuka. Jika Anda menggunakan kode atau model dari repositori ini, silakan sitasi artikel terkait sesuai rujukan publikasi pada manuskrip.
