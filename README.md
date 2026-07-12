# Studi Komparatif Unsupervised NLP: Pemodelan Topik Distorsif pada Forum Komunitas Open-Source

## 1. Justifikasi Metodologis dan Korelasi Terhadap Silabus Utama
Proyek ini secara mandiri mengambil studi kasus spesifik di luar daftar judul konvensional yang disediakan oleh instruktur (seperti *"Klasifikasi Sentimen Teks"* atau *"Prediksi Kategori Berita Biner"*). 

Pemilihan topik eksplorasi komunitas OS Linux ini tetap memegang korelasi absolut terhadap capaian pembelajaran mata kuliah *Machine Learning*, namun ditingkatkan ke tingkat kompleksitas yang lebih tinggi melalui visualisasi berikut:

[Judul Standar Dosen]       ──> Analisis Teks Terstruktur (Supervised / Sentimen Kaku)
Vs.
[Topik Mandiri Kelompok]     ──> Pemodelan Pola Laten (Unsupervised NLP / Dinamis)


### Mengapa Topik Ini Dipilih? (Rasionalisasi Ilmiah)
1. **Kompleksitas Karakteristik Data:** Berbeda dengan dataset standar yang cenderung bersih, korpus data dari diskusi komunitas Linux memiliki tingkat distorsi semantik (*semantic noise*) yang tinggi, densitas jargon teknis yang padat, serta struktur sintaksis yang tidak baku. 
2. **Keterbatasan Pendekatan Supervised:** Judul konvensional berbasis *supervised learning* tidak mampu memetakan tren isu baru yang berkembang di komunitas tanpa adanya proses pelabelan manual (*labeling bottleneck*). Proyek ini menyelesaikan masalah tersebut secara absolut menggunakan pendekatan *unsupervised learning*.
3. **Pengujian Batas Efisiensi Algoritma:** Eksperimen ini menguji batas efisiensi dari dua paradigma matematika yang bertolak belakang dalam mengklasifikasikan 26.954 dokumen secara mandiri:
   * **Latent Dirichlet Allocation (LDA):** Pemodelan generatif probabilistik berbasis pendekatan distribusi Dirichlet dan inferensi Bayesian untuk memetakan hubungan probabilitas antara dokumen, topik, dan kata.
   * **Non-Negative Matrix Factorization (NMF):** Pemodelan deterministik berbasis faktorisasi aljabar linier tingkat tinggi untuk mereduksi dimensi matriks *term-document* menjadi dua matriks non-negatif yang merepresentasikan bobot fitur secara tegas.

---

## 2. Alur Kerja Sistem (Pipeline Arsitektur)
Sistem memproses data mentah melalui tahapan terstruktur berikut untuk memastikan hasil inferensi yang valid:

[Dataset Mentah] ──> [Text Preprocessing] ──> [Fitur Vektor (TF-IDF/Count)]
│
┌─────────────┴─────────────┐
▼                           ▼
[Engine 1: LDA]             [Engine 2: NMF]
│                           │
└─────────────┬─────────────┘
▼
[Dasbor Evaluasi Streamlit]


1. **Pra-Pemrosesan Korpus:** Pembersihan karakter non-alfabet, tokenisasi, eliminasi *stop-words* spesifik repositori Linux, dan lematisasi data teks.
2. **Ekstraksi Fitur:** Transformasi teks menjadi representasi numerik menggunakan *CountVectorizer* (untuk pemodelan berbasis LDA) dan *TF-IDF Vectorizer* (untuk pemodelan berbasis NMF).
3. **Komparasi Manifold:** Reduksi dimensi dan ekstraksi matriks fitur menjadi 5 klaster topik utama secara simultan untuk membandingkan stabilitas model pada dasbor interaktif.

---

## 3. Matriks Kontribusi Kolektif (Manajemen Proyek Setara)
Proyek ini diselesaikan secara swadaya oleh kelompok kecil yang terdiri dari 2 personel dengan prinsip kesetaraan beban kerja, transparansi, dan distribusi tanggung jawab yang berimbang:

| Personel | Ranah Fungsional Backend & Algoritma | Ranah Fungsional Frontend & Kosmetik |
| :--- | :--- | :--- |
| **M. Jafar Sodik** | - Desain Arsitektur *Dual-Engine* (LDA & NMF)<br>- Serialisasi Model Biner (`.pkl`) & Pipeline Ekstraksi<br>- Penyusunan Logika Inferensi Real-Time | - Integrasi *Custom Layout* & DOM Streamlit<br>- Arsitektur Kosmetik `style.css` (Linux UI)<br>- Implementasi Grafik Dinamis Plotly |
| **Umar** | - Pengadaan & Analisis Validitas Dataset 26K Records<br>- Pipeline *Text Preprocessing* & Tokenisasi<br>- Penyusunan Skrip Training Evaluasi Komparatif | - Manajemen Integrasi Modul *Bulk Processing* (CSV)<br>- Validasi Penanganan Error & Ketersediaan Server Cloud<br>- Dokumentasi Teknis Proyek |

---

## 4. Spesifikasi Berkas dan Repositori
* `app.py` — Pengendali utama antarmuka pengguna Streamlit, manajemen state, dan alur kerja inferensi model.
* `style.css` — Konfigurasi lembar gaya kustom untuk optimasi visual antarmuka bertema konsol terminal gelap.
* `requirements.txt` — Manifes dependensi library Python untuk standarisasi lingkungan eksekusi server cloud.
* `dataset.csv` — Korpus data utama berskala 26.954 record ulasan komunitas sebagai basis data pengujian.
* `model_lda.pkl` & `model_nmf.pkl` — Aset biner hasil serialisasi model latih.
* `vectorizer_lda.pkl` & `vectorizer_nmf.pkl` — Representasi matriks pembobotan fitur kata.

---

## 5. Replikasi Sistem Lokal

```bash
# Kloning repositori resmi
git clone https://github.com/sskrasa/project_machine_learning.git
cd project_machine_learning

# Menjalankan server lokal
streamlit run app.py
6. Referensi Ilmiah

    Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.

    Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755), 788-791.

Dokumentasi Teknis Proyek Akhir - Sistem Komputasi NLP Terintegrasi.
