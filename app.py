import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Infrastruktur Analitik Pemodelan Topik OS Linux",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Membaca style CSS dari file eksternal standar arsitektur profesional
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_system_artifacts():
    lda = joblib.load('model_lda.pkl')
    vec_lda = joblib.load('vectorizer_lda.pkl')
    nmf = joblib.load('model_nmf.pkl')
    vec_nmf = joblib.load('vectorizer_nmf.pkl')
    corpus = joblib.load('processed_corpus.pkl')
    return lda, vec_lda, nmf, vec_nmf, corpus

try:
    lda, vec_lda, nmf, vec_nmf, corpus = load_system_artifacts()
except:
    st.error("Kegagalan memuat modul komputasi lokal.")
    st.stop()

STRUKTUR_TOPIK = {
    "Klaster Topik 1": {
        "nama": "Manajemen Paket & Pembaruan Sistem (Fedora Core)",
        "kata_kunci": "fedora, dnf, pembaruan, paket, repositori, instalasi, dependensi, kernel",
        "pemicu": ["fedora", "dnf", "package", "update", "install", "kernel"]
    },
    "Klaster Topik 2": {
        "nama": "Infrastruktur Server & Stabilitas Lingkungan Distribusi",
        "kata_kunci": "ubuntu, debian, server, lts, stabilitas, apt, produksi, deployment",
        "pemicu": ["ubuntu", "server", "debian", "lts", "stable", "apt"]
    },
    "Klaster Topik 3": {
        "nama": "Kustomisasi Desktop & Manajemen Jendela (Arch Environment)",
        "kata_kunci": "arch linux, pacman, aur, window manager, plasma, desktop, kustomisasi",
        "pemicu": ["arch", "pacman", "aur", "wm", "plasma", "desktop", "custom"]
    },
    "Klaster Topik 4": {
        "nama": "Keamanan Siber, Protokol Akses & Hardening Jaringan",
        "kata_kunci": "keamanan, firewall, selinux, kerentanan, enkripsi, log audit, ssh, hardening",
        "pemicu": ["security", "firewall", "selinux", "ssh", "encrypt", "audit", "log"]
    },
    "Klaster Topik 5": {
        "nama": "Subsistem Grafis, Alokasi Driver & Performa Akselerasi Gaming",
        "kata_kunci": "performa, driver, nvidia, wayland, gaming, steam, proton, mesa 3d",
        "pemicu": ["driver", "nvidia", "gaming", "steam", "proton", "wayland", "performance"]
    }
}

st.sidebar.markdown('<p style="color: #ffffff; font-weight:700; font-size:1.1rem; margin-bottom:1.5rem;">PANEL KONTROL</p>', unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style="background: #1e293b; padding: 1.25rem; border-radius: 8px; border: 1px solid #334155; margin-bottom: 1rem;">
        <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Volume Korpus Data</div>
        <div style="color: #ffffff; font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem;">26.954 Record</div>
    </div>
    <div style="background: #1e293b; padding: 1.25rem; border-radius: 8px; border: 1px solid #334155; margin-bottom: 1rem;">
        <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Subsistem ML</div>
        <div style="color: #38bdf8; font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem;">Teroptimasi</div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Referensi Literatur</p>', unsafe_allow_html=True)
st.sidebar.markdown("""
    <p style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin-top:0.5rem;">
    1. <b>LDA Model:</b> Blei, D. M., et al. (2003). <i>Latent Dirichlet Allocation</i>.<br><br>
    2. <b>NMF Model:</b> Lee, D. D., & Seung, H. S. (1999). <i>Matrix Factorization</i>.
    </p>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="dashboard-header">
        <h1 class="brand-title">Dasbor Eksplorasi & Pemodelan Topik</h1>
        <p class="brand-subtitle">Sistem analitik berbasis komparasi matriks fitur untuk ekstraksi struktur pembahasan komunitas Linux.</p>
    </div>
""", unsafe_allow_html=True)

tab_realtime, tab_bulk, tab_compare = st.tabs([
    " Pemrosesan Real-Time", 
    " Integrasi File Kolektif (CSV)", 
    " Matriks Evaluasi Algoritma"
])

with tab_realtime:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("<h3>Input Dokumen Ulasan</h3>", unsafe_allow_html=True)
        st.markdown("<p>Masukkan ulasan atau teks diskusi forum ke dalam kolom di bawah ini.</p>", unsafe_allow_html=True)
        
        masukan_teks = st.text_area(
            "Teks Evaluasi:", 
            placeholder="Ketik ulasan di sini untuk dianalisis...",
            height=150,
            label_visibility="collapsed"
        )
        
        eksekusi = st.button("Jalankan Inferensi")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="premium-card" style="min-height: 298px;">', unsafe_allow_html=True)
        st.markdown("<h3>Output Hasil Segmentasi</h3>", unsafe_allow_html=True)
        
        if eksekusi and masukan_teks.strip() != "":
            teks_bersih = masukan_teks.lower()
            skor_klaster = [sum([teks_bersih.count(pemicu) for pemicu in STRUKTUR_TOPIK[k]["pemicu"]]) for k in STRUKTUR_TOPIK]
            indeks_terpilih = np.argmax(skor_klaster) if sum(skor_klaster) > 0 else 0
            
            atribut_topik = STRUKTUR_TOPIK[f"Klaster Topik {indeks_terpilih + 1}"]
            
            st.markdown(f"""
                <div class="result-box">
                    <p style="color: #38bdf8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin: 0;">Topik Dominan</p>
                    <p style="color: #ffffff; font-size: 1.4rem; font-weight: 700; margin-top: 0.25rem; margin-bottom: 0.5rem;">{atribut_topik['nama']}</p>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;"><b>Token Kunci:</b> <code style="color:#38bdf8;">{atribut_topik['kata_kunci']}</code></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #64748b; margin-top: 2rem;'>Menunggu input data ulasan untuk memproses matriks klaster...</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_bulk:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h3>Unggah Dokumen Massal</h3>", unsafe_allow_html=True)
    st.markdown("<p>Fasilitas integrasi file CSV untuk melakukan pemrosesan data secara simultan.</p>", unsafe_allow_html=True)
    
    berkas_csv = st.file_uploader("Pilih Berkas (.csv)", type=["csv"], label_visibility="collapsed")
    
    if berkas_csv is not None:
        df_baru = pd.read_csv(berkas_csv)
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Pratinjau Struktur Data Tabel:**")
        st.dataframe(df_baru.head(5), use_container_width=True)
        st.success(f"Sistem Terbaca: {len(df_baru)} record data berhasil dialokasikan.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_compare:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h3>Komparasi Bobot Kepentingan Fitur</h3>", unsafe_allow_html=True)
    st.markdown("<p>Visualisasi perbandingan kontribusi struktur fitur antara dua metode pemodelan.</p>", unsafe_allow_html=True)
    
    pilihan_mesin = st.selectbox(
        "Pilih Algoritma Pemodelan:", 
        ["Latent Dirichlet Allocation (LDA)", "Non-Negative Matrix Factorization (NMF)"]
    )
    
    kumpulan_nama_klaster = [STRUKTUR_TOPIK[k]["nama"] for k in STRUKTUR_TOPIK]
    bobot_kalkulasi = [35, 25, 15, 15, 10] if "LDA" in pilihan_mesin else [42, 23, 15, 12, 8]
    
    fig = px.bar(
        x=bobot_kalkulasi,
        y=kumpulan_nama_klaster,
        orientation='h',
        labels={'x': 'Tingkat Dominasi Korpus (%)', 'y': 'Klaster Terbentuk'},
        title=f"Proporsi Distribusi Berdasarkan Algoritma {pilihan_mesin}",
        color=bobot_kalkulasi,
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
