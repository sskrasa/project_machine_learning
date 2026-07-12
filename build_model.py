import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

print("[1/4] Mengunduh dependensi NLP profesional...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("[2/4] Memuat dataset raksasa (26K+ record)...")
df = pd.read_csv('dataset.csv')
documents = df['tweet'].astype(str).dropna().tolist()

print("[3/4] Melakukan text preprocessing standar industri...")
stop_words = set(stopwords.words('english'))

def clean_text_professional(text):
    text = text.lower()
    text = re.sub(r'rt\s+@\w+:', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = nltk.word_tokenize(text)
    cleaned_words = [w for w in words if w not in stop_words and len(w) > 3]
    return " ".join(cleaned_words)

processed_corpus = [clean_text_professional(doc) for doc in documents]

print("[4/4] Ekstraksi Fitur & Pelatihan Ganda (LDA vs NMF)...")
tf_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

tf_matrix = tf_vectorizer.fit_transform(processed_corpus)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

# Model 1: LDA via Scikit-Learn
print("-> Melatih Model 1: Latent Dirichlet Allocation (LDA)...")
lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=5)
lda.fit(tf_matrix)

# Model 2: NMF via Scikit-Learn
print("-> Melatih Model 2: Non-Negative Matrix Factorization (NMF)...")
nmf = NMF(n_components=5, random_state=42, max_iter=100)
nmf.fit(tfidf_matrix)

print("-> Mengevaluasi performa model...")
perplexity = lda.perplexity(tf_matrix)
print(f"-> Evaluasi LDA Perplexity: {perplexity:.2f}")

print("Menyimpan artefak model ke dalam disk...")
joblib.dump(lda, 'model_lda.pkl')
joblib.dump(tf_vectorizer, 'vectorizer_lda.pkl')
joblib.dump(nmf, 'model_nmf.pkl')
joblib.dump(tfidf_vectorizer, 'vectorizer_nmf.pkl')
joblib.dump(processed_corpus, 'processed_corpus.pkl')

print("Sistem kecerdasan buatan siap. Seluruh artefak berhasil diekspor.")
