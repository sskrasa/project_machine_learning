import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score, classification_report

print("Membaca data training dan testing...")
train_df = pd.read_csv('train_clean.csv')
test_df = pd.read_csv('test_clean.csv')

train_df = train_df.dropna()
test_df = test_df.dropna()

X_train = train_df['clean_tweet']
y_train = train_df['label']
X_test = test_df['clean_tweet']
y_test = test_df['label']

print("Mengubah teks menjadi matriks fitur (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Melatih model Multinomial Naive Bayes...")
model = naive_bayes.MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Mengevaluasi performa model...")
y_pred = model.predict(X_test_tfidf)
akurasi = accuracy_score(y_test, y_pred)

print("\nHasil Evaluasi:")
print(f"Akurasi Model: {akurasi * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Menyimpan model dan vectorizer...")
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model berhasil disimpan dengan nama sentiment_model.pkl dan tfidf_vectorizer.pkl")
