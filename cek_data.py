import pandas as pd

# Membaca dataset
df = pd.read_csv('dataset.csv')

print("=== INFORMASI DATASET ===")
print(f"Total baris data: {len(df)}")
print(f"Nama kolom: {list(df.columns)}\n")

print("=== 3 BARIS PERTAMA DATA ===")
print(df[['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']].head(3))
