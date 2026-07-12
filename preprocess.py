import pandas as pd
import re
from sklearn.model_selection import train_test_split

print("Memulai proses preprocessing data...")

df = pd.read_csv('dataset.csv')
df['label'] = df['class'].map({0: 0, 1: 0, 2: 1})

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'rt\s+@\w+:', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)
df = df.dropna(subset=['clean_tweet', 'label'])

df_clean = df[['clean_tweet', 'label']]
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)

train_df.to_csv('train_clean.csv', index=False)
test_df.to_csv('test_clean.csv', index=False)

print("Preprocessing selesai!")
print(f"Jumlah Data Training: {len(train_df)} baris")
print(f"Jumlah Data Testing: {len(test_df)} baris")
