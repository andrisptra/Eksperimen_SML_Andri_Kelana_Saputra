import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

dataset_path = "ASAP2_train_sourcetexts.csv"
df = pd.read_csv(dataset_path)


# Tahapan EDA
score_counts = df['score'].value_counts()
score_value = df['score'].value_counts().index

# Visualize the distribution of scores
plt.figure(figsize=(10, 6))
sns.barplot(x=score_value, y=score_counts, palette='viridis')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()

# Tahapan Preprocessing
def cleaning_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.strip()
    return text

def case_folding(text):
    return text.lower()

def tokenization(text):
    return text.split()

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def to_sentences(tokens):
    return ' '.join(tokens)


def preprocess_text(text):
    cleaned_text = cleaning_text(text)
    folded_text = case_folding(cleaned_text)
    tokens = tokenization(folded_text)
    filtered_tokens = remove_stopwords(tokens)
    final_text = to_sentences(filtered_tokens)
    return final_text

df['processed_text'] = df['full_text'].apply(preprocess_text)


# pelabelan data
def label_score(score):
    if score <= 2:
        return 'Low'
    elif score <= 4:
        return 'Medium'
    else:
        return 'High'
    

df['score_label'] = df['score'].apply(label_score)
df.head()

plt.figure(figsize=(10, 6))
sns.countplot(x='score_label', data=df, palette='viridis')
plt.title('Distribution of Score Labels')
plt.xlabel('Score Label')
plt.ylabel('Count')
plt.show()

# menyimpan dataset yang sudah diproses
processed_dataset_path = "ASAP2_train_sourcetexts_processed.csv"
df.to_csv(processed_dataset_path, index=False)

if __name__ == "__main__":
    print("Preprocessing completed. Processed dataset saved to:", processed_dataset_path)   