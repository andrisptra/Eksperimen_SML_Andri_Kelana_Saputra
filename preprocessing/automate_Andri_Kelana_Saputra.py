"""
==================
File otomatisasi preprocessing.
Konversi dari notebook menjadi pipeline preprocessing otomatis.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ++++++++++++++++++++++++++++++++++++++
# Download NLTK resources if not already downloaded
# ++++++++++++++++++++++++++++++++++++++

def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for res in resources:
        try: 
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except LookupError:
            nltk.download(res, quiet=True)


# ++++++++++++++++++++++++++++++++++++++
# Step 1: Load Dataset
# ++++++++++++++++++++++++++++++++++++++

def load_data(filepath:str) -> pd.DataFrame:
    """Load raw CSV dataset"""
    print(f"[1/6]: Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with shape: {df.shape}")
    return df


# ++++++++++++++++++++++++++++++++++++++
# Step 2: Handle Missing Values
# ++++++++++++++++++++++++++++++++++++++

def handle_missing_values(df:pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values"""
    print(f"[2/6]: Handling missing values...")

    # Fill catergorical columns with Unknown
    cat_cols_with_na = [
        'economically_disadvantaged',
        'student_disability_status',
        'ell_status',
        'race_ethnicity'
    ]

    for col in cat_cols_with_na:
        if col in df.columns:
            before = df[col].isnull().sum()
            df[col] = df[col].fillna('Unknown')
            print(f"      Filled {before} nulls in '{col}' with 'Unknown'")

    # Drop source_text columns (high missing, not needed for modeling)
    source_cols = [c for c in df.columns if 'source_text' in c]
    if source_cols:
        df.drop(columns=source_cols, inplace=True)
        print(f"      Dropped columns: {source_cols}")

    print(f"      Remaining nulls: {df.isnull().sum().sum()}")
    return df


# ++++++++++++++++++++++++++++++++++++++
# Step 3: Text Cleaning
# ++++++++++++++++++++++++++++++++++++++

def clean_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """Clean and preprocess a single essay text."""
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to full_text column."""
    print("[3/6] Cleaning text (lemmatize + stopword removal)...")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    df['processed_text'] = df['full_text'].apply(
        lambda x: clean_text(x, lemmatizer, stop_words)
    )
    print(f"      Text cleaning done. Sample: {df['processed_text'].iloc[0][:80]}...")
    return df

# ─────────────────────────────────────────────
# Step 4: Feature Engineering
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create numerical features from text and categorical encoding."""
    print("[4/6] Feature engineering...")

    # Numerical features from raw text
    df['word_count'] = df['full_text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['full_text'].apply(lambda x: len(str(x)))
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
    df['sentence_count'] = df['full_text'].apply(
        lambda x: len(re.split(r'[.!?]+', str(x)))
    )
    df['unique_word_ratio'] = df['full_text'].apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1)
    )

    # Label encode categorical features
    cat_features = [
        'gender', 'race_ethnicity', 'economically_disadvantaged',
        'student_disability_status', 'ell_status', 'prompt_name'
    ]
    le = LabelEncoder()
    for col in cat_features:
        if col in df.columns:
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

    # Create target label
    df['score_label'] = df['score'].astype(str)

    print("      Numerical features: word_count, char_count, avg_word_length, "
          "sentence_count, unique_word_ratio")
    print("      Encoded categorical features.")
    return df


# ─────────────────────────────────────────────
# Step 5: Select Final Features
# ─────────────────────────────────────────────
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns needed for modeling."""
    print("[5/6] Selecting final feature columns...")

    final_cols = [
        'essay_id', 'score', 'score_label',
        'processed_text',
        'word_count', 'char_count', 'avg_word_length',
        'sentence_count', 'unique_word_ratio',
        'gender_encoded', 'race_ethnicity_encoded',
        'economically_disadvantaged_encoded',
        'student_disability_status_encoded',
        'ell_status_encoded', 'prompt_name_encoded'
    ]

    # Only keep columns that exist
    available_cols = [c for c in final_cols if c in df.columns]
    df_final = df[available_cols].copy()
    print(f"      Final shape: {df_final.shape}")
    return df_final


# ─────────────────────────────────────────────
# Step 6: Train/Test Split & Save
# ─────────────────────────────────────────────
def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    """Split into train/test and save all output CSVs."""
    print(f"[6/6] Splitting and saving to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=['essay_id', 'score', 'score_label'], errors='ignore')
    y = df['score_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Full preprocessed
    df.to_csv(os.path.join(output_dir, 'asap2_preprocessed.csv'), index=False)

    # Train split
    train_df = X_train.copy()
    train_df['score_label'] = y_train
    train_df.to_csv(os.path.join(output_dir, 'train_preprocessed.csv'), index=False)

    # Test split
    test_df = X_test.copy()
    test_df['score_label'] = y_test
    test_df.to_csv(os.path.join(output_dir, 'test_preprocessed.csv'), index=False)

    print(f"\n✅ Preprocessing complete!")
    print(f"   Total samples    : {len(df)}")
    print(f"   Train samples    : {len(train_df)}")
    print(f"   Test samples     : {len(test_df)}")
    print(f"   Output directory : {output_dir}")
    print(f"   Files saved      :")
    print(f"     - asap2_preprocessed.csv")
    print(f"     - train_preprocessed.csv")
    print(f"     - test_preprocessed.csv")


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def run_preprocessing(input_path: str, output_dir: str) -> pd.DataFrame:
    """
    Run full preprocessing pipeline.

    Args:
        input_path: Path to raw CSV file
        output_dir: Directory to save preprocessed outputs

    Returns:
        Preprocessed DataFrame
    """
    print("=" * 55)
    print("  ASAP2 Essay Score - Preprocessing Pipeline")
    print("=" * 55)

    download_nltk_resources()

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = preprocess_text(df)
    df = feature_engineering(df)
    df = select_features(df)
    split_and_save(df, output_dir)

    print("=" * 55)
    return df


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Automate preprocessing for ASAP2 Essay Score dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='ASAP2_train_sourcetexts.csv',
        help='Path to raw dataset CSV (default: ../asap2_raw/train.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='preprocessing/asap2_preprocessing',
        help='Output folder for preprocessed files (default: asap2_preprocessing)'
    )
    args = parser.parse_args()

    run_preprocessing(input_path=args.input, output_dir=args.output)