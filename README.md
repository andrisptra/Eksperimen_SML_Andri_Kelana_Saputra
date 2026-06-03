# Eksperimen_SML_Student

Repository eksperimen Machine Learning untuk dataset **ASAP2 Essay Score Classification**.

## 📁 Struktur Repository

```
Eksperimen_SML_Student
├── .github/
│   └── workflows/
│       └── main.yml                             # GitHub Actions workflow (Advance)
├── preprocessing/
│   ├── Eksperimen_Andri_Kelana_Saputra.ipynb    # Notebook eksperimen (Basic)
│   ├── automate_Andri_Kelana_Saputra.py         # Script otomatisasi preprocessing (Skilled/Advance)
│   └── asap2_preprocessing/                     # Output preprocessed data
│       ├── asap2_preprocessed.csv
│       ├── train_preprocessed.csv
│       └── test_preprocessed.csv
├──ASAP2_train_sourcetexts.csv                   # Dataset mentah (dari Kaggle)
└── README.md
```

## 📊 Dataset

- **Nama**: ASAP2 Assessment Essay Scoring
- **Task**: Multi-class Classification (Score 1–4)
- **Jumlah data**: 24,728 baris
- **Fitur utama**: `full_text` (essay), fitur demografis (gender, race, ell_status, dll.)
- **Target**: `score` (nilai essay 1–4)

## 🔄 Preprocessing Steps

1. **Data Loading** — Load CSV dari `asap2_raw/`
2. **Handle Missing Values** — Fill dengan 'Unknown', drop `source_text_*`
3. **Text Cleaning** — Lowercase, hapus karakter khusus, tokenize, lemmatize, hapus stopwords
4. **Feature Engineering** — Tambah fitur numerik: `word_count`, `char_count`, `avg_word_length`, `sentence_count`, `unique_word_ratio`
5. **Label Encoding** — Encode fitur kategorik (gender, race, dll.)
6. **Train/Test Split** — 80:20 stratified split

## 🚀 Cara Menjalankan

### Manual (via notebook)
Buka dan jalankan `preprocessing/Eksperimen_Andri_Kelana_Saputra.ipynb`

### Otomatis (via script)
```bash
# Default (asap2_raw/train.csv -> preprocessing/asap2_preprocessing/)
python preprocessing/automate_Andri_Kelana_Saputra.py

# Custom path
python preprocessing/automate_Andri_Kelana_Saputra.py \
  --input path/to/train.csv \
  --output path/to/output_folder
```

### Install dependencies
```bash
pip install pandas numpy scikit-learn nltk
```

## ⚙️ GitHub Actions

Workflow otomatis berjalan ketika:
- Push ke branch `main` yang mengubah file di `asap2_raw/` atau `preprocessing/automate_Andri_Kelana_Saputra.py`
- Manual trigger via tab **Actions** > **Run workflow**

Hasil preprocessing akan otomatis di-commit ke `preprocessing/asap2_preprocessing/`.
