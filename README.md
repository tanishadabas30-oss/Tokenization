# 🎬 IMDB Sentiment Analysis — Tokenization

> A complete NLP preprocessing pipeline built on the IMDB 50K Movie Reviews dataset — covering text cleaning, stopword removal, and tokenization as the foundation for sentiment classification.

---

## 📌 Approach

The goal of this project is to preprocess raw movie reviews from the IMDB dataset so they can be used to train a sentiment analysis model. The pipeline follows these stages:

1. **Load** the raw CSV dataset
2. **Normalize** text to lowercase
3. **Clean** by removing punctuation
4. **Filter** by removing stopwords (common words like "the", "is", "and")
5. **Tokenize** each review into individual words
6. **Output** a structured, model-ready DataFrame

This approach ensures the text data is stripped of noise before being fed into any machine learning model.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.12** | Core programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations and batch processing |
| **NLTK** | Stopword removal and tokenization |
| **Zipfile** | Extracting compressed dataset |
| **String / Re** | Punctuation removal and regex operations |

---

## 📁 Project Structure

```
Tokenization/
│
├── dataset/
│   └── IMDB Dataset.csv
│
├── notebook.ipynb
├── main.py
├── config.txt
├── output.txt
└── README.md
```

| File | Description |
|---|---|
| `dataset/IMDB Dataset.csv` | Raw dataset with 50,000 movie reviews |
| `main.py` | Main preprocessing pipeline script |
| `notebook.ipynb` | Step-by-step Jupyter notebook version |
| `config.txt` | Configuration settings |
| `output.txt` | Sample output logs |
| `README.md` | Project documentation |

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
https://github.com/tanishadabas30-oss/Tokenization/edit/main/README.md
```

### 2. Install Dependencies
```bash
pip install pandas numpy nltk
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 4. Add the Dataset
- Place `IMDB Dataset.csv` inside the `DataSets/` folder
- Or place the `.zip` file and let the script extract it automatically

### 5. Run the Script
```bash
python main.py
```

---

## 💻 Sample Code

```python
import pandas as pd
import nltk
import string
import zipfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
REQUIRED = ["stopwords", "punkt", "punkt_tab"]
for i in REQUIRED:
    nltk.download(i, quiet=True)

# Extract and Load Dataset
with zipfile.ZipFile('DataSets/IMDB Dataset.csv (3).zip', 'r') as z:
    z.extractall('DataSets/')

df = pd.read_csv('DataSets/IMDB Dataset.csv')

# Step 1 - Lowercase
df['review'] = df['review'].str.lower()

# Step 2 - Remove Punctuation
exclude = string.punctuation
def remove_punc(text):
    return text.translate(str.maketrans('', '', exclude))

df['review'] = df['review'].apply(remove_punc)

# Step 3 - Remove Stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([w for w in text.split() if w not in stop_words])

df['cleaned_review'] = df['review'].apply(remove_stopwords)

# Step 4 - Tokenization
df['tokenized_review'] = df['cleaned_review'].apply(word_tokenize)

print(df[['review', 'cleaned_review', 'tokenized_review']].head())
```

---

## 📊 Output Analysis

### Sample Output

| review | cleaned_review | tokenized_review |
|---|---|---|
| "this movie was absolutely amazing..." | "movie absolutely amazing..." | ['movie', 'absolutely', 'amazing'] |
| "worst film i have ever seen..." | "worst film ever seen..." | ['worst', 'film', 'ever', 'seen'] |

### What Each Stage Does

```
ORIGINAL   → "This movie was absolutely amazing and I loved it"
LOWERCASE  → "this movie was absolutely amazing and i loved it"
NO PUNCT   → "this movie was absolutely amazing and i loved it"
NO STOPS   → "movie absolutely amazing loved"
TOKENIZED  → ['movie', 'absolutely', 'amazing', 'loved']
```

### Dataset Overview
- **Total Reviews:** 50,000
- **Classes:** Positive / Negative (balanced — 25,000 each)
- **Avg Review Length:** ~230 words
- **After Cleaning:** ~60–80% of words retained (stopwords are ~20–40% of raw text)

### Key Observations
- Removing stopwords significantly reduces noise without losing sentiment-relevant words
- Lowercasing ensures words like "Movie" and "movie" are treated as the same token
- Punctuation removal prevents tokens like `"amazing!"` and `"amazing"` from being counted separately
- Tokenization converts cleaned text into a list format ready for vectorization (TF-IDF, Word2Vec, etc.)

---

## 🔮 Next Steps

- [ ] Stemming / Lemmatization
- [ ] TF-IDF Vectorization
- [ ] Model Training (Logistic Regression / Naive Bayes)
- [ ] Model Evaluation (Accuracy, F1 Score)
- [ ] Deploy as a Web App

---

## 🙋 Author

**Tanisha Dabas**
- GitHub: (https://github.com/tanishadabas30-oss)

---
