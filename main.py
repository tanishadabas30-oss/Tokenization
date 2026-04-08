import pandas as pd
import numpy as np 
import zipfile
import re 
import nltk 
import string 

REQUIRED = [
    "stopwords",
    "punkt",
    "punkt_tab",
]
print("checking/ downloading NLTK data...\n")
for i in REQUIRED:
    nltk.download(i, quiet= True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation 

with zipfile.ZipFile('DataSets/IMDB Dataset.csv (3).zip', 'r') as z:
    z.extractall('DataSets/')

df = pd.read_csv('DataSets/IMDB Dataset.csv')
print(df.head())  

# Convert text to lowercase 
df['review']= df['review'].str.lower()
print("The data in lower case\n",df) 

#Removing punctuation 
exclude = string.punctuation
def remove_punc1(text):
    return text.translate(str.maketrans('','',exclude))
df['review']= df['review'].apply(remove_punc1)
print("The data without any punctuation\n",df['review'])

#Removing StopWords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    new_text = []
    for i in text.split():
        if i in stop_words:
            new_text.append('')
        else:
            new_text.append(i)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)
df['cleaned_review'] = df['review'].apply(remove_stopwords)
print("Stopwords Removed!")

# Tokenization
df['tokenized_review'] = df['cleaned_review'].apply(word_tokenize)
print("Tokenization Done!")

print(df[['review', 'cleaned_review', 'tokenized_review']].head())