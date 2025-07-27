# Dependencias
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Descargamos los recursos necesarios de NLTK (solo la primera vez)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = word_tokenize(text, language='english', preserve_line=True)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def vectorize_text(text_series, max_features=10000, ngram_range=(1,3)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer

def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le
