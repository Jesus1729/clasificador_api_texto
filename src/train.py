# Dependencias
import os
import sys
import pathlib
import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

# Agregamos ruta al directorio raíz del proyecto
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.preprocesamiento import clean_text, encode_labels, vectorize_text

# Cargamos y limpiamos los datos
ruta_base = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(ruta_base, "..", "data", "data_clean.csv")
df = pd.read_csv(ruta_csv)
df['cleaned_headline'] = df['headline'].apply(clean_text)

# Preparamos  las etiquetas para el primer modelo
y_encoded, le = encode_labels(df['category'])

# Modelo 1: Logistic Regression
print("Entrenando Logistic Regression...")

# Vectorización TF-IDF
X_lr, vectorizer_lr = vectorize_text(df['cleaned_headline'])


# Oversampling
ros = RandomOverSampler(random_state=2025)
X_resampled_lr, y_resampled_lr = ros.fit_resample(X_lr, y_encoded)

# División entrenamiento/prueba
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_resampled_lr, y_resampled_lr, test_size=0.2, random_state=2025, stratify=y_resampled_lr
)

# GridSearch para hiperparámetros
param_grid_lr = {
    'C': [0.1, 1.0],
    'solver': ['liblinear'],
    'penalty': ['l2'],
    'class_weight': ['balanced', None]
}

grid_lr = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=2025),
    param_grid_lr,
    cv=3,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=0
)

grid_lr.fit(X_train_lr, y_train_lr)
best_model_lr = grid_lr.best_estimator_

# Predicción y métricas
y_pred_lr = best_model_lr.predict(X_test_lr)
f1_lr = f1_score(y_test_lr, y_pred_lr, average='weighted')

print("=== Logistic Regression ===")
print(f"Mejores parámetros: {grid_lr.best_params_}")
print(f"Accuracy: {accuracy_score(y_test_lr, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test_lr, y_pred_lr, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test_lr, y_pred_lr, average='weighted'):.4f}")
print(f"F1-score: {f1_lr:.4f}")
print()

# Modelo 2: Multinomial Naive Bayes
print("Entrenando Multinomial Naive Bayes...")

# Dividimos los datos antes de vectorizar
X_train_text, X_test_text, y_train_nb, y_test_nb = train_test_split(
    df['cleaned_headline'], df['category'], test_size=0.2, stratify=df['category'], random_state=2025
)

# Vectorización CountVectorizer
vectorizer_nb = CountVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_nb = vectorizer_nb.fit_transform(X_train_text)
X_test_nb = vectorizer_nb.transform(X_test_text)

# Oversampling
ros = RandomOverSampler(random_state=2025)
X_train_res_nb, y_train_res_nb = ros.fit_resample(X_train_nb, y_train_nb)

# GridSearch para hiperparámetros
param_grid_nb = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

grid_nb = GridSearchCV(
    MultinomialNB(),
    param_grid_nb,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=0
)

grid_nb.fit(X_train_res_nb, y_train_res_nb)
best_model_nb = grid_nb.best_estimator_

# Predicción y métricas
y_pred_nb = best_model_nb.predict(X_test_nb)
f1_nb = f1_score(y_test_nb, y_pred_nb, average='weighted')

print("=== Multinomial Naive Bayes ===")
print(f"Mejores parámetros: {grid_nb.best_params_}")
print(f"Accuracy: {accuracy_score(y_test_nb, y_pred_nb):.4f}")
print(f"Precision: {precision_score(y_test_nb, y_pred_nb, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test_nb, y_pred_nb, average='weighted'):.4f}")
print(f"F1-score: {f1_nb:.4f}")
print()

# Guardamos el mejor modelo

ruta_modelo = os.path.join(ruta_base, "..", "model", "modelo.joblib")

if f1_lr >= f1_nb:
    objeto_a_guardar = {
        'model': best_model_lr,
        'vectorizer': vectorizer_lr,
        'label_encoder': le
    }
    joblib.dump(objeto_a_guardar, ruta_modelo)
    print("Modelo guardado: Logistic Regression")
else:
    objeto_a_guardar = {
        'model': best_model_nb,
        'vectorizer': vectorizer_nb
    }
    joblib.dump(objeto_a_guardar, ruta_modelo)
    print("Modelo guardado: Multinomial Naive Bayes")
