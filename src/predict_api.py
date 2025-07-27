# Dependencias
import os
import sys
import pathlib
import time
import logging
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Agregamos ruta al directorio raíz del proyecto
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.preprocesamiento import clean_text

# Configuración para el logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Cargamos el modelo
ruta_base = os.path.dirname(os.path.abspath(__file__))  
ruta_modelo = os.path.join(ruta_base, "..", "model", "modelo.joblib")
ruta_modelo = os.path.abspath(ruta_modelo)
modelo_data = joblib.load(ruta_modelo)
model = modelo_data['model']
vectorizer = modelo_data['vectorizer']
label_encoder = modelo_data['label_encoder']

# Creamos la app de FastAPI
app = FastAPI(title="Clasificador de Titulares de Noticias")

# Métricas Prometheus
REQUEST_COUNT = Counter('predict_request_count', 'Número total de predicciones realizadas')
REQUEST_LATENCY = Histogram('predict_request_latency_seconds', 'Latencia para procesar una predicción')

# Clase para entrada del request
class Headline(BaseModel):
    titular: str

@app.post("/predict")
@REQUEST_LATENCY.time()  # Esto mide la duración de la función
def predict(headline: Headline):
    REQUEST_COUNT.inc()

    if not headline.titular:
        logging.warning("Texto vacío recibido en /predict")
        raise HTTPException(status_code=400, detail="No se proporcionó un texto.")

    start_time = time.time()

    cleaned_text = clean_text(headline.titular)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    latency = time.time() - start_time
    logging.info(f"Predicción realizada para texto: '{headline.titular[:30]}...' => {predicted_label} (latencia: {latency:.4f}s)")

    return {"categoria_predicha": predicted_label}

@app.get("/metrics")
def metrics():
    all_metrics = generate_latest().decode('utf-8')
    filtered_lines = []

    allowed_metrics = [
        "predict_request_count_total",
        "predict_request_latency_seconds_sum",
        "predict_request_latency_seconds_created",
        "predict_request_latency_seconds_bucket",
        "predict_request_latency_seconds_count",
    ]

    show_line = False
    for line in all_metrics.splitlines():
        if line.startswith("# HELP") or line.startswith("# TYPE"):
            if any(metric in line for metric in allowed_metrics):
                filtered_lines.append(line)
                show_line = True
            else:
                show_line = False
        else:
            if any(metric in line for metric in allowed_metrics) and show_line:
                filtered_lines.append(line)

    filtered_metrics = "\n".join(filtered_lines)
    return Response(filtered_metrics, media_type=CONTENT_TYPE_LATEST)
