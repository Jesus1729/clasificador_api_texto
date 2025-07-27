# Clasificador de Titulares de Noticias

**Descripción:** Este proyecto consiste en un modelo de clasificación de titulares de noticias en diferentes categorías temáticas. El objetivo es proporcionar una API que, dada un titular, prediga la categoría a la que pertenece, utilizando un modelo de regresión logística entrenado con datos preprocesados y balanceados.

## Estructura del proyecto

```bash
clasificador_texto/
│
├── data/                   # Archivos de datos (data_clean.csv, data.json)
├── model/                  # Modelo entrenado (modelo.joblib)
├── src/                    # Código fuente
│   ├── predict_api.py      # API FastAPI para predicción
│   ├── preprocesamiento.py # Funciones de limpieza y preprocesamiento de texto
│   └── train.py            # Código para entrenamiento del modelo
│
├── requirements.txt        # Dependencias del proyecto
└── app.yaml                # Configuración para despliegue en App Engine
```

## Instalación

Clonar el repositorio

```bash
git clone https://github.com/Jesus1729/clasificador_api_texto.git
cd clasificador_api_texto/clasificador_texto
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Uso local de la API

Para ejecutar la API localmente:

```bash
uvicorn src.predict_api:app --reload
```

Luego, acceder a http://127.0.0.1:8000/docs para ver la documentación automática y probar los endpoint /predict y /metrics.

## Pruebas unitarias

Para asegurar el correcto funcionamiento de la API y del procesamiento del texto, se implementaron pruebas unitarias utilizando fastapi.testclient y pruebas directas a las funciones de preprocesamiento.

Las pruebas sobre la API validan:

- Que la predicción retorne correctamente la categoría dada una entrada válida.

- Que se manejen correctamente casos con campos vacíos o faltantes, retornando los códigos de error adecuados.

Las pruebas de preprocesamiento verifican que:

- Se elimine la puntuación y se convierta el texto a minúsculas.

- Se eliminen las palabras vacías (stopwords).

- Se manejen correctamente entradas vacías.

Para ejecutar las pruebas, se pueden usar los siguientes comandos:

```bash
pytest tests/test_api.py
pytest tests/test_preprocesamiento.py
```

Esto garantiza que las funciones críticas y los endpoints funcionen como se espera antes de realizar despliegues o cambios mayores.

## Despliegue en Google Cloud App Engine

La API ha sido desplegada en App Engine de Google Cloud.
Pasos realizados:

- Configuración del proyecto con gcloud init.
- Preparación del archivo app.yaml con la configuración de entorno.
- Despliegue con gcloud app deploy.

Para el despliegue, solo se requiere la carpeta model, la carpeta src, además del archivo requirements.txt y app.yaml. Esto es suficiente para ejecutar la API sin necesidad de incluir notebooks o archivos del análisis.

Puedes acceder a la API desplegada en:
https://glowing-net-447805-q9.uc.r.appspot.com/docs

## Ejemplo de uso con curl

Puedes probar la API directamente desde la terminal con el siguiente comando:

```bash
curl -X 'POST' \
  'https://glowing-net-447805-q9.uc.r.appspot.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "titular": "2 Democratic Senators Say Neil Gorsuch Refused To Meet With Them"
}'
```

## Notas sobre el análisis y modelo

- Se realizó un análisis exploratorio de datos (EDA) para comprender la distribución de clases y las características generales de los titulares de noticias.

- Se eliminaron titulares excesivamente cortos y muy largos para mejorar la calidad del entrenamiento y reducir ruido.

- El dataset presentaba un desbalance significativo entre las categorías, por lo que se aplicó oversampling para balancearlo y evitar que el modelo favoreciera la clase mayoritaria.

- Se exploraron dos modelos de clasificación: Naive Bayes y Regresión Logística, siendo esta última la elegida por ofrecer mejores métricas de desempeño. Para encontrar la mejor combinación de hiperparámetros se utilizó Grid Search. Las métricas obtenidas fueron las siguientes:

    - Regresión Logística:
        Mejores parámetros: {'C': 1.0, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'liblinear'}
        Accuracy: 0.7276
        Precision: 0.7233
        Recall: 0.7276
        F1-score: 0.7232

    - Multinomial Naive Bayes:
        Mejores parámetros: {'alpha': 0.01, 'fit_prior': False}
        Accuracy: 0.4586
        Precision: 0.5167
        Recall: 0.4586
        F1-score: 0.4789

- En el procesamiento del texto se aplicaron los siguientes pasos:

    - Eliminación de stopwords para reducir palabras poco informativas.

    - Uso de LabelEncoder para convertir las etiquetas de texto en valores numéricos.

    - Se probaron tanto TfidfVectorizer como CountVectorizer para transformar los titulares en representaciones numéricas, siendo TfidfVectorizer el más efectivo en este caso.

    - También se utilizaron técnicas como ngram_range y max_features para capturar más contexto sin sobreajustar el modelo.

## Monitorización

Se implementó un sistema de monitorización utilizando Prometheus para registrar el comportamiento del endpoint `/predict`.  
Se expone un endpoint adicional `/metrics`, el cual devuelve únicamente las métricas relevantes, filtrando las innecesarias.

Las métricas disponibles son:

- `predict_request_count_total`: total de solicitudes al endpoint `/predict`.
- `predict_request_latency_seconds_sum`: suma total del tiempo de respuesta.
- `predict_request_latency_seconds_count`: cantidad de mediciones de latencia.
- `predict_request_latency_seconds_bucket`: distribución de latencias en buckets.
- `predict_request_latency_seconds_created`: timestamp de creación del histograma.