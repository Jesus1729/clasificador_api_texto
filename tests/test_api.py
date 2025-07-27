# Dependencias
from fastapi.testclient import TestClient
import sys
import pathlib

# Agregamos ruta al directorio raíz del proyecto
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from src.predict_api import app

client = TestClient(app)

def test_predict_success():
    response = client.post(
        "/predict",
        json={"titular": "14 Movie Clips That Will Make You Love Your Mom Even More This Mother's Day!"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "categoria_predicha" in json_data
    assert isinstance(json_data["categoria_predicha"], str)

def test_predict_missing_text_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422  

def test_predict_empty_text():
    response = client.post("/predict", json={"titular": ""})
    assert response.status_code == 400  
    json_data = response.json()
    assert json_data["detail"] == "No se proporcionó un texto."


