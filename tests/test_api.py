# Import Libraries
import os
import sys
import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import Custom Modules
from app.main import app

def _mock_load_model_weights(model_role):
    """Return weights from model_info.json without hitting the database."""
    info_path = os.path.join(BASE_DIR, "models", model_role, "model_info.json")
    with open(info_path) as f:
        info = json.load(f)
    return info["weights"], info["run_id"]

@pytest.fixture(scope = "session")
def client():
    with (
        patch('app.main.init_pool'),
        patch('app.main.init_db'),
        patch('app.main.sync_model_registry'),
        patch('app.main.log_prediction'),
        patch('app.main.log_raw_input'),
        patch('app.main.API_KEY', 'test-api-key'),
        patch('app.load_model.load_model_weights', side_effect=_mock_load_model_weights)
    ):
        with patch.dict('os.environ', {'API_KEY': 'test-api-key'}):
            with TestClient(app) as c:
                c.headers.update({"X-API-Key": "test-api-key"})
                yield c

def valid_payload():
    return {
        "CustomerID": "C123",
        "Gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0
    }

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Ok"}

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200

def test_valid_payload_should_pass(client):
    response = client.post("/predict", json = valid_payload())
    assert response.status_code == 200

def test_response_structure_should_be_correct(client):
    response = client.post("/predict", json = valid_payload())
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "model_predictions" in data
    assert "lgb" in data["model_predictions"]
    assert "xgb" in data["model_predictions"]
    assert "cat" in data["model_predictions"]

def test_extra_field_should_fail(client):
    payload = valid_payload()
    payload['Extra'] = 0
    response = client.post("/predict", json = payload)
    assert response.status_code == 422