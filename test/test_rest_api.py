import pytest

from fastapi.testclient import TestClient
from src.app import app, DEFAULT_HELLO_MESSAGE

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == DEFAULT_HELLO_MESSAGE


def test_service_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_service_handle_audio_file_less_1_second():
    request = {'data': [[1 for _ in range(7999)]],
               'sample_rate': 8000}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Audio file length should be at least 1s"


def test_service_handle_audio_file_more_60_second():
    request = {'data': [[1 for _ in range(60 * 8000 + 1)]],
               'sample_rate': 8000}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 400
        assert response.json()['detail'] == "Audio file is too big"


def test_service_can_predict():
    request = {'data': [[1 for _ in range(10 * 24000)]],
               'sample_rate': 24000}

    with TestClient(app) as client:
        response = client.get("/predict", json=request)
        assert response.status_code == 200
