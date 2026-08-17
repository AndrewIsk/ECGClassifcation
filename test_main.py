import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_create_upload_file():
    with open("00999_lr.dat", "rb") as dat, open("00999_lr.hea", "rb") as hea:

        response = client.post("/uploadfile/", files={
            "datFile": ("00999_lr.dat", dat),
            "heaFile": ("00999_lr.hea", hea)
        })
    
        assert response.status_code == 200
        body = response.json()
        assert "Outputs" in body
        assert set(body["Outputs"].keys()) == {"NORM", "MI", "STTC", "CD", "HYP"}

def test_missing_file():
    with open("00999_lr.dat", "rb") as dat:

        response = client.post("/uploadfile/", files={
            "datFile": ("00999_lr.dat", dat),
        })
    
        assert response.status_code == 422


def test_invalid_type():
    with open("00999_lr.txt", "rb") as file:

        response = client.post("/uploadfile/", files={
            "datFile": ("00999_lr.txt", file),
            "heaFile": ("00999_lr.txt", file),
        })
    
        assert response.status_code == (400)

def test_health():
    response = client.get("/health")
    assert response.status_code == (200)