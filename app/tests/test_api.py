from starlette.testclient import TestClient
from app.api import app


def test_doc_redirect():
    """Redirect docs from `https://domain.com/docs`
    to `https://domain.com/<prefix>/docs`
    """
    client = TestClient(app)
    response = client.get('/')

    assert response.history[0].status_code == 302
    assert response.status_code == 200
    assert response.url == 'http://testserver/docs'


def test_api():
    """Test out a simple API request for prediction."""
    client = TestClient(app)

    request_data = {
        "values": [{
            "record_id": "1",
            "model_name": "Support Vector Machine",
            "data": {
                "age": 63,
                "sex": 0,
                "cp": 3,
                "trestbps": 155,
                "chol": 190,
                "fbs": 0,
                "restecg": 0,
                "thalach": 123,
                "exang": 1,
                "oldpeak": 1.7,
                "slope": 1,
                "ca": 2,
                "thal": 0,
            }
        }]
    }

    response = client.post('/predict', json=request_data)
    assert response.status_code == 200

    first_record = response.json()['values'][0]
    assert first_record['record_id'] == '1'
    assert first_record['errors'] is None
    assert first_record['warnings'] is None

    assert first_record['data']['model_name'] == 'Support Vector Machine'
    assert first_record['data']['confidence_score'] == 90.2
    assert first_record['data']['has_heart_disease'] is True
