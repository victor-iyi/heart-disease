from starlette.testclient import TestClient
from app.api import app


def test_doc_redirect():
    client = TestClient(app)
    response = client.get('/')

    assert response.history[0].status_code == 302
    assert response.status_code == 200
    assert response.url == 'http://testserver/docs'


def test_api():
    client = TestClient(app)

    feature_1 = "Feature 1 value"
    feature_2 = "Feature 2 value"
    feature_3 = "Feature 3 value"

    request_data = {
        "values": [{
            "record_id": "1",
            "data": {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "feature_3": feature_3,
            }
        }]
    }

    response = client.post('/predict', json=request_data)
    assert response.status_code == 200
    
    first_record = response.json()['values'][0]
    assert first_record['record_id'] == '1'
    assert first_record['errors'] == None
    assert first_record['warnings'] == None

    assert first_record['data']['model_name'] == 'Support Vector Machine'
    assert first_record['data']['confidence_score'] == 90.2
    assert first_record['data']['has_heart_disease'] == True
