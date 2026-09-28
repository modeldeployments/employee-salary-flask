import requests


url = 'http://localhost:5000/predict_api'

data = {
    'experience': 2,
    'test_score': 9,
    'interview_score': 6
}

response = requests.post(
    url,
    json=data,
    timeout=10
)

response.raise_for_status()

print(response.json())