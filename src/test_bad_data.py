"""Test bad input data"""
import requests

BASE_URL = "http://localhost:9000"

print("Testing API with bad input data...\n")

# Test 1: Negative amount
response = requests.post(f"{BASE_URL}/predict", json={
    "amount": -96,
    "hour": 14,
    "day_of_week": 6,
    "merchant_category": "travel"
})
print(f"Status: {response.status_code}")
print(f"Response body: {response.json()}")