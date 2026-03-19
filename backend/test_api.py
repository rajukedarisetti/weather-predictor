import requests
import json

# Test all 3 endpoints
print("=" * 50)
print("TESTING BACKEND API ENDPOINTS")
print("=" * 50)

# 1. Test /api/locations
print("\n1. GET /api/locations")
resp = requests.get("http://localhost:8001/api/locations")
print(f"   Status: {resp.status_code}")
data = resp.json()
print(f"   States count: {len(data)}")
print(f"   First 3: {list(data.keys())[:3]}")

# 2. Test /api/model_info
print("\n2. GET /api/model_info")
resp = requests.get("http://localhost:8001/api/model_info")
print(f"   Status: {resp.status_code}")
info = resp.json()
print(f"   Accuracy: {info['accuracy']}%")
print(f"   RMSE: {info['rmse']}")

# 3. Test /api/predict
print("\n3. POST /api/predict")
payload = {"state": "Telangana", "district": "Hyderabad", "date": "2026-12-25"}
resp = requests.post("http://localhost:8001/api/predict", json=payload)
print(f"   Status: {resp.status_code}")
if resp.status_code == 200:
    pred = resp.json()
    s = pred["summary"]
    print(f"   Condition: {s['condition']}")
    print(f"   Temperature: {s['temperature']}°C")
    print(f"   Humidity: {s['humidity']}%")
    print(f"   Wind Speed: {s['wind_speed']} km/h")
    print(f"   AQI: {s['aqi']}")
    print(f"   Hourly forecasts: {len(pred['hourly'])} time slots")
else:
    print(f"   ERROR: {resp.text}")

print("\n" + "=" * 50)
print("ALL TESTS COMPLETE")
print("=" * 50)
