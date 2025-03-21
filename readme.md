`import requests
import json

BASE_URL = "http://localhost:8094"

# Test Việt -> Anh
vi_texts = [
    "Xin chào, bạn khỏe không?",
    "Hôm nay là một ngày đẹp trời.",
    "Tôi thích nghe nhạc và ăn uống."
]

response = requests.post(
    f"{BASE_URL}/vi2en",
    json={"texts": vi_texts}
)

if response.status_code == 200:
    translations = response.json()
    print("Vietnamese to English:")
    for orig, trans in zip(vi_texts, translations):
        print(f"{orig} -> {trans}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# Test Anh -> Việt
en_texts = [
    "Hello, how are you?",
    "It's a beautiful day.",
    "I like to listen to music and eat."
]

response = requests.post(
    f"{BASE_URL}/en2vi",
    json={"texts": en_texts}
)

if response.status_code == 200:
    translations = response.json()
    print("\nEnglish to Vietnamese:")
    for orig, trans in zip(en_texts, translations):
        print(f"{orig} -> {trans}")
else:
    print(f"Error: {response.status_code} - {response.text}")

# Test health check
response = requests.get(f"{BASE_URL}/health")
print(f"\nHealth check: {response.json()}")``
