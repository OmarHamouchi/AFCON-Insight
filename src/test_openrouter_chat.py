import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY", "")
model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

if not api_key:
    raise SystemExit("❌ OPENROUTER_API_KEY manquante dans .env")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
    "X-Title": os.getenv("OPENROUTER_APP_NAME", "AFCON AI SBI"),
}

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Dis bonjour en français en une phrase."}
    ],
    "temperature": 0.2
}

r = requests.post(url, headers=headers, json=payload, timeout=45)
print("Status:", r.status_code)
print("Response:", r.text[:900])



