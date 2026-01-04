import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_explanation(user_question: str, context: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY manquante. Mets-la dans .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Recommandés par OpenRouter
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "AFCON AI SBI"),
    }

    system = (
        "Tu es un assistant d'analyse football (AFCON/CAN). "
        "Tu dois utiliser UNIQUEMENT les preuves fournies. "
        "Si une information manque, dis-le. "
        "Réponds en français, structuré, clair."
    )

    prompt = f"""EVIDENCE (ne pas inventer au-delà) :
{context}

QUESTION :
{user_question}

Réponds en français. Utilise des puces si utile. Ne fabrique pas de faits.
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3
    }

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
    except Exception as e:
        raise RuntimeError(f"Network error calling OpenRouter: {e}")

    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()
