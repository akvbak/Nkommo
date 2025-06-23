import requests

# Rasa REST webhook URL
RASA_WEBHOOK_URL = "http://localhost:5055/webhook"


def parse_intent_with_rasa(sender_id: str, message: str, language: str):
    """
    Sends `message` to Rasa and returns (intent, response_text, confidence).
    """
    payload = {
        "sender": sender_id,
        "message": message,
        "metadata": {"language": language}
    }
    try:
        resp = requests.post(
            RASA_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Rasa request failed: {e}")
        return None, None, 0.0

    data = resp.json()
    if not data:
        return None, None, 0.0

    first = data[0].get("custom", {})
    intent     = first.get("intent")
    confidence = first.get("confidence", 0.0)
    rasa_resp  = first.get("response")
    return intent, rasa_resp, confidence