import requests


def answer(question, passage):
    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "qwen2.5:7b",
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Savolga faqat berilgan parchaga asoslanib, "
                        "o‘zbek tilida qisqa javob ber. Sonlarni aniq saqla. "
                        "Javob parchada bo‘lmasa, «Ma’lumot topilmadi» deb yoz."
                    ),
                },
                {"role": "user", "content": f"Parcha: {passage}\nSavol: {question}"},
            ],
        },
        timeout=(5, 300),
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
