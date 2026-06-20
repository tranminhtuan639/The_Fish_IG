# chatbot/llm_client.py
import os
import io
import base64
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from chatbot.prompt_builder import build_initial_prompt, build_chat_prompt

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3-flash-preview"

# ─── Gọi Gemini lần đầu kèm ảnh ──────────────────────────
def analyze_fish(species: str, confidence: float, image_input) -> tuple[str, list]:
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt_data   = build_initial_prompt(species, confidence, image_base64)
    system_prompt = prompt_data["system"]
    user_message  = prompt_data["first_message"]

    response = client.models.generate_content(
        model  = MODEL_NAME,
        config = types.GenerateContentConfig(system_instruction=system_prompt),
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            user_message,
        ]
    )

    response_text = response.text
    chat_history  = [
        {"role": "user",  "parts": user_message},
        {"role": "model", "parts": response_text},
    ]

    return response_text, chat_history


# ─── Chat tiếp theo ───────────────────────────────────────
def chat(species: str, chat_history: list, user_message: str) -> tuple[str, list]:
    prompt_data   = build_chat_prompt(species, chat_history, user_message)
    system_prompt = prompt_data["system"]

    # Chuyển history sang format mới
    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role  = role,
            parts = [types.Part(text=msg["parts"])]
        ))
    contents.append(types.Content(
        role  = "user",
        parts = [types.Part(text=user_message)]
    ))

    response = client.models.generate_content(
        model    = MODEL_NAME,
        config   = types.GenerateContentConfig(system_instruction=system_prompt),
        contents = contents,
    )

    response_text = response.text
    chat_history  = chat_history + [
        {"role": "user",  "parts": user_message},
        {"role": "model", "parts": response_text},
    ]

    return response_text, chat_history


# ─── Test nhanh ───────────────────────────────────────────
if __name__ == "__main__":
    print("Test analyze_fish với ảnh arowana...\n")

    response, history = analyze_fish(
        species     = "arowana",
        confidence  = 0.96,
        image_input = "data/raw/arowana/check2.jpg"
    )

    print("🤖 Gemini:\n")
    print(response)
    print("\n" + "="*50)

    # Test chat tiếp
    user_q = "Bể nuôi cá rồng cần bao nhiêu lít?"
    print(f"\n👤 User: {user_q}\n")

    response2, history = chat("arowana", history, user_q)
    print(f"🤖 Gemini:\n{response2}")