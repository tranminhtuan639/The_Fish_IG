# chatbot/prompt_builder.py
from rag.retriever import retrieve_by_topic, retrieve_all

# ─── Tên tiếng Việt ───────────────────────────────────────
SPECIES_VI = {
    "arowana":    "Cá Rồng (Arowana)",
    "oscar":      "Cá Oscar",
    "flowerhorn": "Cá La Hán (Flowerhorn)",
    "snakehead":  "Cá Lóc (Snakehead)",
}

# ─── Prompt lần đầu: nhận diện ảnh ───────────────────────
def build_initial_prompt(species: str, confidence: float, image_base64: str) -> list:
    """
    Prompt gửi kèm ảnh → Gemini Vision phân tích màu → đoán biến thể.

    Returns:
        list: messages format cho Gemini API
    """
    species_vi   = SPECIES_VI.get(species, species)
    variants_ctx = retrieve_by_topic(species, "variants")
    care_ctx     = retrieve_by_topic(species, "care")
    food_ctx     = retrieve_by_topic(species, "food")
    confidence_pct = f"{confidence * 100:.1f}%"

    system_prompt = f"""Bạn là chuyên gia cá cảnh với nhiều năm kinh nghiệm.
Người dùng vừa tải lên ảnh một con cá được hệ thống nhận diện là **{species_vi}** (độ tin cậy: {confidence_pct}).

Nhiệm vụ của bạn:
1. Quan sát màu sắc, hoa văn trên ảnh
2. Đối chiếu với danh sách biến thể bên dưới → đưa ra 1-2 biến thể có thể nhất
3. Giới thiệu ngắn gọn về biến thể đó: thức ăn, chăm sóc, kích thước, giá

Trả lời bằng tiếng Việt, thân thiện, rõ ràng. Không bịa thông tin ngoài tài liệu.

---
📋 TÀI LIỆU BIẾN THỂ:
{variants_ctx}

---
🏠 TÀI LIỆU CHĂM SÓC:
{care_ctx}

---
🍖 TÀI LIỆU THỨC ĂN:
{food_ctx}
"""

    return {
        "system": system_prompt,
        "image_base64": image_base64,
        "first_message": f"Đây là ảnh con cá của tôi, hệ thống nhận diện là {species_vi}. Bạn có thể cho tôi biết đây là biến thể nào không?"
    }


# ─── Prompt chat tiếp theo: hỏi đáp tự do ────────────────
def build_chat_prompt(species: str, chat_history: list, user_message: str) -> dict:
    """
    Prompt cho các lượt chat sau khi đã giới thiệu xong.

    Args:
        species      : loài đã nhận diện
        chat_history : list các lượt chat trước [{"role": "user/model", "parts": "..."}]
        user_message : câu hỏi mới của user

    Returns:
        dict: system + history + user_message
    """
    species_vi = SPECIES_VI.get(species, species)
    all_ctx    = retrieve_all(species)

    system_prompt = f"""Bạn là chuyên gia cá cảnh, đang tư vấn về **{species_vi}**.

Trả lời câu hỏi của người dùng dựa trên tài liệu bên dưới.
Nếu câu hỏi nằm ngoài tài liệu, hãy nói thật là bạn không chắc chắn.
Trả lời bằng tiếng Việt, ngắn gọn, thực tế.

---
📋 BIẾN THỂ:
{all_ctx["variants"]}

---
🏠 CHĂM SÓC:
{all_ctx["care"]}

---
🍖 THỨC ĂN:
{all_ctx["food"]}
"""

    return {
        "system":       system_prompt,
        "history":      chat_history,
        "user_message": user_message,
    }