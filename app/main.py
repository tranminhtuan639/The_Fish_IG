# app/main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# streamlit run app/main.py
import streamlit as st
from PIL import Image
from model.predict import predict, load_model
from chatbot.llm_client import analyze_fish, chat

# ─── Config trang ─────────────────────────────────────────
st.set_page_config(
    page_title = "🐟 Fish Expert",
    page_icon  = "🐟",
    layout     = "centered"
)

SPECIES_VI = {
    "arowana":    "Cá Rồng (Arowana)",
    "oscar":      "Cá Oscar",
    "flowerhorn": "Cá La Hán (Flowerhorn)",
    "snakehead":  "Cá Lóc (Snakehead)",
}

# ─── Load model 1 lần ─────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()

# ─── Session state ────────────────────────────────────────
def init_state():
    defaults = {
        "phase":        "upload",   # upload | result | chat
        "species":      None,
        "confidence":   None,
        "chat_history": [],
        "messages":     [],         # Hiển thị UI
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Header ───────────────────────────────────────────────
st.title("🐟 Fish Expert")
st.caption("Tải ảnh cá cảnh lên để nhận diện loài và hỏi đáp chuyên gia")
st.divider()

# ══════════════════════════════════════════════════════════
# PHASE 1: Upload ảnh
# ══════════════════════════════════════════════════════════
if st.session_state.phase == "upload":
    uploaded = st.file_uploader(
        "Chọn ảnh cá cảnh",
        type   = ["jpg", "jpeg", "png", "webp"],
        help   = "Hỗ trợ JPG, PNG, WEBP"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Ảnh đã tải lên", use_container_width=True)

        if st.button("🔍 Nhận diện ngay", use_container_width=True, type="primary"):
            with st.spinner("Đang nhận diện loài cá..."):
                cv_model, classes = get_model()
                result = predict(img, model=cv_model, classes=classes)

            if result["status"] == "uncertain":
                st.warning(result["message"])
            else:
                st.session_state.species    = result["species"]
                st.session_state.confidence = result["confidence"]

                with st.spinner("Đang phân tích chi tiết với AI..."):
                    response, history = analyze_fish(
                        species     = result["species"],
                        confidence  = result["confidence"],
                        image_input = img,
                    )

                st.session_state.chat_history = history
                st.session_state.messages     = [
                    {"role": "assistant", "content": response}
                ]
                st.session_state.phase = "chat"
                st.rerun()

# ══════════════════════════════════════════════════════════
# PHASE 2: Chat
# ══════════════════════════════════════════════════════════
elif st.session_state.phase == "chat":
    species    = st.session_state.species
    confidence = st.session_state.confidence
    species_vi = SPECIES_VI.get(species, species)

    # ── Info bar ──────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### {species_vi}")
    with col2:
        st.metric("Độ tin cậy", f"{confidence:.1%}")
    with col3:
        if st.button("📷 Ảnh mới"):
            for k in ["phase","species","confidence","chat_history","messages"]:
                del st.session_state[k]
            st.rerun()

    st.divider()

    # ── Hiển thị lịch sử chat ─────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input chat ────────────────────────────────────────
    user_input = st.chat_input(f"Hỏi thêm về {species_vi}...")

    if user_input:
        # Hiển thị câu hỏi user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Gọi Gemini
        with st.chat_message("assistant"):
            with st.spinner("Đang trả lời..."):
                response, updated_history = chat(
                    species      = species,
                    chat_history = st.session_state.chat_history,
                    user_message = user_input,
                )
            st.markdown(response)

        st.session_state.chat_history = updated_history
        st.session_state.messages.append({"role": "assistant", "content": response})