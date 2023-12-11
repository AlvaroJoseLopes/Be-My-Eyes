import streamlit as st
from gtts import gTTS

import base64
import requests
from io import BytesIO


# Parameters and utils function
MODEL_NAMES = ["pretrained", "trained"]
ALLOWED_IMG_TYPES = ["png", "jpeg", "jpg"]
URL = "http://127.0.0.1:8000/inference"


@st.cache_data(show_spinner="Fetching backend API to get image caption...")
def get_caption(model: str, image: bytes, filename: str) -> str:
    params = {"model_name": model}
    content = base64.b64encode(image).decode("utf-8")
    data = {"filename": filename, "content": content}
    try:
        res = requests.post(
            URL,
            params=params,
            json=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        data = res.json()
        if res.status_code == 200:
            return data["caption"]
        else:
            print(f"Message error from the API: {data}")
            return f"Unable to get caption"
    except Exception as e:
        print(e)
        return f"Unable to get caption"


@st.cache_data(show_spinner="Converting caption text to speech...")
def get_audio(caption: str) -> BytesIO:
    mp3_fp = BytesIO()
    tts = gTTS(caption, lang="en")
    tts.write_to_fp(mp3_fp)

    return mp3_fp


# Title config
st.set_page_config(
    page_title="Be My Eyes",
    page_icon=":eye:",
)
st.title("Be My Eyes :eye:")

# User flow
st.markdown("### Select Model")
model = st.selectbox("Select a model", options=MODEL_NAMES, label_visibility="hidden")

st.markdown("### Upload a image file")
image = st.file_uploader(
    "Allowed types: ", type=ALLOWED_IMG_TYPES, label_visibility="hidden"
)

if image is not None:
    bytes_data = image.read()
    caption = get_caption(model, bytes_data, image.name)
    st.image(bytes_data, caption=caption)

    st.markdown("### Audio Description")
    audio_bytes = get_audio(caption)
    st.audio(audio_bytes, format="audio/mp3", start_time=0)
