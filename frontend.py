from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
from gtts import gTTS

import requests
from io import BytesIO

# Parameters and utils function
MODEL_NAMES = ["pretrained", "trained"]
ALLOWED_IMG_TYPES = ["png", "jpeg", "jpg"]
URL = "http://127.0.0.1:8000/inference"


@st.cache_data(show_spinner="Fetching backend API to get image caption...")
def get_caption(model: str, image: UploadedFile):
    params = {"model_name": model, "image_url": "teste"}
    try:
        res = requests.get(URL, params=params)
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
def get_audio(caption: str):
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
    caption = get_caption(model, image)
    st.image(bytes_data, caption=caption)

    st.markdown("### Audio Description")
    audio_bytes = get_audio(caption)
    st.audio(audio_bytes, format="audio/mp3", start_time=0)
