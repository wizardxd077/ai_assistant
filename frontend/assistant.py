import streamlit as st
import requests
import os

st.set_page_config(page_title="Multimodal AI Assistant", page_icon="🤖")

st.title("🤖 Multimodal AI Assistant")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Select Interaction Mode", 
    ["Text Processing", "Image Classification", "Speech Recognition"]
)

# API Configuration
API_BASE_URL = "http://localhost:5000"

def process_text(text):
    response = requests.post(f"{API_BASE_URL}/process/text", json={"text": text})
    return response.json().get("response", "No response")

def process_image(uploaded_file):
    files = {'image': uploaded_file}
    response = requests.post(f"{API_BASE_URL}/process/image", files=files)
    return response.json().get("classification", "Unable to classify")

def process_audio(uploaded_file):
    files = {'audio': uploaded_file}
    response = requests.post(f"{API_BASE_URL}/process/audio", files=files)
    return response.json().get("transcription", "Unable to transcribe")

if mode == "Text Processing":
    st.header("🔤 Text Processing")
    text_input = st.text_area("Enter text for processing")
    if st.button("Process Text"):
        result = process_text(text_input)
        st.write("Processed Result:", result)

elif mode == "Image Classification":
    st.header("🖼️ Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify Image"):
            result = process_image(uploaded_file)
            st.write("Classification Result:", result)

else:  # Speech Recognition
    st.header("🎤 Speech Recognition")
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Transcribe Audio"):
            result = process_audio(uploaded_file)
            st.write("Transcription:", result)

st.sidebar.info("Multimodal AI Assistant processes text, images, and audio using advanced transformer models.")
