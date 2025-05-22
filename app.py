# app.py

import streamlit as st
import os
import tempfile
import sys
import zipfile
import io

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Lecture Notes Generator",
    page_icon="📝",
    layout="wide"
)
# --- End Streamlit Page Configuration ---


# Add the parent directory of core_processing.py to the Python path
sys.path.append(os.path.dirname(__file__))

# Import functions and configurations from your core_processing.py file.
from core_processing import (
    transcribe_audio_whisper,
    generate_notes_with_gemini_api,
    create_pdf_from_text,
    WHISPER_MODEL_SIZE,
    NOTE_STYLE_PROMPT
)

# Import necessary libraries that core_processing depends on, specifically for Streamlit's
# caching mechanism or direct use in app.py's setup.
import whisper
import google.generativeai as genai


# --- Initial Setup for Streamlit App ---

# Load Whisper model once when the app starts.
@st.cache_resource
def load_whisper_model(model_size):
    """Loads the Whisper model, caches it, and ensures CPU usage."""
    try:
        with st.spinner(f"Loading Whisper model ({model_size})... This may take a moment."):
            model = whisper.load_model(model_size, device="cpu")
        st.success(f"Whisper model ({model_size}) loaded successfully!", icon="✅")
        return model
    except Exception as e:
        st.error(
            f"FATAL ERROR: Could not load Whisper model. Please check your installation and dependencies (e.g., PyTorch, FFmpeg). Error: {e}",
            icon="❌")
        return None


# Load the Whisper model instance.
whisper_model_instance = load_whisper_model(WHISPER_MODEL_SIZE)

# --- Gemini API Key Management (Crucial for Deployment Security) ---
# Check if API key is in Streamlit secrets (recommended for deployment)
gemini_api_key = st.secrets.get("GEMINI_API_KEY")

# --- Sidebar ---
with st.sidebar:
    st.header("Developed by Gourav Das")  # Added this header

    # Manual API key input for local testing if not in secrets
    if not gemini_api_key:
        st.warning(
            "Gemini API key not found in Streamlit secrets. Please enter it below for full functionality (for local testing only).",
            icon="⚠️")
        gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")
        if not gemini_api_key:
            st.error("Please provide a Gemini API key to proceed.", icon="🚫")
        else:
            st.success("API key accepted!", icon="🔑")
    else:
        st.success("Gemini API key loaded securely.", icon="🔒")
        st.info("Your API key is managed securely via Streamlit secrets for deployment.", icon="ℹ️")

    st.divider()

    # About Section - Simplified description
    with st.expander("About This App"):
        st.markdown(
            """
            This application helps you transform your spoken lectures and audio recordings
            into organized, readable study notes in PDF format.
            Simply upload your audio, and the app will generate a comprehensive summary
            that you can download and review.
            """
        )

    st.divider()
    # Reset button
    if st.button("Reset Application", help="Clear all uploaded files and generated notes"):
        st.session_state.clear()
        st.rerun()

# --- Main Application User Interface ---
st.title("🎧 AI-Powered Lecture Notes Generator 📝")
st.markdown(
    """
    **Transform your audio lectures into highly organized and comprehensive study notes in PDF format.**
    Upload one or more audio files, and let AI do the heavy lifting!
    """
)

st.divider()

# File Uploader Widget
uploaded_files = st.file_uploader(
    "📂 **Upload your audio files here**",
    type=["mp3", "wav", "m4a", "flac", "ogg"],
    accept_multiple_files=True,
    help="Supported formats: MP3, WAV, M4A, FLAC, OGG. Max file size depends on hosting (typically 25-200MB per file)."
)

all_generated_pdf_paths = []
all_generated_pdf_names = []

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s). This may take a while...", icon="⏳")
    st.divider()

    for i, uploaded_file in enumerate(uploaded_files):
        with st.container(border=True):
            st.subheader(f"✨ Processing: {uploaded_file.name}")
            st.audio(uploaded_file, format=uploaded_file.type)

            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_audio_file:
                tmp_audio_file.write(uploaded_file.read())
                audio_path = tmp_audio_file.name

            current_pdf_path = None

            try:
                if whisper_model_instance:
                    with st.spinner(f"Step 1/3 ({uploaded_file.name}): Transcribing audio..."):
                        transcribed_content = transcribe_audio_whisper(audio_path, whisper_model_instance)

                    if "Error:" in transcribed_content or "no speech" in transcribed_content:
                        st.error(f"Transcription failed for '{uploaded_file.name}': {transcribed_content}", icon="❌")
                        transcribed_content = None
                    else:
                        st.success(f"Transcription complete for '{uploaded_file.name}'!", icon="✅")
                        display_transcription = transcribed_content
                        if len(display_transcription) > 1000:
                            display_transcription = display_transcription[
                                                    :1000] + "\n\n... (Transcription truncated for display)"

                        with st.expander(f"Raw Transcription for {uploaded_file.name}"):
                            st.text_area(f"Raw Transcript:", display_transcription, height=150,
                                         key=f"transcription_text_{i}")

                        if transcribed_content and gemini_api_key:
                            with st.spinner(f"Step 2/3 ({uploaded_file.name}): Generating detailed notes with AI..."):
                                notes_content = generate_notes_with_gemini_api(transcribed_content, gemini_api_key,
                                                                               NOTE_STYLE_PROMPT)

                            if "Cannot generate notes" in notes_content or "Could not generate notes using Gemini API" in notes_content:
                                st.error(f"Note generation failed for '{uploaded_file.name}': {notes_content}",
                                         icon="❌")
                                notes_content = None
                            else:
                                st.success(f"Notes generated successfully for '{uploaded_file.name}'!", icon="✅")
                                with st.expander(f"View Generated Notes for {uploaded_file.name}"):
                                    st.markdown(notes_content)

                                    with st.spinner(f"Step 3/3 ({uploaded_file.name}): Creating PDF document..."):
                                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
                                            current_pdf_path = tmp_pdf_file.name

                                        pdf_title = f"Lecture Notes: {os.path.splitext(uploaded_file.name)[0]}"
                                        create_pdf_from_text(notes_content, current_pdf_path, pdf_title)

                                    if os.path.exists(current_pdf_path):
                                        st.success(f"PDF created for '{uploaded_file.name}'!", icon="✅")
                                        with open(current_pdf_path, "rb") as file:
                                            st.download_button(
                                                label=f"Download Notes PDF for {uploaded_file.name}",
                                                data=file,
                                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_notes.pdf",
                                                mime="application/pdf",
                                                key=f"download_button_single_{i}"
                                            )
                                        all_generated_pdf_paths.append(current_pdf_path)
                                        all_generated_pdf_names.append(
                                            f"{os.path.splitext(uploaded_file.name)[0]}_notes.pdf")
                                    else:
                                        st.error(f"Failed to create PDF for '{uploaded_file.name}'.", icon="❌")
                        elif not gemini_api_key:
                            st.warning(
                                f"AI note generation is disabled for '{uploaded_file.name}' as the API key is not configured.",
                                icon="⚠️")
                else:
                    st.error(
                        f"Transcription service is unavailable for '{uploaded_file.name}'. Please check the backend setup.",
                        icon="❌")

            except Exception as e:
                st.error(f"An unexpected error occurred during processing of '{uploaded_file.name}': {e}", icon="🚨")
            finally:
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.remove(audio_path)

            st.markdown("---")

    if all_generated_pdf_paths:
        st.success(f"All {len(all_generated_pdf_paths)} lecture notes have been processed!", icon="🎉")
        st.subheader("Download All Notes")

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, pdf_path in enumerate(all_generated_pdf_paths):
                if os.path.exists(pdf_path):
                    zf.write(pdf_path, arcname=all_generated_pdf_names[i])

        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All Notes as ZIP",
            data=zip_buffer,
            file_name="All_Lecture_Notes.zip",
            mime="application/zip",
            key="download_all_notes_button"
        )

        st.info(
            "Your temporary files will be cleaned up automatically after download or app refresh. For a fresh start, use the 'Reset Application' button in the sidebar.",
            icon="🗑️")

    for pdf_path in all_generated_pdf_paths:
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception as e:
                st.warning(f"Could not delete temporary PDF file '{pdf_path}': {e}")