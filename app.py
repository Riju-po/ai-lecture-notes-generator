# app.py

import streamlit as st
import os
import tempfile
import sys
import zipfile
import io
from weasyprint import HTML, CSS

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
    st.header("Developed by Gourav Das")

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
            Simply upload your audio, or provide an existing transcription, and the app will generate a comprehensive summary
            that you can download and review.
            """
        )

    st.divider()
    # Reset button
    if st.button("Reset Application", help="Clear all uploaded files and generated notes"):
        st.session_state.clear()
        st.rerun()

# --- Define the CSS for the PDF footer ---
pdf_footer_css = CSS(string='''
    @page {
        @bottom-center {
            content: "Created By Gourav Das";
            font-size: 10pt;
            color: #555;
            margin-top: 10mm;
        }
    }
''')

# --- Helper function to process transcription and generate PDF (reused for all tabs) ---
def process_transcription_and_generate_output(transcript_text, file_identifier, key_suffix=""):
    """
    Processes transcription text to generate notes and PDF.
    `file_identifier` is used for display and naming output files.
    Includes progress bar for notes generation.
    """
    if not gemini_api_key:
        st.error(f"Gemini API key is not configured. Cannot generate notes for {file_identifier}.", icon="🚫")
        return

    if not transcript_text.strip():
        st.warning(f"Transcription for {file_identifier} is empty. Cannot generate notes.", icon="⚠️")
        return

    notes_placeholder = st.empty() # Placeholder for note generation progress
    with notes_placeholder.container():
        notes_progress_bar = st.progress(0, text=f"Generating detailed notes for {file_identifier} with AI...")
        # Note: Gemini API doesn't offer real-time progress for a single call,
        # so we'll simulate a start and end for the progress bar.
        notes_progress_bar.progress(10) # Indicate start
        notes_content = generate_notes_with_gemini_api(transcript_text, gemini_api_key, NOTE_STYLE_PROMPT)
        notes_progress_bar.progress(100) # Indicate completion

    if "Cannot generate notes" in notes_content or "Could not generate notes using Gemini API" in notes_content:
        st.error(f"Note generation failed for '{file_identifier}': {notes_content}", icon="❌")
        notes_placeholder.empty() # Clear the progress bar on error
        return
    else:
        st.success(f"Notes generated successfully for '{file_identifier}'!", icon="✅")
        notes_placeholder.empty() # Clear the progress bar on success
        with st.expander(f"View Generated Notes for {file_identifier}"):
            st.markdown(notes_content)

        st.info(f"Creating PDF document for {file_identifier}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf_file:
            current_pdf_path = tmp_pdf_file.name

        pdf_title = f"Lecture Notes: {file_identifier.replace(' ', '_')}"
        create_pdf_from_text(notes_content, current_pdf_path, pdf_title,
                             stylesheets=[pdf_footer_css])

        if os.path.exists(current_pdf_path):
            st.success(f"PDF created for '{file_identifier}'!", icon="✅")
            with open(current_pdf_path, "rb") as file:
                st.download_button(
                    label=f"Download Notes PDF for {file_identifier}",
                    data=file,
                    file_name=f"{file_identifier.replace(' ', '_')}_notes.pdf",
                    mime="application/pdf",
                    key=f"download_button_{key_suffix}_{file_identifier}"
                )
            all_generated_pdf_paths.append(current_pdf_path)
            all_generated_pdf_names.append(f"{file_identifier.replace(' ', '_')}_notes.pdf")
        else:
            st.error(f"Failed to create PDF for '{file_identifier}'.", icon="❌")


# --- Main Application User Interface ---
st.title("🎧 AI-Powered Lecture Notes Generator 📝")
st.markdown(
    """
    **Transform your audio lectures or raw text into highly organized and comprehensive study notes in PDF format.**
    Simply upload your audio, or provide an existing transcription, and the app will generate a comprehensive summary
    that you can download and review.
    """
)

st.divider()

# --- Tabbed Interface for Input Options ---
tab1, tab2, tab3 = st.tabs([
    "Upload Audio",
    "Upload Transcription File",
    "Paste Transcription Text"
])

all_generated_pdf_paths = []
all_generated_pdf_names = []


# --- Tab 1: Upload Audio ---
with tab1:
    st.header("1. Upload Audio for Transcription")
    st.info("Upload an audio file. Transcription will start automatically, followed by AI note generation.")
    st.markdown("---")

    audio_file = st.file_uploader(
        "📂 **Upload your audio file here**",
        type=["mp3", "wav", "m4a", "flac", "ogg"],
        key="audio_uploader_tab1",
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG. Max file size depends on hosting."
    )

    if audio_file:
        st.audio(audio_file, format=audio_file.type)
        st.success("Audio file uploaded successfully! Starting transcription...", icon="✅")

        if whisper_model_instance:
            audio_path = None
            try:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_audio_file:
                    tmp_audio_file.write(audio_file.read())
                    audio_path = tmp_audio_file.name

                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    st.error(f"Error: Uploaded audio file '{audio_file.name}' is empty or corrupted.", icon="❌")
                else:
                    transcription_placeholder = st.empty() # Placeholder for transcription progress
                    with transcription_placeholder.container():
                        transcription_progress_bar = st.progress(0, text=f"Transcribing audio '{audio_file.name}' with Whisper... (This may take a while)")
                        # Pass the progress_callback to the transcribe function
                        transcribed_content = transcribe_audio_whisper(audio_path, whisper_model_instance, progress_callback=lambda p: transcription_progress_bar.progress(p))
                        transcription_progress_bar.progress(100) # Ensure it reaches 100%

                    if "Error:" in transcribed_content or "no speech" in transcribed_content or not transcribed_content.strip():
                        st.error(f"Transcription failed for '{audio_file.name}': {transcribed_content}", icon="❌")
                        transcribed_content = None
                        transcription_placeholder.empty() # Clear the progress bar on error
                    else:
                        st.success(f"Transcription complete for '{audio_file.name}'!", icon="✅")
                        transcription_placeholder.empty() # Clear the progress bar on success
                        display_transcription = transcribed_content
                        if len(display_transcription) > 1000:
                            display_transcription = display_transcription[:1000] + "\n\n... (Transcription truncated for display)"

                        with st.expander(f"Raw Transcription for {audio_file.name}"):
                            st.text_area(f"Raw Transcript:", display_transcription, height=150,
                                         key=f"transcription_text_{audio_file.name.replace('.', '_')}")

                        # Call the helper function to process transcript and generate notes/PDF
                        process_transcription_and_generate_output(transcribed_content, audio_file.name, key_suffix="audio")
            except Exception as e:
                st.error(f"An unexpected error occurred during audio processing of '{audio_file.name}': {e}", icon="🚨")
            finally:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
        else:
            st.error(
                f"Transcription service is unavailable. Please check the backend setup (Whisper model failed to load).",
                icon="❌")


# --- Tab 2: Upload Transcription File ---
with tab2:
    st.header("2. Upload Existing Transcription File")
    st.info("Upload a text file containing your transcription. Notes will be generated by Gemini AI.")
    st.markdown("---")

    uploaded_transcription_file = st.file_uploader("Upload a text file (.txt, .md, etc.)", type=["txt", "md"], key="file_uploader_tab2")

    if uploaded_transcription_file is not None:
        # Read the content of the uploaded file
        string_io = io.StringIO(uploaded_transcription_file.getvalue().decode("utf-8"))
        uploaded_transcript_text = string_io.read()

        if uploaded_transcript_text.strip(): # Check if content is not empty
            st.subheader("Uploaded Transcription Content:")
            st.text_area("Review Uploaded Transcription:", uploaded_transcript_text, height=300, key="uploaded_transcript_display_tab2")

            # Trigger processing immediately upon valid upload
            process_transcription_and_generate_output(uploaded_transcript_text, uploaded_transcription_file.name, key_suffix="file")
        else:
            st.warning("The uploaded file appears to be empty or could not be read. Please upload a file with content.", icon="⚠️")

# --- Tab 3: Paste Transcription Text ---
with tab3:
    st.header("3. Paste Transcription Text")
    st.info("Paste your transcription directly into the text area below. Notes will be generated by Gemini AI.")
    st.markdown("---")

    pasted_transcript_text = st.text_area("Paste your lecture transcription here:", height=400, key="pasted_transcript_input_tab3")

    if pasted_transcript_text.strip(): # Check if content is not empty
        st.subheader("Pasted Transcription Content:")
        st.text_area("Review Pasted Text:", pasted_transcript_text, height=300, key="pasted_transcript_display_tab3")

        # Trigger processing immediately upon valid paste (if text is present)
        process_transcription_and_generate_output(pasted_transcript_text, "Pasted Text", key_suffix="paste")
    else:
        st.info("Paste your lecture transcription into the text area above to generate notes.")


# --- Download All Notes (Combined for all sources) ---
st.divider()
if all_generated_pdf_paths:
    st.success(f"All {len(all_generated_pdf_paths)} lecture notes have been processed across all inputs!", icon="🎉")
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
        key="download_all_notes_button_final"
    )

    st.info(
        "Your temporary files will be cleaned up automatically after download or app refresh. For a fresh start, use the 'Reset Application' button in the sidebar.",
        icon="🗑️")

    # Clean up temporary PDF files
    for pdf_path in all_generated_pdf_paths:
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception as e:
                st.warning(f"Could not delete temporary PDF file '{pdf_path}': {e}")
