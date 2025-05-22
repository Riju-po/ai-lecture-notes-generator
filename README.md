# AI Lecture Notes Generator

This Streamlit application allows you to generate comprehensive lecture notes from audio recordings using Google's Gemini API for summarization and Whisper for transcription. It also supports converting the generated notes into a PDF format.

## Features

* **Audio Transcription:** Converts spoken audio into text using OpenAI's Whisper model.
* **AI-Powered Summarization:** Uses Google Gemini to summarize transcribed text into concise lecture notes.
* **PDF Export:** Generates downloadable PDF documents of your summarized notes using WeasyPrint.

## How to Use

1.  **Upload Audio:** Drag and drop your audio file (e.g., MP3, WAV) into the uploader.
2.  **Transcribe:** The app will automatically transcribe the audio.
3.  **Generate Notes:** The AI will then process the transcript to create structured notes.
4.  **Download PDF:** Click the "Download PDF" button to save your notes.

## Deployment

This application is deployed on Streamlit Cloud.
[Link to your deployed app here] - *Replace with your actual Streamlit Cloud app URL.*

## Technologies Used

* [Streamlit](https://streamlit.io/)
* [Google Generative AI (Gemini API)](https://ai.google.dev/)
* [OpenAI Whisper](https://github.com/openai/whisper)
* [WeasyPrint](https://weasyprint.org/) (for PDF generation)
* Python

## Installation (for local development, optional)

1.  Clone this repository:
    `git clone https://github.com/your-username/AI-Lecture-Notes-Generator.git`
    `cd AI-Lecture-Notes-Generator`
2.  Create a virtual environment and activate it.
3.  Install dependencies:
    `pip install -r requirements.txt`
4.  Ensure `ffmpeg` is installed on your system.
5.  Run the app:
    `streamlit run app.py`

## License

All Rights Reserved. Copyright (c) 2025 Gourav Das.
