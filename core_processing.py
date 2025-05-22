# core_processing.py

# Core Libraries
import os
import json
import nltk
import time
import tempfile

# Google Gemini API
import google.generativeai as genai

# OpenAI Whisper for Transcription
import whisper
import torch

# For Robust Markdown to PDF Conversion
import markdown
from weasyprint import HTML, CSS # Ensure HTML and CSS are imported here

# --- NLTK Data Check (ensure these are downloaded) ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'stopwords' or 'punkt' data not found. Attempting to download...")
    try:
        nltk.download('stopwords')
        nltk.download('punkt')
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Failed to download NLTK data: {e}. Some text processing features might be affected.")

# --- Global Whisper Model Configuration ---
WHISPER_MODEL_SIZE = "base.en"

# --- Configuration for Note Generation ---
# This prompt provides additional specific instructions to Gemini for note generation.
# Customize this string heavily to guide Gemini to generate notes exactly how you want them
# for specific lecture types or content.
NOTE_STYLE_PROMPT = """
Ensure the notes are suitable for a university-level data science course.
Focus on core concepts, algorithms, and practical implementation details.
Include theoretical definitions, practical application examples, and specific library/framework mentions (e.g., Pandas, Scikit-learn, TensorFlow) if discussed.
Also, extract any open questions or discussion points raised during the lecture for further exploration.
"""

# Function to Transcribe Audio with Whisper
def transcribe_audio_whisper(audio_file_path, whisper_model_instance):
    """
    Transcribes an audio file into text using a pre-loaded Whisper model instance.
    Handles file existence checks and basic error reporting.
    """
    print(f"\n--- Transcribing: {os.path.basename(audio_file_path)} ---")
    if whisper_model_instance is None:
        return "Error: Whisper model not loaded. Cannot transcribe."

    try:
        if not os.path.exists(audio_file_path):
            return f"Error: Audio file not found at '{audio_file_path}'"

        result = whisper_model_instance.transcribe(audio_file_path)
        transcribed_text = result["text"]

        if not transcribed_text.strip():
            return "Whisper recognized no speech or produced empty transcription."
        else:
            return transcribed_text.strip()

    except Exception as e:
        return f"An unexpected error occurred during Whisper transcription: {e}"

# Function to Generate Notes (with Gemini API)
def generate_notes_with_gemini_api(text_input, api_key, note_style_prompt=""):
    """
    Generates structured, detailed, and visually appealing notes from transcribed text
    using the Gemini API with a refined prompt for Markdown output.
    Takes API key as an argument for secure handling.
    """
    print(f"\n--- Generating Note with Gemini API ---")

    if not text_input or "Error:" in text_input or "no speech" in text_input:
        return "Cannot generate notes from empty or failed transcription."
    if not api_key:
        return "Gemini API key is not set. Cannot generate notes."

    try:
        genai.configure(api_key=api_key)

        # The detailed prompt for Gemini
        enhanced_prompt = f"""
        You are an expert data science instructor specializing in creating highly detailed and structured study notes from lecture transcripts. Your goal is to produce notes that are visually appealing and comprehensive, suitable for a university-level data science course.

        The following is a transcript from an audio recording of a data science lecture.

        **Instructions for Note Generation:**
        1.  **Strictly use ONE Level 1 Heading (H1, i.e., #) ONLY at the very beginning of the notes.** This will be the main title of the lecture. Do NOT include any other H1 headings.
        2.  **Organize the content into logical sections using Level 2 Headings (H2, i.e., ##).** For complex topics, use Level 3 Headings (H3, i.e., ###) for sub-sections.
        3.  **Ensure notes are highly detailed and comprehensive.** Elaborate on concepts, provide explanations, and include sufficient context. Do not just list bullet points; expand on them.
        4. Within each section, use:
            * **Bullet points (`*` or `-`)** for key concepts, definitions, and short explanations.
            * **Numbered lists (`1.`)** for ordered steps, processes, or algorithms.
            * **Bold text (`**text**`)** for important terms, keywords, or highlights.
            * **Italic text (`*text*` or `_text_`)** for emphasis or foreign terms.
            * **Code examples** should be formatted within a distinct code block (```python ... ```, ```sql ... ```, or similar, indicating the language if applicable). Ensure code is syntactically correct and includes comments if useful.
            * **Mathematical expressions:** Enclose inline math within single dollar signs, e.g., `$E=mc^2$`. For display math, use double dollar signs, e.g., `$$F=ma$$`. (Note: WeasyPrint will display these literally, not render them as typeset math symbols).
        5.  **Always include a "Key Concepts" or "Vocabulary" section.** Define terms clearly.
        6.  **Always include a "Practical Applications" or "Use Cases" section.** Describe how the concepts are applied in real-world data science scenarios.
        7.  **If any questions or discussions about future topics arise in the transcript, summarize them** under a "Discussion Points" or "Further Exploration" section.
        8.  **Conclude with a "Summary" section** that briefly recaps the main takeaways. If applicable, create a simple Markdown table for a summary of key components (like concept-description pairs).
        9.  **Maintain a formal, academic tone.**
        10. **Do NOT include conversational filler, speaker remarks, or irrelevant digressions.** Focus purely on the data science content.
        11. **Ensure a smooth and logical flow** of information throughout the notes.

        **Additional Specific Focus (from Note Style Prompt, if provided):**
        {note_style_prompt if note_style_prompt else "No additional specific focus provided. Summarize comprehensively based on general data science lecture content."}

        **Transcript:**
        ---
        {text_input}
        ---

        **Please generate the notes in Markdown format, strictly adhering to the structure, detail, and visual appeal guidelines above.**
        """

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(enhanced_prompt)

        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            return f"Gemini API returned an empty or unreadable response. Debug info: {response}"

    except Exception as e:
        print(f"An error occurred during Gemini API note generation: {e}")
        return f"Could not generate notes using Gemini API: {e}"

# Function to Create PDF from Text (Using Markdown and WeasyPrint)
# Updated to accept optional stylesheets
def create_pdf_from_text(markdown_text, output_pdf_path, title="Audio Transcription Notes", stylesheets: list = None):
    """
    Converts Markdown text into a visually appealing and well-structured PDF
    using the 'markdown' library to convert Markdown to HTML, and 'WeasyPrint'
    to render that HTML into a PDF document. Includes comprehensive CSS for styling
    and applies additional stylesheets for headers/footers.
    """
    print(f"\n--- Creating PDF with robust Markdown rendering ---")
    if stylesheets is None:
        stylesheets = [] # Initialize as empty list if no stylesheets are provided

    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                @page {{
                    size: A4;
                    margin: 1in;
                }}
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    font-size: 10pt;
                    margin: 0;
                    padding: 0;
                }}
                h1 {{
                    font-size: 2.2em;
                    color: #1a237e;
                    text-align: center;
                    margin-top: 1.5em;
                    margin-bottom: 0.8em;
                    border-bottom: 3px double #b0c4de;
                    padding-bottom: 0.7em;
                    padding-left: 0;
                }}
                h2 {{
                    font-size: 1.6em;
                    color: #2c3e50;
                    margin-top: 1.5em;
                    margin-bottom: 0.7em;
                    border-left: 6px solid #3498db;
                    padding-left: 15px;
                    background-color: #f8fbfc;
                    padding-top: 5px;
                    padding-bottom: 5px;
                }}
                h3 {{
                    font-size: 1.3em;
                    color: #34495e;
                    margin-top: 1.2em;
                    margin-bottom: 0.6em;
                    border-bottom: 1px dashed #cccccc;
                    padding-bottom: 0.3em;
                }}
                h4 {{
                    font-size: 1.1em;
                    color: #555;
                    margin-top: 1em;
                    margin-bottom: 0.4em;
                }}
                p {{
                    margin-bottom: 1em;
                    text-align: justify;
                }}
                ul, ol {{
                    margin-left: 30px;
                    margin-bottom: 1em;
                    padding-left: 0;
                }}
                li {{
                    margin-bottom: 0.6em;
                }}
                strong {{
                    font-weight: bold;
                    color: #004d40;
                }}
                em {{
                    font-style: italic;
                    color: #884400;
                }}
                pre {{
                    background-color: #f4f7f6; /* Lighter background for code blocks */
                    border: 1px solid #d4deda;
                    border-left: 4px solid #4CAF50; /* Accent border for code blocks */
                    border-radius: 6px;
                    padding: 15px;
                    font-family: 'Courier New', Courier, monospace; /* Monospaced font */
                    font-size: 0.9em;
                    overflow-x: auto; /* Enable horizontal scroll for long lines */
                    white-space: pre-wrap; /* Wrap long lines */
                    word-wrap: break-word; /* Break words if necessary */
                    margin-top: 1.5em;
                    margin-bottom: 1.5em;
                    line-height: 1.4;
                    color: #000000; /* Darker text for better contrast */
                }}
                code {{
                    font-family: 'Courier New', Courier, monospace;
                    background-color: #e8e8e8; /* Slightly darker background for inline code */
                    padding: 2px 5px;
                    border-radius: 4px;
                    font-size: 0.95em;
                    color: #c7254e; /* Distinct color for inline code */
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 1.5em;
                    margin-bottom: 1.5em;
                    font-size: 0.95em;
                }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                    color: #444;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            {markdown.markdown(markdown_text, extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])}
        </body>
        </html>
        """
        # Pass the stylesheets list to write_pdf
        HTML(string=html_content).write_pdf(output_pdf_path, stylesheets=stylesheets)
        print(f"PDF successfully created at: {output_pdf_path}")
    except Exception as e:
        print(f"Error creating PDF: {e}")
        print("Please ensure WeasyPrint and its dependencies (GTK+ Runtime on Windows) are correctly configured.")
