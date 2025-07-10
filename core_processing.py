# core_processing.py

# Core Libraries
import os
import json
import nltk
import time
import tempfile

# NEW IMPORT: For video processing (using pydub instead of moviepy)
from pydub import AudioSegment # <--- CHANGED IMPORT

# Google Gemini API
import google.generativeai as genai

# OpenAI Whisper for Transcription
import whisper
import torch

# For Robust Markdown to PDF Conversion
import markdown
from weasyprint import HTML, CSS

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

# --- NEW FUNCTION: Extract Audio from Video (using pydub) ---
def extract_audio_from_video(video_file_path):
    """
    Extracts the audio track from a video file and saves it as a temporary MP3 file.
    Uses pydub for processing.
    """
    print(f"\n--- Extracting audio from: {os.path.basename(video_file_path)} using pydub ---")
    audio_output_path = None
    try:
        # Create a temporary file path for the extracted audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio_file:
            audio_output_path = tmp_audio_file.name

        # Load the video file (pydub can read directly if ffmpeg is configured)
        audio = AudioSegment.from_file(video_file_path, format="mp4") # Adjust format if your video is not mp4
        
        # Export the audio to MP3
        audio.export(audio_output_path, format="mp3")
        
        print(f"Audio extracted successfully to: {audio_output_path}")
        return audio_output_path

    except Exception as e:
        print(f"Error extracting audio from video '{os.path.basename(video_file_path)}' using pydub: {e}")
        # Ensure temporary audio file is cleaned up if extraction fails
        if audio_output_path and os.path.exists(audio_output_path):
            os.remove(audio_output_path)
        return None

# Function to Transcribe Audio with Whisper
def transcribe_audio_whisper(audio_file_path, whisper_model_instance, progress_callback=None):
    """
    Transcribes an audio file into text using a pre-loaded Whisper model instance.
    Handles file existence checks and basic error reporting.
    Includes an optional progress_callback for UI updates.
    """
    print(f"\n--- Transcribing: {os.path.basename(audio_file_path)} ---")
    if whisper_model_instance is None:
        if progress_callback:
            progress_callback(0) # Ensure progress bar doesn't start if model failed to load
        return "Error: Whisper model not loaded. Cannot transcribe."

    try:
        if not os.path.exists(audio_file_path):
            if progress_callback:
                progress_callback(0)
            return f"Error: Audio file not found at '{audio_file_path}'"

        if os.path.getsize(audio_file_path) == 0:
            if progress_callback:
                progress_callback(0)
            return f"Error: Audio file at '{audio_file_path}' is empty. Cannot transcribe."

        if progress_callback:
            # Indicate initial progress as transcription starts
            progress_callback(10)

        # Perform the transcription
        result = whisper_model_instance.transcribe(audio_file_path)
        transcribed_text = result["text"]

        if progress_callback:
            progress_callback(100) # Indicate completion of transcription

        if not transcribed_text.strip():
            return "Whisper recognized no speech or produced empty transcription."
        else:
            return transcribed_text.strip()

    except Exception as e:
        if progress_callback:
            progress_callback(0) # Reset or indicate failure
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
        enhanced_prompt = f"""You are an elite educational AI. Your primary task is to convert the provided transcript into TWO distinct Markdown outputs: (1) Comprehensive study notes, and (2) A personalized teaching explanation. Adhere STRICTLY to all formatting rules.

                **PART 1: ENHANCED STUDY NOTES**

                **Formatting Rules - CRITICAL Adherence Required:**
                1.  **ABSOLUTE DOCUMENT START - H1 Main Title:** The entire Markdown output for Part 1 MUST begin IMMEDIATELY with a single H1 heading (e.g., `# Lecture Title`). There must be NO characters, NO blank lines, NO whitespace, NO comments, and NO other Markdown elements of any kind preceding this H1 heading. This H1 heading is the very first line of the output. It must be a standalone line and NOT wrapped in any other block-level element (like a paragraph, blockquote, or div). This is the ONLY H1 heading in the entire document.
                2.  **Section Headings (H2):** All main sections (e.g., Learning Objectives, Key Concepts, Core Content topics) MUST use H2 headings (e.g., `## 2. Key Concepts`).
                3.  **Subsection Headings (H3):** All subsections within H2 sections (e.g., specific concepts under "Core Content") MUST use H3 headings (e.g., `### 3.1. Specific Concept`).
                4.  **NO DEEPER HEADINGS (H4+):** STRICTLY DO NOT use H4 (####), H5 (#####), or any deeper heading levels for any titles, sub-titles, or structural elements.
                5.  **Formatting Sub-items within H3:** For elements like 'Why This Matters', 'Prerequisites', 'Worked Examples', individual examples, or 'Try It Yourself' points that fall under an H3 subsection, use ONLY bolded text (e.g., `**Why This Matters:** This is important because...`), bulleted lists (`* Item`), or numbered lists (`1. Item`). DO NOT use H4 or deeper headings for these. For instance, a worked example introduction should be `**Worked Example 1:** Description`, not `#### Worked Example 1`.

                **Content & Structure - Study Notes:**

                **Enhanced Content Elements for Study Notes (Reminder: Format these using bold text or lists within H3 sections, NOT H4+ headings):**

                📚 **Conceptual Framework:**
                - Within an H3 section for a major concept, start with `**Why This Matters:** ` (bolded text) followed by its context.
                - If applicable, include `**Prerequisites:** ` (bolded text) followed by the prerequisites.
                - If relevant, include `**Common Misconceptions:** ` (bolded text, perhaps in an admonition-like format if possible using Markdown blockquotes with a leading bold title) for common errors.

                🎯 **Visual Learning Aids:**
                - Create ASCII diagrams or flowcharts for processes.
                - Use tables to compare/contrast related concepts.
                - Include `**Quick Reference:** ` boxes (e.g., using admonitions or styled blockquotes) with formulas/syntax.

                💡 **Learning Enhancements:**
                - Add `**Pro Tips:** ` (bolded) from industry experience.
                - Include `**Warning/Pitfall:** ` sections (e.g., using admonitions) for common errors.
                - Create `**Memory Tricks:** ` or mnemonics for complex concepts.

                🔧 **Practical Elements:**
                - Within each major concept's H3 section, introduce worked examples with `**Worked Examples:**` (bolded text).
                - Each individual example should be clearly delineated (e.g., `**Example 1:** Name of Example` or using a list item) followed by code blocks and explanations as requested. Do NOT use H4 for individual examples.
                - Include `**Try It Yourself:** ` (bolded text) followed by mini-exercises.
                - Add `**Real Industry Scenario:** ` (bolded text) followed by case studies.

                📊 **Code & Technical Content:**
                - All code blocks must be production-ready with error handling.
                - Include multiple implementation approaches when relevant.
                - Add performance considerations and Big O notation.
                - Show both "beginner-friendly" and "optimized" versions.

                **Mandatory Sections for Study Notes (use H2 for these section titles, ensure content adheres to heading depth rules):**
                1. ## Learning Objectives (at the start)
                2. ## Key Concepts & Vocabulary (with etymology/intuition; sub-definitions under H3 if needed, or bolded terms)
                3. ## Core Content (organized by topic, using H2 for main topics and H3 for subtopics)
                4. ## Practical Applications (3+ real-world examples, use H3 for each application title)
                5. ## Common Interview Questions & Answers
                    - Identify 3-5 common interview questions that could be asked based on the core concepts of this lecture.
                    - For each question:
                    - State the question clearly (you can use H3 for each question, or a bolded list item like `* **Question:** What is...?`).
                    - Provide a concise, accurate, and comprehensive answer immediately following the question. The answer should be plain paragraph text, potentially with bullet points if listing items.
                    - Ensure the answer is directly derived from the lecture content and explains the concept thoroughly enough for an interview setting.
                    - If the lecture content does not naturally lend itself to distinct interview questions, state "No specific interview questions directly arise from this lecture's core content." under this H2 heading.
                6. ## Further Learning Resources with proper links
                7. ## Quick Review Checklist


                **PART 2: PROFESSIONAL-LEVEL PERSONALIZED TEACHING EXPLANATION**

                After the comprehensive study notes, create a separate, substantial section precisely titled:

                ## 🎓 Let Me Explain This Like We're Colleagues

                Adopt the persona of an experienced, articulate, and engaging senior colleague or mentor guiding a bright junior colleague or advanced student through the complexities of the lecture topic. The goal is deep conceptual understanding and intuition, not just surface-level recall. This section should be as valuable, if not more so, than the formal notes for true learning.

                **Core Philosophy for This Section:**
                *   **"Why before What":** Always explain the *motivation* and *purpose* behind a concept before diving into its mechanics. Why does this exist? What problem does it solve?
                *   **Intuition First, Formalism Later (If Necessary):** Build a strong intuitive grasp using analogies and simpler terms. Formal definitions or mathematical rigor should only follow if it genuinely aids deeper understanding for this audience, and should still be explained clearly.
                *   **Connect the Dots:** Explicitly show how different concepts within the lecture (and potentially related prior knowledge) link together to form a cohesive picture.
                *   **Beyond the Obvious:** Offer insights that go beyond textbook definitions, perhaps touching upon historical context (briefly, if relevant), common pitfalls in application, or subtle nuances often missed by beginners.

                **Structure and Content Elements for Each Major Concept/Topic Covered:**

                1.  **The "Big Idea" & Its Significance (The "Why"):**
                    *   Start with a compelling hook or question related to the concept.
                    *   Clearly state the core problem or need the concept addresses.
                    *   Explain its importance in the broader field or for specific applications (e.g., "Why is understanding operator precedence non-negotiable for any serious Python developer?").

                2.  **Intuitive Explanation(s) (The "How it Works, Simply"):**
                    *   **Masterful Analogies (Minimum 2 per key idea):** Go beyond simple comparisons. Develop rich, well-explained analogies from diverse domains (technology, nature, history, everyday life, complex systems like city planning or an orchestra) that illuminate the core mechanics and relationships. Explain *how* the analogy maps to the concept.
                    *   **Storytelling/Scenario-Based Walkthroughs:** For processes or algorithms, narrate a step-by-step scenario as if solving a real, relatable problem. Show the concept in action.
                    *   **Multiple Angles of Attack:** Explain the same core idea from at least two different perspectives or using different mental models to cater to varied learning styles.

                3.  **Key Distinctions & Nuances (The "Watch Out For This"):**
                    *   **Clarify Common Confusion Points:** Proactively address areas where students typically get stuck or misunderstand. ("A frequent point of confusion is X, but think of it this way...")
                    *   **Subtle but Crucial Differences:** If there are closely related concepts, spend time clearly differentiating them (e.g., "equality `==` vs. identity `is` in Python – subtle but critical!").
                    *   **Edge Cases & Limitations (Briefly):** Hint at situations where the concept might break down or have specific limitations, fostering critical thinking.

                4.  **Bridging to Practicality (The "So What? How is this Used?"):**
                    *   **Illustrative Mini-Examples (Code or Pseudocode if applicable):** Provide concise, heavily-commented code snippets or clear pseudocode that demonstrate the concept in a simplified, practical context. Focus on clarity of the concept, not complex application logic.
                    *   **Real-World Implications (Beyond the Obvious):** Briefly touch upon how this concept underpins more advanced topics or enables significant real-world technologies or solutions.

                5.  **"Aha!" Moment Synthesis (The "Putting It All Together"):**
                    *   Conclude the explanation of each major concept with a summary that crystallizes the main takeaway or the "aha!" insight.
                    *   Use reflective questions like, "So, what's the golden rule here?" or "What's the one thing you absolutely must remember about X?"

                **Teaching Style Guidelines for This Section:**
                *   **Professional yet Conversational:** Maintain an intelligent, articulate tone suitable for an advanced learner ("you," "we," "let's explore," "consider this scenario"). Avoid overly simplistic or patronizing language.
                *   **Clarity and Precision:** While using analogies, ensure the underlying technical points are communicated accurately.
                *   **Enthusiasm and Engagement:** Convey a genuine passion for the subject matter.
                *   **Strategic Use of Formatting:** Use bolding for emphasis on key terms or insights. Use bullet points or numbered lists for clarity where appropriate within explanations.
                *   **Minimal Emojis:** Use emojis *very* sparingly, only if they add significant, professional-level emphasis or a touch of lightheartedness where appropriate (e.g., a single 💡 for a key insight). The focus is on rich textual explanation.


                **Example of Desired Depth and Tone (Expanding on Previous Example):**

                *Instead of just the ice cream analogy for linear regression, consider this structure:*

                "Alright, let's talk about linear regression. **Why does this even exist?** Imagine you're a data scientist at a streaming service. Your boss wants to know: if we spend more on marketing a new show, will we actually get significantly more new subscribers? You've got data on past shows – marketing spend and new sign-ups. Linear regression is essentially your smart tool for trying to draw the most honest, representative straight line through that scattered data.

                **So, how does it 'find' this line?**
                *   **Analogy 1: The 'Fairest Referee'.** Think of each data point (a show's marketing spend and sign-ups) as a player in a game. The regression line is like a referee trying to position themselves on the field such that they are, on average, as close as possible to all players simultaneously. It's not going to be perfectly next to every player, but it's the 'least wrong' position overall. Mathematically, it's minimizing the sum of the squared distances (the 'errors') from each point to the line. Why squared? It penalizes big misses more heavily and makes the math work out nicely.
                *   **Analogy 2: The 'Economic Forecaster's Trend Line'.** You've seen those charts of stock prices or economic indicators with a line cutting through them? That's often a regression line (or a more complex version). It's trying to capture the underlying trend, smoothing out the daily noise. It helps answer: 'Generally, is this going up or down, and how steeply?'

                **A common point of confusion:** People sometimes think linear regression *predicts the future with certainty*. Not quite. It gives you the *best linear estimate* based on past data. The real world has more variables and randomness. So, the line helps us understand the *relationship* and make *informed estimations*, but it's not a crystal ball. Another nuance is that 'linear' doesn't just mean a straight line in raw X vs. Y; you can transform your variables (e.g., log(X)) and still fit a linear model to that transformed relationship.

                **Practically, in Python (with Scikit-learn, for instance),** you'd feed it your `marketing_spend` (X) and `new_signups` (Y) data. It then calculates the slope (how many more sign-ups you get for each extra dollar spent, on average) and the intercept (the baseline sign-ups if marketing spend was zero).

                **The 'Aha!' moment:** Linear regression isn't just about drawing a line; it's about quantifying a linear relationship between variables in the presence of noise, allowing us to understand trends and make predictions, however imperfect. It's a foundational tool for understanding how one thing influences another."


                **Additional Focus for Both Parts (Study Notes & Teaching Explanation):**
                {note_style_prompt if note_style_prompt else "Focus on making complex concepts accessible to beginners while maintaining technical accuracy."}

                **Transcript:**
                ---
                {text_input}
                ---

                **Final Instruction:** Generate both Part 1 and Part 2 with clear separation. The Study Notes (Part 1) MUST adhere to all specified formatting rules, especially the H1, H2, H3 heading structure and the absolute start of the document with the H1 title. The teaching explanation (Part 2) should follow its own distinct style guide.

        """

        model = genai.GenerativeModel('gemini-2.5-flash')
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
