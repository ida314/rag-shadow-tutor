import streamlit as st
import os
import sys

from src.pipeline.pipeline import Pipeline, LanguageNotAvailableError
from src.pipeline.yt_fetch import YTFetch
import re
from urllib.parse import urlparse, parse_qs
from openai import OpenAI

import tempfile
from typing import Optional

# Comprehensive language mapping from full names to ISO 639-1 codes
LANGUAGE_MAPPING = {
    # Major European Languages
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Polish": "pl",
    "Dutch": "nl",
    "Greek": "el",
    
    # Asian Languages
    "Chinese (Simplified)": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Arabic": "ar",
    "Hebrew": "he",
    "Turkish": "tr",
    "Thai": "th",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Malay": "ms",
    "Tagalog": "tl",
    "Bengali": "bn",
    "Urdu": "ur",
    "Persian": "fa",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Georgian": "ka",
    
    # Nordic Languages
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da",
    "Finnish": "fi",
    
    # Eastern European Languages
    "Czech": "cs",
    "Slovak": "sk",
    "Ukrainian": "uk",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Serbian": "sr",
    "Slovenian": "sl",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Estonian": "et",
    "Albanian": "sq",
    "Macedonian": "mk",
    
    # Other Languages
    "Swahili": "sw",
    "Catalan": "ca",
    "Basque": "eu",
    "Galician": "gl"
}

# Reverse mapping for displaying language names from codes
LANGUAGE_CODE_TO_NAME = {v: k for k, v in LANGUAGE_MAPPING.items()}

# Group languages by region for better UX
LANGUAGE_GROUPS = {
    "Popular": ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese (Simplified)", "Japanese", "Korean", "Arabic", "Hindi", "Russian"],
    "European": ["Dutch", "Greek", "Polish", "Swedish", "Norwegian", "Danish", "Finnish", "Czech", "Romanian", "Hungarian", "Turkish"],
    "Asian": ["Thai", "Vietnamese", "Indonesian", "Malay", "Tagalog", "Bengali", "Urdu", "Persian", "Tamil", "Telugu", "Hebrew"],
    "Eastern European": ["Ukrainian", "Bulgarian", "Croatian", "Serbian", "Slovak", "Slovenian", "Lithuanian", "Latvian", "Estonian", "Albanian", "Macedonian"],
    "Other": ["Swahili", "Catalan", "Basque", "Galician", "Gujarati", "Marathi", "Mongolian", "Georgian"]
}

# Flatten all languages for the selectbox
ALL_LANGUAGES = []
for group_languages in LANGUAGE_GROUPS.values():
    ALL_LANGUAGES.extend(group_languages)
# Remove duplicates while preserving order
ALL_LANGUAGES = list(dict.fromkeys(ALL_LANGUAGES))

# Page config
st.set_page_config(
    page_title="Language Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state - must happen before any other st calls
if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = Pipeline()
if 'lesson_data' not in st.session_state:
    st.session_state['lesson_data'] = None
if 'video_id' not in st.session_state:
    st.session_state['video_id'] = None
if 'client' not in st.session_state:
    st.session_state['client'] = None
if 'available_languages' not in st.session_state:
    st.session_state['available_languages'] = None
if 'current_url' not in st.session_state:
    st.session_state['current_url'] = None


def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    try:
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        parsed = urlparse(url)
        if parsed.hostname in ["youtu.be"]:
            return parsed.path.lstrip("/")
        elif parsed.hostname in ["www.youtube.com", "youtube.com"]:
            query = parse_qs(parsed.query)
            return query.get("v", [None])[0]
    except:
        return None
    return None


def is_youtube_url(text: str) -> bool:
    """Check if text is a YouTube URL."""
    youtube_domains = ["youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"]
    try:
        parsed = urlparse(text)
        return any(domain in parsed.netloc for domain in youtube_domains)
    except:
        return False


def check_and_display_languages(url: str) -> bool:
    """
    Check available languages for a YouTube video and display them.
    Returns True if check was successful, False otherwise.
    """
    try:
        # Check if we need to refresh the language list
        if st.session_state['current_url'] != url:
            is_available, languages = st.session_state['pipeline'].check_language_availability(url, "dummy")
            st.session_state['available_languages'] = languages
            st.session_state['current_url'] = url
        
        if st.session_state['available_languages']:
            # Create a formatted list of available languages
            lang_info = []
            for lang in st.session_state['available_languages']:
                lang_code = lang.get('language_code', 'unknown')
                lang_name = lang.get('language', 'Unknown')
                is_generated = lang.get('is_generated', False)
                
                # Try to get the display name from our mapping
                display_name = LANGUAGE_CODE_TO_NAME.get(lang_code, lang_name)
                
                if is_generated:
                    lang_info.append(f"{display_name} ({lang_code}) - Auto-generated")
                else:
                    lang_info.append(f"{display_name} ({lang_code})")
            
            # Display available languages in an expander
            with st.expander("📋 Available languages for this video", expanded=False):
                st.markdown("**Available transcripts:**")
                for info in lang_info:
                    st.markdown(f"• {info}")
            
            return True
        return False
    except Exception as e:
        st.error(f"Error checking available languages: {str(e)}")
        return False


def get_pronunciation_feedback(original: str, transcribed: str, level: str, language_code: str) -> str:
    """Get feedback on pronunciation using GPT."""
    try:
        # Get language name for the prompt
        language_name = LANGUAGE_CODE_TO_NAME.get(language_code, "the target language")
        
        prompt = f"""
        As a language coach for {language_name}, compare what the student said to the original text.
        Be encouraging and specific. Consider the pronunciation challenges specific to {language_name}.
        
        Original: "{original[:150]}"
        Student said: "{transcribed}"
        Level: {level}
        
        In 2-3 sentences, mention what they did well and one area to improve.
        """

        response = st.session_state['client'].chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        return response.choices[0].message.content
    except:
        return "Good effort! Keep practicing to improve your pronunciation."


def transcribe_audio(audio_bytes: bytes, language_code: str) -> str:
    """Transcribe audio using Whisper API."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        with open(tmp_path, "rb") as audio_file:
            response = st.session_state['client'].audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language_code  # Hint Whisper about the expected language
            )

        os.unlink(tmp_path)
        return response.text
    except Exception as e:
        return f"Error transcribing: {str(e)}"


# Header
st.title("🎓 Language Learning with YouTube")
st.markdown("Transform YouTube videos into personalized language lessons in 50+ languages!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        if not st.session_state['client']:
            st.session_state['client'] = OpenAI(api_key=api_key)

    st.markdown("---")
    st.markdown("### 📚 How it works:")
    st.markdown("1. Paste a YouTube URL")
    st.markdown("2. Choose your language & level")
    st.markdown("3. Get simplified lessons")
    st.markdown("4. Practice with Shadow Mode!")
    
    st.markdown("---")
    st.markdown("### 🌍 Supported Languages:")
    st.markdown(f"**{len(LANGUAGE_MAPPING)}** languages available!")
    
    # Show language groups
    with st.expander("View all languages"):
        for group, languages in LANGUAGE_GROUPS.items():
            st.markdown(f"**{group}:**")
            st.markdown(", ".join(languages[:5]) + ("..." if len(languages) > 5 else ""))

# Main content area
if not api_key:
    st.warning("👈 Please enter your OpenAI API key in the sidebar to get started.")
else:
    # Input section
    col1, col2 = st.columns([3, 1])

    with col1:
        url_input = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=..."
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        # Check languages button
        if url_input and is_youtube_url(url_input):
            check_langs_btn = st.button("🔍 Check Languages", use_container_width=True)
            if check_langs_btn:
                check_and_display_languages(url_input)

    # Show available languages if we have a valid URL
    if url_input and is_youtube_url(url_input):
        check_and_display_languages(url_input)

    # Language and level selection
    col3, col4, col5 = st.columns(3)

    with col3:
        # Advanced language selector with search
        language = st.selectbox(
            "Language",
            ALL_LANGUAGES,
            index=0,  # Default to English
            help="Choose the language of the video (50+ languages supported!)",
            placeholder="Search for a language..."
        )

    with col4:
        topic = st.text_input(
            "Topic (optional)",
            placeholder="e.g., cooking, science, travel",
            help="Helps find the most relevant parts"
        )

    with col5:
        level = st.selectbox(
            "Your Level",
            ["A2", "B1", "B2"],
            index=1,
            help="A2: Elementary, B1: Intermediate, B2: Upper-intermediate"
        )

    # Generate lesson button
    generate_btn = st.button("🎯 Generate Lesson", type="primary", use_container_width=True)

    # Generate lesson
    if generate_btn and url_input:
        if not is_youtube_url(url_input):
            st.error("Please enter a valid YouTube URL")
        else:
            video_id = extract_video_id(url_input)
            if not video_id:
                st.error("Could not extract video ID from URL")
            else:
                with st.spinner(f"🔄 Creating your personalized lesson in {language}..."):
                    try:
                        # Get language code
                        language_code = LANGUAGE_MAPPING.get(language, "en")
                        
                        # Use the pipeline to generate lesson
                        print(f"Generating lesson for: {language} ({language_code})")
                        results = st.session_state['pipeline'].generate_simplified_lesson(
                            url=url_input,
                            language=language_code,
                            topic=topic or language,  # Use language as fallback topic
                            level=level,
                            n_chunks=3
                        )

                        if results:
                            st.session_state['lesson_data'] = results
                            st.session_state['video_id'] = video_id
                            st.session_state['current_language'] = language_code
                            st.success(f"✅ Lesson generated successfully in {language}!")
                        else:
                            st.error("Could not generate lesson. Please try another video.")

                    except LanguageNotAvailableError as e:
                        st.error("🚫 " + str(e))
                        
                        # Show available languages in a nice format
                        if e.available_languages:
                            st.info("💡 **Tip:** This video has transcripts in the following languages:")
                            
                            # Group available languages
                            manual_langs = []
                            auto_langs = []
                            
                            for lang in e.available_languages:
                                lang_code = lang.get('language_code', 'unknown')
                                lang_name = lang.get('language', 'Unknown')
                                display_name = LANGUAGE_CODE_TO_NAME.get(lang_code, lang_name)
                                
                                if lang.get('is_generated', False):
                                    auto_langs.append(f"{display_name} ({lang_code})")
                                else:
                                    manual_langs.append(f"{display_name} ({lang_code})")
                            
                            if manual_langs:
                                st.markdown("**Manual transcripts (recommended):**")
                                st.markdown(", ".join(manual_langs))
                            
                            if auto_langs:
                                st.markdown("**Auto-generated transcripts:**")
                                st.markdown(", ".join(auto_langs))
                            
                            st.markdown("---")
                            st.markdown("**What you can do:**")
                            st.markdown("1. Select one of the available languages above")
                            st.markdown("2. Try a different YouTube video")
                            st.markdown("3. Use YouTube's auto-translate feature if available")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Display lesson content (rest of the code remains the same)
    if 'lesson_data' in st.session_state and st.session_state['lesson_data'] is not None and 'video_id' in st.session_state and st.session_state['video_id'] is not None:
        st.markdown("---")

        # Two column layout
        lesson_col, shadow_col = st.columns([2, 1])

        with lesson_col:
            # Video player
            st.markdown("### 🎥 Video")
            st.markdown(
                f'<iframe width="100%" height="315" '
                f'src="https://www.youtube.com/embed/{st.session_state["video_id"]}" '
                f'frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

            # Lesson content
            current_lang = st.session_state.get('current_language', 'en')
            lang_name = LANGUAGE_CODE_TO_NAME.get(current_lang, "Unknown")
            st.markdown(f"### 📚 Your Lesson ({lang_name})")

            for i, chunk in enumerate(st.session_state['lesson_data'], 1):
                with st.expander(f"Lesson Part {i}", expanded=(i == 1)):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("**Original Text**")
                        st.info(chunk.get('original', ''))
                        st.caption(f"Words: {len(chunk.get('original', '').split())}")

                    with col_b:
                        st.markdown(f"**Simplified ({level})**")
                        st.success(chunk.get('rewritten', ''))
                        st.caption(f"Words: {chunk.get('word_count', 0)}")

        with shadow_col:
            st.markdown("### 🎤 Shadow Mode")
            st.markdown("Practice by repeating the simplified text!")

            # Select which chunk to practice
            chunk_to_practice = st.selectbox(
                "Choose part to practice",
                options=range(len(st.session_state['lesson_data'])),
                format_func=lambda x: f"Part {x + 1}"
            )

            # Display the text to practice
            practice_text = st.session_state['lesson_data'][chunk_to_practice]['rewritten']
            st.info(practice_text)

            # Audio recorder
            st.markdown("**Record yourself:**")
            audio = st.audio_input("Click to start recording")

            if audio:
                with st.spinner("Analyzing your pronunciation..."):
                    audio_bytes = audio.getvalue()
                    current_lang = st.session_state.get('current_language', 'en')
                    
                    # Transcribe with language hint
                    user_text = transcribe_audio(audio_bytes, current_lang)

                    # Show results
                    st.markdown("**You said:**")
                    st.warning(user_text)

                    # Get feedback
                    feedback = get_pronunciation_feedback(
                        practice_text,
                        user_text,
                        level,
                        current_lang
                    )

                    st.markdown("**Feedback:**")
                    st.success(feedback)

            # Tips
            with st.expander("💡 Practice Tips"):
                st.markdown("""
                - Listen to the video section first
                - Read the simplified text aloud
                - Record yourself
                - Compare and try again!
                - Focus on one part at a time
                """)

# Footer
st.markdown("---")
st.caption(f"Made with ❤️ using Streamlit, YouTube Transcript API, and OpenAI | Supporting {len(LANGUAGE_MAPPING)} languages!")