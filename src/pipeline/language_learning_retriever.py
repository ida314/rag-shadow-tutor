#src/pipeline/language_learning_retriever.py
import os
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage

LANGUAGE_INSTRUCTION_MAP = {
    # English
    "en": "Write your entire response in English only.",
    # Spanish
    "es": "Escribe toda tu respuesta únicamente en español.",
    # French
    "fr": "Écrivez votre réponse entière uniquement en français.",
    # German
    "de": "Schreiben Sie Ihre gesamte Antwort nur auf Deutsch.",
    # Italian
    "it": "Scrivi la tua intera risposta solo in italiano.",
    # Portuguese
    "pt": "Escreva sua resposta inteira apenas em português.",
    # Chinese (Simplified)
    "zh": "请用中文写出你的完整回答。",
    # Japanese
    "ja": "回答は全て日本語で書いてください。",
    # Korean
    "ko": "답변은 모두 한국어로만 작성해 주세요.",
    # Russian
    "ru": "Напишите весь ваш ответ только на русском языке.",
    # Arabic
    "ar": "اكتب إجابتك الكاملة باللغة العربية فقط.",
    # Hindi
    "hi": "अपना पूरा उत्तर केवल हिंदी में लिखें।",
    # Dutch
    "nl": "Schrijf je volledige antwoord alleen in het Nederlands.",
    # Swedish
    "sv": "Skriv hela ditt svar endast på svenska.",
    # Polish
    "pl": "Napisz całą swoją odpowiedź tylko po polsku.",
    # Turkish
    "tr": "Yanıtınızın tamamını yalnızca Türkçe yazın.",
    # Greek
    "el": "Γράψτε ολόκληρη την απάντησή σας μόνο στα ελληνικά.",
    # Hebrew
    "he": "כתוב את כל התשובה שלך בעברית בלבד.",
    # Thai
    "th": "เขียนคำตอบทั้งหมดของคุณเป็นภาษาไทยเท่านั้น",
    # Vietnamese
    "vi": "Viết toàn bộ câu trả lời của bạn chỉ bằng tiếng Việt.",
    # Indonesian
    "id": "Tulis seluruh jawaban Anda hanya dalam bahasa Indonesia.",
    # Czech
    "cs": "Napište celou svou odpověď pouze v češtině.",
    # Danish
    "da": "Skriv hele dit svar kun på dansk.",
    # Finnish
    "fi": "Kirjoita koko vastauksesi vain suomeksi.",
    # Norwegian
    "no": "Skriv hele svaret ditt kun på norsk.",
    # Ukrainian
    "uk": "Напишіть всю вашу відповідь лише українською мовою.",
    # Romanian
    "ro": "Scrie întregul tău răspuns doar în limba română.",
    # Hungarian
    "hu": "Írja meg teljes válaszát csak magyarul.",
    # Bengali
    "bn": "আপনার সম্পূর্ণ উত্তর শুধুমাত্র বাংলায় লিখুন।",
    # Tagalog/Filipino
    "tl": "Isulat ang iyong buong sagot sa Tagalog lamang.",
    # Malay
    "ms": "Tulis keseluruhan jawapan anda dalam bahasa Melayu sahaja.",
    # Swahili
    "sw": "Andika jibu lako lote kwa Kiswahili pekee.",
    # Persian/Farsi
    "fa": "پاسخ کامل خود را فقط به فارسی بنویسید.",
    # Urdu
    "ur": "اپنا پورا جواب صرف اردو میں لکھیں۔",
    # Tamil
    "ta": "உங்கள் முழு பதிலையும் தமிழில் மட்டுமே எழுதுங்கள்.",
    # Gujarati
    "gu": "તમારો સંપૂર્ણ જવાબ ફક્ત ગુજરાતીમાં લખો.",
    # Marathi
    "mr": "तुमचे संपूर्ण उत्तर फक्त मराठीत लिहा.",
    # Telugu
    "te": "మీ పూర్తి సమాధానాన్ని తెలుగులో మాత్రమే వ్రాయండి.",
    # Bulgarian
    "bg": "Напишете целия си отговор само на български език.",
    # Croatian
    "hr": "Napišite cijeli svoj odgovor samo na hrvatskom jeziku.",
    # Serbian
    "sr": "Напишите цео свој одговор само на српском језику.",
    # Slovak
    "sk": "Napíšte celú svoju odpoveď iba v slovenčine.",
    # Slovenian
    "sl": "Napišite celoten odgovor samo v slovenščini.",
    # Lithuanian
    "lt": "Parašykite visą savo atsakymą tik lietuvių kalba.",
    # Latvian
    "lv": "Rakstiet visu savu atbildi tikai latviešu valodā.",
    # Estonian
    "et": "Kirjutage kogu oma vastus ainult eesti keeles.",
    # Albanian
    "sq": "Shkruani të gjithë përgjigjen tuaj vetëm në shqip.",
    # Macedonian
    "mk": "Напишете го целиот ваш одговор само на македонски јазик.",
    # Mongolian
    "mn": "Хариултаа бүхэлд нь зөвхөн монгол хэлээр бичнэ үү.",
    # Georgian
    "ka": "დაწერეთ თქვენი სრული პასუხი მხოლოდ ქართულად.",
    # Catalan
    "ca": "Escriu tota la teva resposta només en català.",
    # Basque
    "eu": "Idatzi zure erantzun osoa euskaraz soilik.",
    # Galician
    "gl": "Escribe toda a túa resposta só en galego."
}


class LanguageLearningRetriever:
    """
    Retrieves and rewrites content for language learners based on topic and CEFR level.
    """
    
    language_instructions = {
        "Spanish": "Escribe la respuesta completamente en español.",
        "French": "Écris la réponse entièrement en français.",
        # etc...
    }
    
    def __init__(self, embedding_model: str = "text-embedding-3-small", llm_model: str = "gpt-3.5-turbo"):
        """
        Initialize the retriever with embeddings and language model.
        
        Args:
            embedding_model: OpenAI embedding model to use
            llm_model: OpenAI language model for rewriting
        """
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0.3)
        self.vectorstore = None
        
        # CEFR level descriptions for prompts
        self.cefr_descriptions = {
            "A2": "elementary level with simple vocabulary and short sentences",
            "B1": "intermediate level with everyday vocabulary and clear structure",
            "B2": "upper-intermediate level with varied vocabulary and complex sentences"
        }
    
    def add_content(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add educational content to the vector store.
        
        Args:
            texts: List of text chunks
            metadatas: Optional metadata (language, topic, source, etc.)
        """
        if self.vectorstore is None:
            self.vectorstore = InMemoryVectorStore.from_texts(
                texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
        else:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
    
    def search_and_rewrite(self, language: str, topic: str, cefr_level: str, top_k: Optional[int] = 3) -> List[Dict]:
        """
        Search for content and rewrite it for the specified CEFR level.
        
        Args:
            language: Target language (e.g., "Spanish", "French")
            topic: Topic to search for
            cefr_level: Target CEFR level (A2, B1, or B2)
            
        Returns:
            List of dictionaries with original and rewritten content
        """
        if self.vectorstore is None:
            raise ValueError("No content has been added to the store yet")
        
        if cefr_level not in self.cefr_descriptions:
            raise ValueError(f"Invalid CEFR level. Must be one of: {list(self.cefr_descriptions.keys())}")
        
        # Create search query
        query = f"{language} {topic}"
        
        # TODO: check if the topic is in a different language to the embedding, and translate it to match the language stored in the embeddings
        
        # Retrieve top 3 chunks
        results = self.vectorstore.similarity_search(query, k=top_k)
        
        # Rewrite each chunk for the target CEFR level
        rewritten_results = []
        
        for i, doc in enumerate(results):
            # Create rewriting prompt
            prompt = self._create_rewrite_prompt(doc.page_content, cefr_level, language)
            
            # Get rewritten content
            response = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten_text = response.content.strip()
            
            # Store result
            rewritten_results.append({
                "original": doc.page_content,
                "rewritten": rewritten_text,
                "metadata": doc.metadata,
                "cefr_level": cefr_level,
                "word_count": len(rewritten_text.split())
            })
        
        return rewritten_results
    
    def _create_rewrite_prompt(self, text: str, cefr_level: str, language: str) -> str:
        """
        Create a prompt for rewriting content at the specified CEFR level.
        
        Args:
            text: Original text to rewrite
            cefr_level: Target CEFR level
            
        Returns:
            Formatted prompt for the LLM
        """
        level_description = self.cefr_descriptions[cefr_level]
        
        prompt = f"""Rewrite the following text for a {cefr_level} language learner ({level_description}).

{LANGUAGE_INSTRUCTION_MAP.get(f"{language}", "")}

Requirements:
- Target level: {cefr_level}
- Maximum length: 120 words
- Keep the original meaning
- Use natural, conversational language
- Make it engaging and easy to understand

Original text:
{text}

Rewritten text:"""
        
        return prompt

