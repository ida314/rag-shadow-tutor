from typing import List, Dict, Optional, Tuple
import logging
import traceback
from src.pipeline.yt_fetch import YTFetch
from src.pipeline.chunker import Chunker
from src.pipeline.language_learning_retriever import LanguageLearningRetriever

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass


class YouTubeFetchError(PipelineError):
    """Raised when YouTube fetching/transcription fails"""
    pass


class LanguageNotAvailableError(YouTubeFetchError):
    """Raised when requested language is not available for the video"""
    def __init__(self, message: str, available_languages: List[Dict[str, any]] = None):
        super().__init__(message)
        self.available_languages = available_languages or []


class ChunkingError(PipelineError):
    """Raised when text chunking fails"""
    pass


class RetrievalError(PipelineError):
    """Raised when retrieval/rewriting fails"""
    pass


class Pipeline:
    def __init__(self):
        self.yt_fetch = None
        self.chunker = None
        self.retriever = None
        logger.info("Pipeline initialized")

    def _validate_inputs(self, url: str, language: str, topic: str, level: str, n_chunks: int) -> None:
        """Validate input parameters"""
        if not url or not isinstance(url, str):
            raise ValueError(f"Invalid URL provided: {url}")
        
        if not language or not isinstance(language, str):
            raise ValueError(f"Invalid language provided: {language}")
        
        if not topic or not isinstance(topic, str):
            raise ValueError(f"Invalid topic provided: {topic}")
        
        valid_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        if level not in valid_levels:
            raise ValueError(f"Invalid CEFR level: {level}. Must be one of {valid_levels}")
        
        if n_chunks <= 0 or not isinstance(n_chunks, int):
            raise ValueError(f"Invalid n_chunks: {n_chunks}. Must be a positive integer")

    def _initialize_components(self) -> None:
        """Initialize pipeline components with error handling"""
        try:
            if not self.yt_fetch:
                logger.info("Initializing YTFetch component")
                self.yt_fetch = YTFetch()
                logger.info("YTFetch component initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YTFetch: {str(e)}")
            raise PipelineError(f"YTFetch initialization failed: {str(e)}") from e

        try:
            if not self.chunker:
                logger.info("Initializing Chunker component")
                self.chunker = Chunker()
                logger.info("Chunker component initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Chunker: {str(e)}")
            raise PipelineError(f"Chunker initialization failed: {str(e)}") from e

        try:
            if not self.retriever:
                logger.info("Initializing LanguageLearningRetriever component")
                self.retriever = LanguageLearningRetriever()
                logger.info("LanguageLearningRetriever component initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LanguageLearningRetriever: {str(e)}")
            raise PipelineError(f"LanguageLearningRetriever initialization failed: {str(e)}") from e

    def check_language_availability(self, url: str, target_language: str) -> Tuple[bool, List[Dict[str, any]]]:
        """
        Check if the target language is available for the given YouTube video.
        
        Args:
            url: YouTube video URL
            target_language: Language code to check (e.g., 'en', 'es', 'fr')
            
        Returns:
            Tuple of (is_available, available_languages_list)
        """
        try:
            # Ensure YTFetch is initialized
            if not self.yt_fetch:
                self.yt_fetch = YTFetch()
            
            # Get available languages
            available_languages = self.yt_fetch.get_available_languages(url)
            
            # Check if target language is available
            is_available = any(
                lang.get('language_code') == target_language 
                for lang in available_languages
            )
            
            return is_available, available_languages
            
        except Exception as e:
            logger.error(f"Error checking language availability: {str(e)}")
            # Return empty list if we can't check
            return False, []

    def generate_simplified_lesson(
        self,
        url: str,
        language: str,
        topic: str,
        level: str = "B1",
        n_chunks: int = 3
    ) -> List[Dict[str, str]]:
        """
        Given a YouTube URL and learner profile, return simplified lesson chunks.
        
        Args:
            url: YouTube video URL
            language: Target language for transcription
            topic: Topic for content filtering
            level: CEFR level (A1-C2)
            n_chunks: Number of chunks to return
            
        Returns:
            List of dictionaries containing:
            - original: original transcript text
            - simplified: rewritten CEFR-level version
            - start_time: video timestamp
            - duration: chunk duration
            
        Raises:
            ValueError: If input parameters are invalid
            LanguageNotAvailableError: If requested language is not available
            YouTubeFetchError: If video fetching/transcription fails
            ChunkingError: If text chunking fails
            RetrievalError: If retrieval/rewriting fails
            PipelineError: For general pipeline errors
        """
        logger.info(f"Starting pipeline for URL: {url}, language: {language}, topic: {topic}, level: {level}")
        
        # Validate inputs
        try:
            self._validate_inputs(url, language, topic, level, n_chunks)
        except ValueError as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise

        # Initialize components
        try:
            self._initialize_components()
        except PipelineError as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise

        # Check language availability BEFORE any expensive operations
        logger.info(f"Checking language availability for {language}")
        is_available, available_languages = self.check_language_availability(url, language)
        
        if not is_available:
            # Construct helpful error message
            available_codes = [lang.get('language_code', 'unknown') for lang in available_languages]
            available_names = [lang.get('language', 'Unknown') for lang in available_languages]
            
            error_msg = f"The requested language '{language}' is not available for this video. "
            error_msg += f"Available languages: {', '.join(f'{name} ({code})' for name, code in zip(available_names, available_codes))}"
            
            logger.warning(error_msg)
            raise LanguageNotAvailableError(error_msg, available_languages)

        # Fetch and transcribe YouTube video
        transcribed = None
        try:
            logger.info(f"Fetching and transcribing video from URL: {url}")
            transcribed = self.yt_fetch.transcribe(
                url=url, 
                target_language=language, 
                format_as_text=True
            )
            
            if not transcribed:
                raise YouTubeFetchError("Transcription returned empty result")
                
            logger.info(f"Successfully transcribed video. Text length: {len(transcribed)} characters")
            
        except AttributeError as e:
            logger.error(f"YTFetch method error: {str(e)}")
            raise YouTubeFetchError(f"YTFetch transcribe method failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to fetch/transcribe YouTube video: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise YouTubeFetchError(
                f"Failed to fetch/transcribe video from {url}: {str(e)}"
            ) from e

        # Chunk the transcribed text
        chunks = None
        try:
            logger.info("Starting text chunking")
            chunks = self.chunker.chunk(input=transcribed)
            
            if not chunks:
                raise ChunkingError("Chunker returned empty result")
                
            logger.info(f"Successfully created {len(chunks)} chunks")
            
        except AttributeError as e:
            logger.error(f"Chunker method error: {str(e)}")
            raise ChunkingError(f"Chunker chunk method failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise ChunkingError(f"Failed to chunk transcribed text: {str(e)}") from e

        # Extract text content and add to retriever
        try:
            logger.info("Extracting text content from chunks")
            texts = [c.content for c in chunks]
            
            if not texts:
                raise RetrievalError("No text content found in chunks")
                
            logger.info(f"Adding {len(texts)} texts to retriever")
            self.retriever.add_content(texts=texts)
            
        except AttributeError as e:
            logger.error(f"Chunk content extraction error: {str(e)}")
            raise RetrievalError(
                f"Failed to extract content from chunks. Chunks may not have 'content' attribute: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to add content to retriever: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RetrievalError(f"Failed to add content to retriever: {str(e)}") from e

        # Search and rewrite content
        try:
            logger.info(f"Searching and rewriting content for topic: {topic}, level: {level}")
            results = self.retriever.search_and_rewrite(
                language=language,
                topic=topic,
                cefr_level=level,
                top_k=n_chunks
            )
            
            if not results:
                logger.warning("No results returned from search_and_rewrite")
                return []
                
            logger.info(f"Successfully generated {len(results)} simplified lessons")
            
            # Validate result format
            for i, result in enumerate(results):
                if not isinstance(result, dict):
                    raise RetrievalError(f"Result {i} is not a dictionary: {type(result)}")
                    
                required_fields = ["original", "rewritten"]
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    logger.warning(
                        f"Result {i} missing fields: {missing_fields}. "
                        f"Available fields: {list(result.keys())}"
                    )
            
            return results
            
        except AttributeError as e:
            logger.error(f"Retriever method error: {str(e)}")
            raise RetrievalError(f"Retriever search_and_rewrite method failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to search and rewrite content: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RetrievalError(f"Failed to search and rewrite content: {str(e)}") from e

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self.retriever:
                logger.info("Cleaning up retriever resources")
                # Add any cleanup methods if available
                pass
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Example usage with error handling
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    pipeline = Pipeline()
    
    try:
        # First check language availability
        url = "https://youtube.com/watch?v=example"
        target_lang = "es"
        
        is_available, langs = pipeline.check_language_availability(url, target_lang)
        if not is_available:
            print(f"Language {target_lang} not available. Available languages:")
            for lang in langs:
                print(f"  - {lang['language']} ({lang['language_code']})")
        else:
            results = pipeline.generate_simplified_lesson(
                url=url,
                language=target_lang,
                topic="technology",
                level="B1",
                n_chunks=3
            )
            
            for result in results:
                print(f"Original: {result.get('original', 'N/A')[:100]}...")
                print(f"Simplified: {result.get('rewritten', 'N/A')[:100]}...")
                print("-" * 80)
            
    except ValueError as e:
        print(f"Invalid input: {e}")
    except LanguageNotAvailableError as e:
        print(f"Language not available: {e}")
        if e.available_languages:
            print("Available languages:")
            for lang in e.available_languages:
                print(f"  - {lang['language']} ({lang['language_code']})")
    except YouTubeFetchError as e:
        print(f"YouTube fetch error: {e}")
    except ChunkingError as e:
        print(f"Chunking error: {e}")
    except RetrievalError as e:
        print(f"Retrieval error: {e}")
    except PipelineError as e:
        print(f"Pipeline error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        logging.exception("Unexpected error in pipeline")
    finally:
        pipeline.cleanup()