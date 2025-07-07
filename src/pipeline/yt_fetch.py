from typing import Optional, List, Dict, Any, Union
from urllib.parse import urlparse, parse_qs, quote
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
from requests import Session
import re
import requests


class YTFetch:
    """Simple and elegant YouTube transcript fetcher with language support."""
    
    SUPPORTED_DOMAINS = {"www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com"}
    
    def __init__(self, 
                 webshare_username: Optional[str] = None,
                 webshare_password: Optional[str] = None,
                 http_proxy: Optional[str] = None,
                 https_proxy: Optional[str] = None,
                 custom_session: Optional[Session] = None):
        """
        Initialize YTFetch with optional proxy configuration.
        
        Args:
            webshare_username: Webshare proxy username for residential proxies
            webshare_password: Webshare proxy password for residential proxies
            http_proxy: Generic HTTP proxy URL (format: http://user:pass@domain:port)
            https_proxy: Generic HTTPS proxy URL (format: https://user:pass@domain:port)
            custom_session: Custom requests.Session for advanced configuration
        """
        proxy_config = None
        
        # Configure Webshare proxy (takes precedence)
        if webshare_username and webshare_password:
            proxy_config = WebshareProxyConfig(
                proxy_username=webshare_username,
                proxy_password=webshare_password
            )
        
        # Configure generic proxy if no Webshare config
        elif http_proxy or https_proxy:
            proxy_config = GenericProxyConfig(
                http_url=http_proxy,
                https_url=https_proxy
            )
        
        # Initialize the API with proxy configuration
        if proxy_config:
            self.api = YouTubeTranscriptApi(proxy_config=proxy_config)
        elif custom_session:
            self.api = YouTubeTranscriptApi(http_client=custom_session)
        else:
            self.api = YouTubeTranscriptApi()
            
        self.formatter = TextFormatter()
        
        # Store session for search functionality
        self.session = custom_session or Session()
        if http_proxy or https_proxy:
            self.session.proxies = {
                'http': http_proxy,
                'https': https_proxy or http_proxy
            }
    
    @classmethod
    def with_webshare_proxy(cls, username: str, password: str) -> 'YTFetch':
        """
        Create YTFetch instance with Webshare residential proxy configuration.
        
        Args:
            username: Webshare proxy username
            password: Webshare proxy password
            
        Returns:
            YTFetch instance configured with Webshare proxy
        """
        return cls(webshare_username=username, webshare_password=password)
    
    @classmethod
    def with_generic_proxy(cls, http_url: Optional[str] = None, 
                          https_url: Optional[str] = None) -> 'YTFetch':
        """
        Create YTFetch instance with generic proxy configuration.
        
        Args:
            http_url: HTTP proxy URL (format: http://user:pass@domain:port)
            https_url: HTTPS proxy URL (format: https://user:pass@domain:port)
            
        Returns:
            YTFetch instance configured with generic proxy
        """
        return cls(http_proxy=http_url, https_proxy=https_url)
    
    @classmethod
    def with_custom_session(cls, session: Session) -> 'YTFetch':
        """
        Create YTFetch instance with custom requests.Session.
        
        Args:
            session: Custom requests.Session with your configuration
            
        Returns:
            YTFetch instance using the custom session
        """
        return cls(custom_session=session)

    def transcribe(self, url: str, target_language: Optional[str] = None,
                   format_as_text: bool = True) -> Union[str, List[Dict[str, Any]]]:
        """
        Fetch transcript from YouTube video.
        
        Args:
            url: YouTube video URL
            target_language: Language code (e.g., 'en', 'es', 'fr'). If None, uses first available.
            format_as_text: If True, returns formatted string. If False, returns raw transcript data.
            
        Returns:
            Transcript as string (default) or list of transcript entries with timestamps
        """
        video_id = self._extract_video_id(url)
        
        # Get the transcript using the new API
        if target_language:
            languages = [target_language]
        else:
            languages = ['en']  # Default to English, but will fall back to any available
        
        try:
            # Try to fetch with specified language(s)
            fetched_transcript = self.api.fetch(video_id, languages=languages)
        except Exception:
            # Fall back to any available language
            try:
                fetched_transcript = self.api.fetch(video_id)
            except Exception as e:
                raise ValueError(f"Could not fetch transcript for video {video_id}: {e}")
        
        if format_as_text:
            return self.formatter.format_transcript(fetched_transcript)
        else:
            return fetched_transcript.to_raw_data()
    
    def search_and_transcribe(self, query: str, k: int = 5, 
                            target_language: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Search YouTube for videos matching the query and transcribe the top k results.
        
        Args:
            query: Search query string
            k: Number of videos to transcribe (default: 5)
            target_language: Language code for transcripts (e.g., 'en', 'es', 'fr')
            
        Returns:
            List of dictionaries containing video metadata and transcripts
            Each dict has: 'title', 'url', 'video_id', 'transcript', 'error' (if any)
        """
        # Search for videos
        search_results = self._search_youtube(query, max_results=k)
        
        # Transcribe each video
        transcribed_results = []
        
        for video in search_results:
            result = {
                'title': video['title'],
                'url': video['url'],
                'video_id': video['video_id'],
                'transcript': None,
                'error': None
            }
            
            try:
                # Attempt to transcribe the video
                transcript = self.transcribe(video['url'], target_language=target_language)
                result['transcript'] = transcript
            except Exception as e:
                # Store error if transcription fails
                result['error'] = str(e)
            
            transcribed_results.append(result)
        
        return transcribed_results
    
    def _search_youtube(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search YouTube and return video information.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dicts with 'title', 'url', and 'video_id' keys
        """
        # URL encode the query
        encoded_query = quote(query)
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        
        # Set headers to appear as a regular browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }
        
        try:
            # Make the search request
            response = self.session.get(search_url, headers=headers)
            response.raise_for_status()
            
            # Parse the response to extract video IDs
            video_ids = self._extract_video_ids_from_search(response.text, max_results)
            
            # Build results
            results = []
            for video_id, title in video_ids:
                results.append({
                    'title': title,
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'video_id': video_id
                })
            
            return results
            
        except Exception as e:
            raise ValueError(f"Failed to search YouTube: {e}")
    
    def _extract_video_ids_from_search(self, html_content: str, max_results: int) -> List[tuple]:
        """
        Extract video IDs and titles from YouTube search results HTML.
        
        Args:
            html_content: HTML content from YouTube search
            max_results: Maximum number of results to extract
            
        Returns:
            List of tuples (video_id, title)
        """
        # Method 1: Try to extract from JSON data in the page
        video_data = []
        
        # Look for the initial data JSON
        json_pattern = r'var ytInitialData = ({.*?});'
        json_match = re.search(json_pattern, html_content, re.DOTALL)
        
        if json_match:
            try:
                import json
                data = json.loads(json_match.group(1))
                
                # Navigate through the JSON structure to find video results
                contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
                
                for section in contents:
                    items = section.get('itemSectionRenderer', {}).get('contents', [])
                    for item in items:
                        video_renderer = item.get('videoRenderer', {})
                        if video_renderer:
                            video_id = video_renderer.get('videoId')
                            title = video_renderer.get('title', {}).get('runs', [{}])[0].get('text', 'Unknown Title')
                            if video_id:
                                video_data.append((video_id, title))
                                if len(video_data) >= max_results:
                                    return video_data
            except:
                pass
        
        # Method 2: Fallback to regex pattern matching
        if not video_data:
            # Pattern to find video IDs in various contexts
            patterns = [
                r'"/watch\?v=([a-zA-Z0-9_-]{11})"',
                r'"videoId":"([a-zA-Z0-9_-]{11})"',
                r'/vi/([a-zA-Z0-9_-]{11})/',
            ]
            
            video_ids_set = set()
            for pattern in patterns:
                matches = re.findall(pattern, html_content)
                for match in matches:
                    if match not in video_ids_set:
                        video_ids_set.add(match)
                        # Try to find title near the video ID
                        title = self._extract_title_near_video_id(html_content, match)
                        video_data.append((match, title))
                        if len(video_data) >= max_results:
                            return video_data
        
        return video_data[:max_results]
    
    def _extract_title_near_video_id(self, html_content: str, video_id: str) -> str:
        """
        Attempt to extract video title near a video ID in HTML.
        
        Args:
            html_content: HTML content
            video_id: Video ID to search near
            
        Returns:
            Title string or 'Unknown Title'
        """
        patterns = [
            # …"title":{"runs":[{"text":"THE TITLE"}],"accessibility…
            rf'"title"\s*:\s*{{\s*"runs"\s*:\s*\[\s*{{\s*"text"\s*:\s*"([^"]+?)"\s*}}\s*\]\s*}}\s*,\s*"videoId"\s*:\s*"{re.escape(video_id)}"',
            
            # …"videoId":"<id>", "title":{"runs":[{"text":"THE TITLE"}]…
            rf'"videoId"\s*:\s*"{re.escape(video_id)}"\s*,\s*"title"\s*:\s*{{\s*"runs"\s*:\s*\[\s*{{\s*"text"\s*:\s*"([^"]+?)"',
            
            # Fallback:   <a href="/watch?v=<id>" … > TITLE </a>
            rf'href="/watch\?v={re.escape(video_id)}"[^>]*>\s*([^<]+?)\s*</a>',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                return match.group(1)
        
        return "Unknown Title"
    
    def get_available_languages(self, url: str) -> List[Dict[str, Any]]:
        """Get list of available transcript languages for a video."""
        video_id = self._extract_video_id(url)
        try:
            transcript_list = self.api.list(video_id)
            return [
                {
                    "language": transcript.language,
                    "language_code": transcript.language_code,
                    "is_generated": transcript.is_generated,
                    "is_translatable": transcript.is_translatable
                }
                for transcript in transcript_list
            ]
        except Exception as e:
            raise ValueError(f"Could not fetch available languages for video {video_id}: {e}")
    
    def transcribe_with_translation(self, url: str, target_language: str, 
                                   format_as_text: bool = True) -> Union[str, List[Dict[str, Any]]]:
        """
        Fetch transcript and translate it to target language using YouTube's translation feature.
        
        Args:
            url: YouTube video URL
            target_language: Language code to translate to (e.g., 'es', 'fr', 'de')
            format_as_text: If True, returns formatted string. If False, returns raw transcript data.
            
        Returns:
            Translated transcript as string (default) or list of transcript entries with timestamps
        """
        video_id = self._extract_video_id(url)
        
        try:
            # Get available transcripts
            transcript_list = self.api.list(video_id)
            
            # Find the first available transcript (preferring manual over generated)
            transcript = transcript_list.find_transcript(['en'])  # Start with English
            
            # Translate to target language
            translated_transcript = transcript.translate(target_language)
            fetched_transcript = translated_transcript.fetch()
            
            if format_as_text:
                return self.formatter.format_transcript(fetched_transcript)
            else:
                return fetched_transcript.to_raw_data()
                
        except Exception as e:
            raise ValueError(f"Could not translate transcript for video {video_id}: {e}")
    
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        # Clean the URL
        url = url.strip()
        
        # Try regex pattern first (handles most cases elegantly)
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Fallback to URL parsing
        parsed = urlparse(url)
        self._validate_domain(parsed)
        
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/")
        elif parsed.hostname in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
            query = parse_qs(parsed.query)
            video_id = query.get("v", [None])[0]
            if not video_id:
                raise ValueError("Missing video ID in YouTube URL")
            return video_id
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def _validate_domain(self, parsed_url) -> None:
        """Validate that the URL is from a supported YouTube domain."""
        if parsed_url.hostname not in self.SUPPORTED_DOMAINS:
            raise ValueError(f"Unsupported domain: {parsed_url.hostname}")
    
    def debug_transcript_access(self, url: str) -> Dict[str, Any]:
        """Debug method to understand transcript availability issues."""
        video_id = self._extract_video_id(url)
        debug_info = {"video_id": video_id, "transcripts": [], "errors": []}
        
        try:
            transcript_list = self.api.list(video_id)
            
            for transcript in transcript_list:
                transcript_info = {
                    "language": transcript.language,
                    "language_code": transcript.language_code,
                    "is_generated": transcript.is_generated,
                    "is_translatable": transcript.is_translatable,
                    "fetch_success": False,
                    "fetch_error": None
                }
                
                try:
                    fetched_transcript = transcript.fetch()
                    transcript_info["fetch_success"] = True
                    raw_data = fetched_transcript.to_raw_data()
                    transcript_info["sample_text"] = raw_data[0]["text"] if raw_data else "No text"
                    transcript_info["entry_count"] = len(raw_data)
                except Exception as e:
                    transcript_info["fetch_error"] = str(e)
                    debug_info["errors"].append(f"{transcript.language}: {str(e)}")
                
                debug_info["transcripts"].append(transcript_info)
                
        except Exception as e:
            debug_info["list_error"] = str(e)
            
        return debug_info


# Simple function to search and transcribe YouTube videos
def search_youtube_and_transcribe(query: str, k: int = 5, target_language: Optional[str] = None) -> List[str]:
    """
    Search YouTube for videos matching the query and return transcripts of top k results.
    
    Args:
        query: Search query string
        k: Number of videos to transcribe (default: 5)
        target_language: Language code for transcripts (e.g., 'en', 'es', 'fr')
        
    Returns:
        List of transcript strings (only successful transcriptions)
    """
    fetcher = YTFetch()
    results = fetcher.search_and_transcribe(query, k=k, target_language=target_language)
    
    # Extract only successful transcripts
    transcripts = []
    for result in results:
        if result['transcript'] and not result['error']:
            transcripts.append(result['transcript'])
    
    return transcripts


# Example usage
if __name__ == "__main__":
    # Example 1: Simple usage with the standalone function
    print("=== Example 1: Simple function usage ===")
    transcripts = search_youtube_and_transcribe("machine learning tutorial", k=3)
    for i, transcript in enumerate(transcripts, 1):
        print(f"\nVideo {i} transcript (first 200 chars):")
        print(transcript[:200] + "...")
    
    # Example 2: Using the class method with more details
    print("\n\n=== Example 2: Detailed class usage ===")
    fetcher = YTFetch()
    results = fetcher.search_and_transcribe("python programming", k=3, target_language='en')
    
    for i, result in enumerate(results, 1):
        print(f"\nVideo {i}:")
        print(f"  Title: {result['title']}")
        print(f"  URL: {result['url']}")
        if result['transcript']:
            print(f"  Transcript: {result['transcript'][:150]}...")
        else:
            print(f"  Error: {result['error']}")
    
    # Example 3: With proxy configuration
    print("\n\n=== Example 3: Proxy configuration ===")
    # Uncomment to use with proxy
    # proxy_fetcher = YTFetch.with_generic_proxy(
    #     http_url="http://user:pass@proxy.com:8080"
    # )
    # proxy_transcripts = proxy_fetcher.search_and_transcribe("data science", k=2)