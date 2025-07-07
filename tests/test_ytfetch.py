import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json

# Assuming your YTFetch class is in a module called ytfetch
# from ytfetch import YTFetch, search_youtube_and_transcribe


class TestYTFetch:
    """Test suite for YTFetch with mocked YouTube interactions."""
    
    @pytest.fixture
    def mock_search_response(self):
        """Mock HTML response from YouTube search."""
        # Simulate YouTube's initial data JSON structure
        mock_data = {
            "contents": {
                "twoColumnSearchResultsRenderer": {
                    "primaryContents": {
                        "sectionListRenderer": {
                            "contents": [{
                                "itemSectionRenderer": {
                                    "contents": [
                                        {
                                            "videoRenderer": {
                                                "videoId": "dQw4w9WgXcQ",
                                                "title": {
                                                    "runs": [{"text": "Rick Astley - Never Gonna Give You Up"}]
                                                }
                                            }
                                        },
                                        {
                                            "videoRenderer": {
                                                "videoId": "jNQXAC9IVRw",
                                                "title": {
                                                    "runs": [{"text": "Me at the zoo"}]
                                                }
                                            }
                                        },
                                        {
                                            "videoRenderer": {
                                                "videoId": "9bZkp7q19f0",
                                                "title": {
                                                    "runs": [{"text": "PSY - GANGNAM STYLE"}]
                                                }
                                            }
                                        }
                                    ]
                                }
                            }]
                        }
                    }
                }
            }
        }
        
        # Create HTML with embedded JSON
        html_content = f"""
        <html>
        <head><title>YouTube Search Results</title></head>
        <body>
        <script>
        var ytInitialData = {json.dumps(mock_data)};
        </script>
        </body>
        </html>
        """
        
        mock_response = Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = Mock()
        return mock_response
    
    @pytest.fixture
    def mock_transcript_api(self):
        """Mock YouTubeTranscriptApi."""
        with patch('youtube_transcript_api.YouTubeTranscriptApi') as mock_api:
            # Create mock transcript data
            mock_transcript = Mock()
            mock_transcript.to_raw_data.return_value = [
                {"text": "Hello everyone", "start": 0.0, "duration": 2.0},
                {"text": "Welcome to my video", "start": 2.0, "duration": 3.0},
                {"text": "Today we'll learn about Python", "start": 5.0, "duration": 4.0}
            ]
            
            # Mock the fetch method
            mock_api_instance = Mock()
            mock_api_instance.fetch.return_value = mock_transcript
            
            # Mock the list method for available languages
            mock_transcript_item = Mock()
            mock_transcript_item.language = "English"
            mock_transcript_item.language_code = "en"
            mock_transcript_item.is_generated = False
            mock_transcript_item.is_translatable = True
            mock_transcript_item.fetch.return_value = mock_transcript
            
            mock_transcript_list = Mock()
            mock_transcript_list.__iter__ = Mock(return_value=iter([mock_transcript_item]))
            mock_transcript_list.find_transcript.return_value = mock_transcript_item
            
            mock_api_instance.list.return_value = mock_transcript_list
            
            # Make the class return our instance
            mock_api.return_value = mock_api_instance
            
            yield mock_api_instance
    
    @pytest.fixture
    def mock_formatter(self):
        """Mock TextFormatter."""
        with patch('youtube_transcript_api.formatters.TextFormatter') as mock_fmt:
            formatter_instance = Mock()
            formatter_instance.format_transcript.return_value = (
                "Hello everyone Welcome to my video Today we'll learn about Python"
            )
            mock_fmt.return_value = formatter_instance
            yield formatter_instance
    
    @pytest.fixture
    def ytfetch_instance(self, mock_transcript_api, mock_formatter):
        """Create YTFetch instance with mocked dependencies."""
        # Import here to ensure mocks are in place
        from src.pipeline.yt_fetch import YTFetch
        return YTFetch()
    
    def test_search_and_transcribe(self, ytfetch_instance, mock_search_response):
        """Test search_and_transcribe method."""
        with patch.object(ytfetch_instance.session, 'get', return_value=mock_search_response):
            results = ytfetch_instance.search_and_transcribe("test query", k=2)
            
            # Check we got 2 results
            assert len(results) == 2
            
            # Check the structure of results
            for result in results:
                assert 'title' in result
                assert 'url' in result
                assert 'video_id' in result
                assert 'transcript' in result
                assert 'error' in result
            
            # Check specific values
            assert results[0]['video_id'] == 'dQw4w9WgXcQ'
            assert results[0]['title'] == 'Rick Astley - Never Gonna Give You Up'
            assert results[0]['url'] == 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
            assert results[0]['transcript'] is not None
    
    def test_search_youtube(self, ytfetch_instance, mock_search_response):
        """Test _search_youtube method."""
        with patch.object(ytfetch_instance.session, 'get', return_value=mock_search_response):
            results = ytfetch_instance._search_youtube("test query", max_results=3)
            
            assert len(results) == 3
            assert results[0]['video_id'] == 'dQw4w9WgXcQ'
            assert results[1]['video_id'] == 'jNQXAC9IVRw'
            assert results[2]['video_id'] == '9bZkp7q19f0'
    
    def test_transcribe_single_video(self, ytfetch_instance):
        """Test transcribing a single video."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        transcript = ytfetch_instance.transcribe(url)
        
        assert transcript == "Hello everyone Welcome to my video Today we'll learn about Python"
    
    def test_transcribe_with_language(self, ytfetch_instance):
        """Test transcribing with specific language."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        transcript = ytfetch_instance.transcribe(url, target_language='es')
        
        # The mock should still return English transcript for simplicity
        assert transcript is not None
    
    def test_get_available_languages(self, ytfetch_instance):
        """Test getting available languages."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        languages = ytfetch_instance.get_available_languages(url)
        
        assert len(languages) == 1
        assert languages[0]['language'] == 'English'
        assert languages[0]['language_code'] == 'en'
        assert languages[0]['is_generated'] is False
        assert languages[0]['is_translatable'] is True
    
    def test_extract_video_id_various_formats(self, ytfetch_instance):
        """Test video ID extraction from various URL formats."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]
        
        for url, expected_id in test_cases:
            video_id = ytfetch_instance._extract_video_id(url)
            assert video_id == expected_id
    
    def test_extract_video_id_invalid_url(self, ytfetch_instance):
        """Test video ID extraction with invalid URL."""
        with pytest.raises(ValueError):
            ytfetch_instance._extract_video_id("https://vimeo.com/123456")
    
    def test_search_with_no_results(self, ytfetch_instance):
        """Test search when no results are found."""
        mock_response = Mock()
        mock_response.text = "<html><body>No results found</body></html>"
        mock_response.raise_for_status = Mock()
        
        with patch.object(ytfetch_instance.session, 'get', return_value=mock_response):
            results = ytfetch_instance._search_youtube("very specific query with no results")
            assert len(results) == 0
    
    def test_transcribe_error_handling(self, ytfetch_instance, mock_transcript_api):
        """Test error handling when transcript is not available."""
        # Make the API raise an exception
        mock_transcript_api.fetch.side_effect = Exception("No transcript available")
        
        with pytest.raises(ValueError) as exc_info:
            ytfetch_instance.transcribe("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        
        assert "Could not fetch transcript" in str(exc_info.value)
    
    def test_search_and_transcribe_with_errors(self, ytfetch_instance, mock_search_response, mock_transcript_api):
        """Test search_and_transcribe when some videos fail."""
        # Make the API fail for the second video
        call_count = 0
        def fetch_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("No transcript available")
            else:
                mock_transcript = Mock()
                mock_transcript.to_raw_data.return_value = [{"text": "Success", "start": 0, "duration": 1}]
                return mock_transcript
        
        mock_transcript_api.fetch.side_effect = fetch_side_effect
        
        with patch.object(ytfetch_instance.session, 'get', return_value=mock_search_response):
            results = ytfetch_instance.search_and_transcribe("test", k=2)
            
            # First video should succeed
            assert results[0]['transcript'] is not None
            assert results[0]['error'] is None
            
            # Second video should have error
            assert results[1]['transcript'] is None
            assert results[1]['error'] is not None
            assert "No transcript available" in results[1]['error']


class TestStandaloneFunction:
    """Test the standalone search_youtube_and_transcribe function."""
    
    @pytest.fixture
    def mock_ytfetch_class(self):
        """Mock the entire YTFetch class."""
        with patch('ytfetch.YTFetch') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Mock search_and_transcribe results
            mock_instance.search_and_transcribe.return_value = [
                {
                    'title': 'Video 1',
                    'url': 'https://youtube.com/watch?v=1',
                    'video_id': '1',
                    'transcript': 'This is the first video transcript',
                    'error': None
                },
                {
                    'title': 'Video 2',
                    'url': 'https://youtube.com/watch?v=2',
                    'video_id': '2',
                    'transcript': None,
                    'error': 'No transcript available'
                },
                {
                    'title': 'Video 3',
                    'url': 'https://youtube.com/watch?v=3',
                    'video_id': '3',
                    'transcript': 'This is the third video transcript',
                    'error': None
                }
            ]
            
            yield mock_instance
    
    def test_search_youtube_and_transcribe(self, mock_ytfetch_class):
        """Test the standalone function."""
        from src.pipeline.yt_fetch import search_youtube_and_transcribe
        
        transcripts = search_youtube_and_transcribe("test query", k=3)
        
        # Should only return successful transcripts
        assert len(transcripts) == 2
        assert transcripts[0] == 'This is the first video transcript'
        assert transcripts[1] == 'This is the third video transcript'
        
        # Verify the mock was called correctly
        mock_ytfetch_class.search_and_transcribe.assert_called_once_with(
            "test query", k=3, target_language=None
        )


class TestProxyConfiguration:
    """Test proxy configuration options."""
    
    def test_webshare_proxy_init(self):
        """Test initialization with Webshare proxy."""
        with patch('youtube_transcript_api.proxies.WebshareProxyConfig') as mock_proxy:
            from src.pipeline.yt_fetch import YTFetch
            fetcher = YTFetch.with_webshare_proxy("user", "pass")
            
            mock_proxy.assert_called_once_with(
                proxy_username="user",
                proxy_password="pass"
            )
    
    def test_generic_proxy_init(self):
        """Test initialization with generic proxy."""
        with patch('youtube_transcript_api.proxies.GenericProxyConfig') as mock_proxy:
            from src.pipeline.yt_fetch import YTFetch
            fetcher = YTFetch.with_generic_proxy(
                http_url="http://proxy:8080",
                https_url="https://proxy:8443"
            )
            
            mock_proxy.assert_called_once_with(
                http_url="http://proxy:8080",
                https_url="https://proxy:8443"
            )
    
    def test_custom_session_init(self):
        """Test initialization with custom session."""
        from requests import Session
        custom_session = Session()
        custom_session.headers['User-Agent'] = 'Custom Bot'
        
        with patch('youtube_transcript_api.YouTubeTranscriptApi') as mock_api:
            from src.pipeline.yt_fetch import YTFetch
            fetcher = YTFetch.with_custom_session(custom_session)
            
            mock_api.assert_called_once_with(http_client=custom_session)


# Integration test example (still mocked but more comprehensive)
class TestIntegration:
    """Integration tests that mock external dependencies but test full workflows."""
    
    @pytest.fixture
    def full_mock_setup(self):
        """Set up all mocks for integration testing."""
        with patch('yt_fetch.YouTubeTranscriptApi') as mock_api, \
             patch('yt_fetch.TextFormatter') as mock_formatter, \
             patch('yt_fetch.Session') as mock_session:
            
            # Configure all mocks
            # ... (similar to above fixtures)
            
            yield {
                'api': mock_api,
                'formatter': mock_formatter,
                'session': mock_session
            }
    
    def test_full_search_and_transcribe_workflow(self, full_mock_setup):
        """Test the complete workflow from search to transcription."""
        # This would test the entire flow with all components mocked
        pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])