from typing import List

from src.models.chunk import Chunk


class Chunker:
    def __init__(self):
        pass
    
    def chunk(self, input: str) -> List[Chunk]:
        """
        Split input string into 400-character chunks.
        
        Args:
            input: String to be chunked
            
        Returns:
            List of Chunk objects, each containing up to 400 characters
        """
        if not input:
            return []
        
        chunks = []
        for i in range(0, len(input), 400):
            chunk_content = input[i:i + 400]
            chunks.append(Chunk(content=chunk_content))
        
        return chunks