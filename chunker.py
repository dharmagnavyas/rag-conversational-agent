"""
Text Chunking Module
Implements smart chunking with overlap and metadata preservation.
"""

from typing import List, Dict, Any
import re
import tiktoken


class TextChunker:
    """
    Chunks text while preserving sentence boundaries and metadata.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Use tiktoken for accurate token counting (GPT compatible)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimate
        return len(text.split())

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while handling common edge cases.
        """
        # Handle common abbreviations
        text = re.sub(r'(\b(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|vs|etc|e\.g|i\.e|Rs|Cr|Mn|Bn))\.', r'\1<PERIOD>', text)

        # Handle numbers with decimals
        text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore periods
        sentences = [s.replace('<PERIOD>', '.').replace('<DECIMAL>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk all pages and return chunks with metadata.
        """
        all_chunks = []
        chunk_id = 0

        for page in pages_data:
            page_num = page['page_num']
            text = page['text']

            if not text or len(text.strip()) < self.min_chunk_size:
                continue

            # Get chunks for this page
            page_chunks = self._chunk_text(text)

            for chunk_text in page_chunks:
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    all_chunks.append({
                        'chunk_id': chunk_id,
                        'page_num': page_num,
                        'text': chunk_text.strip(),
                        'token_count': self.count_tokens(chunk_text),
                        'citation': f"[p{page_num}:c{chunk_id}]"
                    })
                    chunk_id += 1

        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text respecting sentence boundaries with overlap.
        """
        sentences = self.split_into_sentences(text)

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_size > self.chunk_size:
                # First, save current chunk if not empty
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split long sentence into smaller parts
                chunks.extend(self._split_long_sentence(sentence))
                continue

            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                    # Keep overlap sentences for next chunk
                    overlap_sentences = []
                    overlap_size = 0

                    for s in reversed(current_chunk):
                        s_size = self.count_tokens(s)
                        if overlap_size + s_size <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += s_size
                        else:
                            break

                    current_chunk = overlap_sentences
                    current_size = overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a sentence that's too long into smaller parts.
        """
        # Split by common delimiters
        parts = re.split(r'[;,]\s*', sentence)

        chunks = []
        current = []
        current_size = 0

        for part in parts:
            part_size = self.count_tokens(part)

            if current_size + part_size > self.chunk_size:
                if current:
                    chunks.append(', '.join(current))
                current = [part]
                current_size = part_size
            else:
                current.append(part)
                current_size += part_size

        if current:
            chunks.append(', '.join(current))

        return chunks


def create_chunks(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Convenience function to create chunks from pages data.
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_pages(pages_data)


if __name__ == "__main__":
    # Test chunking
    test_pages = [
        {
            'page_num': 1,
            'text': "This is the first sentence. This is the second sentence. Here comes a third one with more content. And finally the fourth sentence.",
            'has_tables': False
        }
    ]

    chunks = create_chunks(test_pages, chunk_size=20, chunk_overlap=5)
    for chunk in chunks:
        print(f"\n{chunk['citation']}: {chunk['text']}")
