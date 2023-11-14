from llama_index.text_splitter import SentenceSplitter

from loguru import logger
from typing import List


class TextSplitter:
    def __init__(self, chunk_size: int, overlap: int):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def split(self, text: str) -> List[str]:
        chunks = self.splitter.split_text(text)
        logger.info(f'Chunks: {len(chunks)}')
        return chunks
