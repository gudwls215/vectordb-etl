"""
Semantic Document Chunker Module
"""
import re
import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import ChunkerConfig, get_config


class SemanticChunker:
    """의미 기반 문서 분할 클래스"""
    
    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or get_config().chunker
        
        # 문자 수 기준으로 분할
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,  # 문자 수 기준
            is_separator_regex=False
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (한글 고려) - 메타데이터용"""
        # 한글은 약 1.5자당 1토큰, 영어는 약 4자당 1토큰
        korean_chars = len(re.findall(r'[가-힣]', text))
        other_chars = len(text) - korean_chars
        return int(korean_chars / 1.5 + other_chars / 4)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서 리스트를 청크로 분할"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            # 각 청크에 추가 메타데이터 부여
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                chunk.metadata['chunk_size_chars'] = len(chunk.page_content)
                chunk.metadata['chunk_size_tokens'] = self._estimate_tokens(chunk.page_content)
                
                # 청크 고유 ID 생성
                chunk_id = hashlib.md5(
                    f"{chunk.metadata['source']}_{i}_{chunk.page_content[:50]}".encode()
                ).hexdigest()[:12]
                chunk.metadata['chunk_id'] = chunk_id
                
            all_chunks.extend(chunks)
        
        return all_chunks


def remove_duplicate_chunks(
    chunks: List[Document], 
    similarity_threshold: float = 0.95
) -> List[Document]:
    """중복 청크 제거 (해시 기반)"""
    
    seen_hashes = set()
    unique_chunks = []
    duplicate_count = 0
    
    for chunk in chunks:
        # 정규화된 텍스트의 해시 생성
        normalized_text = ' '.join(chunk.page_content.lower().split())
        text_hash = hashlib.md5(normalized_text.encode()).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_chunks.append(chunk)
        else:
            duplicate_count += 1
    
    print(f"원본 청크 수: {len(chunks)}")
    print(f"중복 제거된 청크 수: {duplicate_count}")
    print(f"최종 청크 수: {len(unique_chunks)}")
    
    return unique_chunks


def chunk_documents(
    documents: List[Document],
    config: Optional[ChunkerConfig] = None,
    remove_duplicates: bool = True,
    similarity_threshold: float = 0.95
) -> List[Document]:
    """문서 청킹 편의 함수"""
    chunker = SemanticChunker(config)
    chunks = chunker.split_documents(documents)
    
    print(f"분할된 청크 수: {len(chunks)}")
    print(f"문서당 평균 청크 수: {len(chunks) / len(documents):.1f}")
    
    if remove_duplicates:
        chunks = remove_duplicate_chunks(chunks, similarity_threshold)
    
    return chunks
