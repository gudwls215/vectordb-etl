"""
Hybrid Document Chunker Module
1단계: 시맨틱 분할 (의미 단위)
2단계: 문자 기준 분할/병합 (크기 최적화)
"""
import re
import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️ langchain-experimental 미설치. 문자 기준 청킹만 사용합니다.")
    print("   설치: pip install langchain-experimental")

from .config import ChunkerConfig, get_config, EmbeddingConfig


class HybridChunker:
    """
    하이브리드 문서 분할 클래스
    
    1단계: 시맨틱 청킹 - 의미적으로 유사한 문장들을 그룹화
    2단계: 크기 최적화 - max_chunk_size 초과시 분할, min_chunk_size 미만시 병합
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None, embedding_config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().chunker
        self.embedding_config = embedding_config or get_config().embedding
        
        # 문자 기준 분할기 (2단계용)
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.target_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
            is_separator_regex=False
        )
        
        # 시맨틱 분할기 (1단계용)
        self.semantic_splitter = None
        if self.config.chunking_mode == "semantic_first" and SEMANTIC_AVAILABLE:
            try:
                from .embeddings import get_embeddings
                embeddings = get_embeddings(self.embedding_config)
                self.semantic_splitter = LangChainSemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type=self.config.breakpoint_threshold_type,
                    breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
                )
                print(f"🧠 하이브리드 청킹 활성화 (1단계: 시맨틱 → 2단계: 문자 기준)")
                print(f"   시맨틱: {self.config.breakpoint_threshold_type}={self.config.breakpoint_threshold_amount}")
                print(f"   문자: target={self.config.target_chunk_size}, min={self.config.min_chunk_size}, max={self.config.max_chunk_size}")
            except Exception as e:
                print(f"⚠️ 시맨틱 청킹 초기화 실패: {e}")
                print("   문자 기준 청킹으로 fallback")
        else:
            print(f"📝 문자 기준 청킹 사용 (size={self.config.target_chunk_size}, overlap={self.config.chunk_overlap})")
    
    def _estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (한글 고려)"""
        korean_chars = len(re.findall(r'[가-힣]', text))
        other_chars = len(text) - korean_chars
        return int(korean_chars / 1.5 + other_chars / 4)
    
    def _split_large_chunk(self, chunk: Document) -> List[Document]:
        """max_chunk_size 초과 청크를 문자 기준으로 분할"""
        return self.char_splitter.split_documents([chunk])
    
    def _merge_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        min_chunk_size 미만 청크를 인접 청크와 병합
        의미적 연속성을 유지하면서 병합
        """
        if not chunks:
            return chunks
        
        merged = []
        buffer = None
        
        for chunk in chunks:
            chunk_len = len(chunk.page_content)
            
            if buffer is None:
                buffer = chunk
            elif len(buffer.page_content) < self.config.min_chunk_size:
                # 버퍼가 너무 작으면 현재 청크와 병합
                merged_content = buffer.page_content + "\n" + chunk.page_content
                
                # 병합해도 max를 넘지 않으면 병합
                if len(merged_content) <= self.config.max_chunk_size:
                    buffer = Document(
                        page_content=merged_content,
                        metadata=buffer.metadata.copy()
                    )
                else:
                    # 병합하면 너무 커지므로 버퍼 저장 후 새 버퍼 시작
                    merged.append(buffer)
                    buffer = chunk
            elif chunk_len < self.config.min_chunk_size:
                # 현재 청크가 너무 작으면 버퍼와 병합 시도
                merged_content = buffer.page_content + "\n" + chunk.page_content
                
                if len(merged_content) <= self.config.max_chunk_size:
                    buffer = Document(
                        page_content=merged_content,
                        metadata=buffer.metadata.copy()
                    )
                else:
                    merged.append(buffer)
                    buffer = chunk
            else:
                # 둘 다 적정 크기
                merged.append(buffer)
                buffer = chunk
        
        if buffer is not None:
            merged.append(buffer)
        
        return merged
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """문서 리스트를 하이브리드 방식으로 청크 분할"""
        all_chunks = []
        
        for doc in documents:
            # 1단계: 시맨틱 분할 (가능한 경우)
            if self.semantic_splitter:
                try:
                    semantic_chunks = self.semantic_splitter.split_documents([doc])
                    print(f"   📊 1단계 시맨틱 분할: {len(semantic_chunks)}개 청크")
                except Exception as e:
                    print(f"   ⚠️ 시맨틱 분할 실패, 문자 기준으로 fallback: {e}")
                    semantic_chunks = [doc]
            else:
                semantic_chunks = [doc]
            
            # 2단계: 크기 최적화
            optimized_chunks = []
            for chunk in semantic_chunks:
                chunk_len = len(chunk.page_content)
                
                if chunk_len > self.config.max_chunk_size:
                    # 너무 크면 문자 기준으로 추가 분할
                    sub_chunks = self._split_large_chunk(chunk)
                    optimized_chunks.extend(sub_chunks)
                else:
                    optimized_chunks.append(chunk)
            
            # 3단계: 작은 청크 병합
            final_chunks = self._merge_small_chunks(optimized_chunks)
            
            if self.semantic_splitter:
                print(f"   📊 2단계 크기 최적화 후: {len(final_chunks)}개 청크")
            
            # 메타데이터 부여
            for i, chunk in enumerate(final_chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(final_chunks)
                chunk.metadata['chunk_size_chars'] = len(chunk.page_content)
                chunk.metadata['chunk_size_tokens'] = self._estimate_tokens(chunk.page_content)
                
                chunk_id = hashlib.md5(
                    f"{chunk.metadata.get('source', 'unknown')}_{i}_{chunk.page_content[:50]}".encode()
                ).hexdigest()[:12]
                chunk.metadata['chunk_id'] = chunk_id
            
            all_chunks.extend(final_chunks)
        
        return all_chunks


# 하위 호환성을 위한 별칭
SemanticChunker = HybridChunker


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
    if len(documents) > 0:
        print(f"문서당 평균 청크 수: {len(chunks) / len(documents):.1f}")
    
    if remove_duplicates:
        chunks = remove_duplicate_chunks(chunks, similarity_threshold)
    
    return chunks
