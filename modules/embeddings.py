"""
BGE-M3 Embedding Module
"""
from typing import List, Optional
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig, get_config


class BGEM3Embeddings(Embeddings):
    """BGE-M3 임베딩 모델 래퍼 클래스"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or get_config().embedding
        self.device = self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        
    @property
    def model(self) -> SentenceTransformer:
        """지연 로딩으로 모델 초기화"""
        if self._model is None:
            print(f"Loading BGE-M3 model on {self.device}...")
            self._model = SentenceTransformer(
                self.config.model_name, 
                device=self.device
            )
            print(f"Model loaded successfully!")
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 임베딩"""
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=self.config.show_progress_bar,
            batch_size=self.config.batch_size
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩"""
        embedding = self.model.encode(
            text, 
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding.tolist()
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        return self.config.dimension


# 싱글톤 인스턴스
_embeddings_instance: Optional[BGEM3Embeddings] = None


def get_embeddings(config: Optional[EmbeddingConfig] = None) -> BGEM3Embeddings:
    """임베딩 모델 싱글톤 인스턴스 반환"""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = BGEM3Embeddings(config)
    return _embeddings_instance


def reset_embeddings():
    """임베딩 인스턴스 리셋 (테스트용)"""
    global _embeddings_instance
    _embeddings_instance = None
