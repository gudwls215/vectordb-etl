"""
Configuration management for VectorDB ETL Pipeline
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# 기본 디렉토리 설정
CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_DIR = os.path.join(CUR_DIR, 'html')
DATA_DIR = os.path.join(CUR_DIR, 'data')


@dataclass
class MilvusConfig:
    """Milvus 연결 설정"""
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "html_documents"
    index_type: str = "IVF_FLAT"  # IVF_FLAT, HNSW, IVF_SQ8
    metric_type: str = "COSINE"   # L2, IP, COSINE
    nlist: int = 128              # IVF 클러스터 수
    nprobe: int = 16              # 검색시 탐색할 클러스터 수
    ef_construction: int = 200    # HNSW 구축 파라미터
    ef_search: int = 100          # HNSW 검색 파라미터
    
    # 연결 URI (로컬 파일 또는 서버)
    uri: Optional[str] = "http://localhost:19530"
    
    def __post_init__(self):
        if self.uri is None:
            # 기본값: 로컬 파일 기반 Milvus Lite
            self.uri = os.path.join(DATA_DIR, "milvus_vectordb.db")
    
    def get_connection_params(self) -> Dict[str, Any]:
        """연결 파라미터 반환"""
        return {
            "uri": self.uri,
        }


@dataclass
class EmbeddingConfig:
    """임베딩 모델 설정"""
    model_name: str = "BAAI/bge-m3"
    device: Optional[str] = None  # None이면 자동 감지 (cuda/cpu)
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress_bar: bool = True
    dimension: int = 1024  # BGE-M3 임베딩 차원


@dataclass
class ChunkerConfig:
    """청크 분할 설정"""
    target_chunk_size: int = 800    # 문자 기준
    chunk_overlap: int = 150        # 오버랩 문자 수
    min_chunk_size: int = 100
    max_chunk_size: int = 1200
    separators: list = field(default_factory=lambda: [
        "\n\n\n",      # 섹션 구분
        "\n\n",        # 문단 구분
        "\n",          # 줄바꿈
        ". ",          # 문장 끝
        "? ",          # 의문문 끝
        "! ",          # 감탄문 끝
        "; ",          # 세미콜론
        ", ",          # 쉼표
        " ",           # 공백
        ""             # 문자
    ])


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정"""
    html_dir: str = field(default_factory=lambda: HTML_DIR)
    data_dir: str = field(default_factory=lambda: DATA_DIR)
    html_glob_pattern: str = field(default="*.html")
    
    # 중복 제거 설정
    duplicate_similarity_threshold: float = field(default=0.95)
    
    # 검색 설정
    default_search_k: int = field(default=3)
    auto_detect_language: bool = field(default=True)
    
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    
    def __post_init__(self):
        # 데이터 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)


# 기본 설정 인스턴스
default_config = PipelineConfig()


def get_config() -> PipelineConfig:
    """기본 설정 반환"""
    return default_config


def create_config(
    milvus_uri: Optional[str] = None,
    collection_name: str = "html_documents",
    embedding_model: str = "BAAI/bge-m3",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    html_dir: str = HTML_DIR,
) -> PipelineConfig:
    """커스텀 설정 생성"""
    milvus_config = MilvusConfig(
        collection_name=collection_name,
        uri=milvus_uri,
    )
    
    embedding_config = EmbeddingConfig(
        model_name=embedding_model,
    )
    
    chunker_config = ChunkerConfig(
        target_chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return PipelineConfig(
        milvus=milvus_config,
        embedding=embedding_config,
        chunker=chunker_config,
        html_dir=html_dir,
    )
