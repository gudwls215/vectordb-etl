"""
Milvus Vector Store Module
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from pymilvus import (
    MilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema,
)

from .config import MilvusConfig, PipelineConfig, get_config
from .embeddings import get_embeddings, BGEM3Embeddings


class MilvusVectorStore:
    """Milvus 벡터 저장소 클래스"""
    
    def __init__(
        self, 
        config: Optional[MilvusConfig] = None,
        embeddings: Optional[BGEM3Embeddings] = None
    ):
        self.config = config or get_config().milvus
        self.embeddings = embeddings or get_embeddings()
        self._client = None
        
        # 로컬 파일 기반인 경우에만 디렉토리 생성
        if not self.config.uri.startswith(('http://', 'https://')):
            uri_dir = os.path.dirname(self.config.uri)
            if uri_dir and not os.path.exists(uri_dir):
                os.makedirs(uri_dir, exist_ok=True)
    
    @property
    def client(self) -> MilvusClient:
        """Milvus 클라이언트 (지연 연결)"""
        if self._client is None:
            print(f"Connecting to Milvus: {self.config.uri}")
            self._client = MilvusClient(uri=self.config.uri)
            print("Connected to Milvus successfully!")
        return self._client
    
    def _get_collection_schema(self) -> CollectionSchema:
        """컬렉션 스키마 생성"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embeddings.get_embedding_dimension()),
            # 메타데이터 필드
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="total_chunks", dtype=DataType.INT32),
            FieldSchema(name="chunk_size_chars", dtype=DataType.INT32),
            FieldSchema(name="chunk_size_tokens", dtype=DataType.INT32),
        ]
        
        return CollectionSchema(
            fields=fields,
            description="HTML Document Chunks Vector Store",
            enable_dynamic_field=True  # 추가 메타데이터 허용
        )
    
    def collection_exists(self) -> bool:
        """컬렉션 존재 여부 확인"""
        return self.client.has_collection(self.config.collection_name)
    
    def _ensure_collection_loaded(self) -> None:
        """컬렉션이 메모리에 로드되어 있는지 확인하고 로드"""
        if self.collection_exists():
            # MilvusClient는 자동으로 컬렉션을 로드하지만, 명시적으로 로드
            self.client.load_collection(self.config.collection_name)
    
    def create_collection(self, drop_existing: bool = False) -> None:
        """컬렉션 생성"""
        if drop_existing and self.collection_exists():
            print(f"Dropping existing collection: {self.config.collection_name}")
            self.client.drop_collection(self.config.collection_name)
        
        if not self.collection_exists():
            print(f"Creating collection: {self.config.collection_name}")
            self.client.create_collection(
                collection_name=self.config.collection_name,
                schema=self._get_collection_schema(),
            )
            
            # 인덱스 생성
            self._create_index()
            print(f"Collection created: {self.config.collection_name}")
        else:
            print(f"Collection already exists: {self.config.collection_name}")
    
    def _create_index(self) -> None:
        """벡터 인덱스 생성"""
        index_params = self.client.prepare_index_params()
        
        if self.config.index_type == "HNSW":
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type=self.config.metric_type,
                params={
                    "M": 16,
                    "efConstruction": self.config.ef_construction
                }
            )
        else:  # IVF_FLAT
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type=self.config.metric_type,
                params={"nlist": self.config.nlist}
            )
        
        self.client.create_index(
            collection_name=self.config.collection_name,
            index_params=index_params
        )
    
    def insert_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """문서 삽입"""
        if not self.collection_exists():
            self.create_collection()
        
        total_inserted = 0
        
        # 배치 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 텍스트 임베딩
            texts = [doc.page_content for doc in batch]
            embeddings = self.embeddings.embed_documents(texts)
            
            # 데이터 준비
            data = []
            for doc, embedding in zip(batch, embeddings):
                record = {
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "text": doc.page_content[:65535],  # VARCHAR 최대 길이
                    "embedding": embedding,
                    "source": doc.metadata.get("source", "")[:512],
                    "filename": doc.metadata.get("filename", "")[:256],
                    "language": doc.metadata.get("language", "unknown")[:32],
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "total_chunks": doc.metadata.get("total_chunks", 0),
                    "chunk_size_chars": doc.metadata.get("chunk_size_chars", 0),
                    "chunk_size_tokens": doc.metadata.get("chunk_size_tokens", 0),
                }
                data.append(record)
            
            # 삽입
            result = self.client.insert(
                collection_name=self.config.collection_name,
                data=data
            )
            total_inserted += len(data)
            print(f"Inserted batch {i // batch_size + 1}: {len(data)} documents")
        
        print(f"Total documents inserted: {total_inserted}")
        return total_inserted
    
    def search(
        self,
        query: str,
        k: int = 3,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """벡터 검색"""
        # 컬렉션 로드 확인 (검색 전 필수)
        self._ensure_collection_loaded()
        
        # 쿼리 임베딩
        query_embedding = self.embeddings.embed_query(query)
        
        # 검색 파라미터
        search_params = {}
        if self.config.index_type == "HNSW":
            search_params = {"ef": self.config.ef_search}
        else:
            search_params = {"nprobe": self.config.nprobe}
        
        # 출력 필드
        if output_fields is None:
            output_fields = [
                "chunk_id", "text", "source", "filename", 
                "language", "chunk_index", "total_chunks",
                "chunk_size_chars", "chunk_size_tokens"
            ]
        
        # 검색 실행
        results = self.client.search(
            collection_name=self.config.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=k,
            filter=filter_expr,
            output_fields=output_fields
        )
        
        return results[0] if results else []
    
    def search_with_scores(
        self,
        query: str,
        k: int = 3,
        filter_expr: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """점수와 함께 검색 (LangChain 호환 형식)"""
        results = self.search(query, k=k, filter_expr=filter_expr)
        
        doc_score_pairs = []
        for hit in results:
            doc = Document(
                page_content=hit['entity'].get('text', ''),
                metadata={
                    'chunk_id': hit['entity'].get('chunk_id', ''),
                    'source': hit['entity'].get('source', ''),
                    'filename': hit['entity'].get('filename', ''),
                    'language': hit['entity'].get('language', ''),
                    'chunk_index': hit['entity'].get('chunk_index', 0),
                    'total_chunks': hit['entity'].get('total_chunks', 0),
                    'chunk_size_chars': hit['entity'].get('chunk_size_chars', 0),
                    'chunk_size_tokens': hit['entity'].get('chunk_size_tokens', 0),
                }
            )
            # Milvus distance를 similarity score로 변환 (COSINE의 경우)
            score = hit['distance']
            doc_score_pairs.append((doc, score))
        
        return doc_score_pairs
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 조회"""
        if not self.collection_exists():
            return {"exists": False}
        
        stats = self.client.get_collection_stats(self.config.collection_name)
        return {
            "exists": True,
            "collection_name": self.config.collection_name,
            "row_count": stats.get("row_count", 0),
        }
    
    def drop_collection(self) -> None:
        """컬렉션 삭제"""
        if self.collection_exists():
            self.client.drop_collection(self.config.collection_name)
            print(f"Collection dropped: {self.config.collection_name}")
        else:
            print(f"Collection does not exist: {self.config.collection_name}")
    
    def close(self) -> None:
        """연결 종료"""
        if self._client is not None:
            self._client.close()
            self._client = None


# 편의 함수들
_store_instance: Optional[MilvusVectorStore] = None


def get_vector_store(
    config: Optional[MilvusConfig] = None,
    embeddings: Optional[BGEM3Embeddings] = None
) -> MilvusVectorStore:
    """벡터 저장소 싱글톤 인스턴스 반환"""
    global _store_instance
    if _store_instance is None:
        _store_instance = MilvusVectorStore(config, embeddings)
    return _store_instance


def reset_vector_store() -> None:
    """벡터 저장소 인스턴스 리셋"""
    global _store_instance
    if _store_instance is not None:
        _store_instance.close()
    _store_instance = None


def create_vector_store(
    documents: List[Document],
    config: Optional[MilvusConfig] = None,
    drop_existing: bool = False
) -> MilvusVectorStore:
    """문서로부터 벡터 저장소 생성"""
    store = MilvusVectorStore(config)
    store.create_collection(drop_existing=drop_existing)
    store.insert_documents(documents)
    return store
