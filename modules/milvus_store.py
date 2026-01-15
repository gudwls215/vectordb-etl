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
    
    def create_collection(self, drop_existing: bool = False, collection_name: Optional[str] = None) -> None:
        """컬렉션 생성"""
        coll_name = collection_name or self.config.collection_name
        
        if drop_existing and self.client.has_collection(coll_name):
            print(f"Dropping existing collection: {coll_name}")
            self.client.drop_collection(coll_name)
        
        if not self.client.has_collection(coll_name):
            print(f"Creating collection: {coll_name}")
            self.client.create_collection(
                collection_name=coll_name,
                schema=self._get_collection_schema(),
            )
            
            # 인덱스 생성
            self._create_index(coll_name)
            print(f"Collection created: {coll_name}")
        else:
            print(f"Collection already exists: {coll_name}")
    
    def _create_collection_by_name(self, collection_name: str) -> None:
        """이름으로 컬렉션 생성 (내부용)"""
        if not self.client.has_collection(collection_name):
            print(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                schema=self._get_collection_schema(),
            )
            self._create_index(collection_name)
            print(f"Collection created: {collection_name}")
    
    def _create_index(self, collection_name: Optional[str] = None) -> None:
        """벡터 인덱스 생성"""
        coll_name = collection_name or self.config.collection_name
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
            collection_name=coll_name,
            index_params=index_params
        )
    
    def insert_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100,
        split_by_folder: bool = True
    ) -> Dict[str, int]:
        """문서 삽입 (폴더별 컬렉션 분리 옵션)"""
        if split_by_folder:
            # 폴더별로 문서 그룹화
            folder_docs = {}
            for doc in documents:
                folder = doc.metadata.get('folder_name', 'root')
                if folder not in folder_docs:
                    folder_docs[folder] = []
                folder_docs[folder].append(doc)
            
            # 각 폴더별로 별도 컬렉션에 저장
            results = {}
            for folder_name, folder_documents in folder_docs.items():
                collection_name = self.config.get_collection_name(folder_name)
                count = self._insert_to_collection(
                    collection_name, 
                    folder_documents, 
                    batch_size
                )
                results[collection_name] = count
                print(f"[{folder_name}] -> Collection '{collection_name}': {count}개 저장")
            
            return results
        else:
            # 기존 방식: 단일 컬렉션에 모두 저장
            count = self._insert_to_collection(
                self.config.collection_name, 
                documents, 
                batch_size
            )
            return {self.config.collection_name: count}
    
    def _insert_to_collection(
        self,
        collection_name: str,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """특정 컬렉션에 문서 삽입"""
        print(f"📝 _insert_to_collection 시작: collection={collection_name}, documents={len(documents)}개")
        
        # 컬렉션 존재 확인 및 생성
        if not self.client.has_collection(collection_name):
            self._create_collection_by_name(collection_name)
        
        total_inserted = 0
        
        # 배치 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"  배치 {i//batch_size + 1}: {len(batch)}개 문서 처리 중...")
            
            # 텍스트 임베딩
            texts = [doc.page_content for doc in batch]
            print(f"  임베딩 생성 중... (texts={len(texts)}개)")
            embeddings = self.embeddings.embed_documents(texts)
            print(f"  임베딩 생성 완료: {len(embeddings)}개")
            
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
            
            print(f"  데이터 레코드 준비 완료: {len(data)}개")
            
            # 삽입
            print(f"  Milvus에 삽입 중...")
            result = self.client.insert(
                collection_name=collection_name,
                data=data
            )
            print(f"  삽입 결과: {result}")
            total_inserted += len(data)
        
        # 데이터 flush (디스크에 커밋)
        print(f"  데이터 flush 중...")
        self.client.flush(collection_name)
        print(f"✅ _insert_to_collection 완료: total_inserted={total_inserted}")
        return total_inserted
    
    def search(
        self,
        query: str,
        k: int = 3,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        search_all_collections: bool = False
    ) -> List[Dict[str, Any]]:
        """벡터 검색 (단일 컬렉션 또는 모든 컬렉션)"""
        if search_all_collections:
            # 모든 컬렉션에서 검색하고 결과 병합
            return self._search_all_collections(query, k, filter_expr, output_fields)
        else:
            # 특정 컬렉션에서만 검색
            coll_name = collection_name or self.config.collection_name
            return self._search_single_collection(coll_name, query, k, filter_expr, output_fields)
    
    def _search_single_collection(
        self,
        collection_name: str,
        query: str,
        k: int = 3,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """단일 컬렉션에서 검색"""
        # 컬렉션 로드 확인
        if not self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist")
            return []
        
        self.client.load_collection(collection_name)
        
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
            collection_name=collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=k,
            filter=filter_expr,
            output_fields=output_fields
        )
        
        return results[0] if results else []
    
    def _search_all_collections(
        self,
        query: str,
        k: int = 3,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """모든 컬렉션에서 검색하고 결과 병합"""
        all_results = []
        collections = self.list_collections()
        
        print(f"\\n{len(collections)}개 컬렉션에서 검색 중...")
        
        for coll_name in collections:
            results = self._search_single_collection(
                coll_name, query, k, filter_expr, output_fields
            )
            for hit in results:
                # 컬렉션 정보 추가
                hit['collection'] = coll_name
            all_results.extend(results)
        
        # 점수로 정렬하고 상위 k개만 반환
        all_results.sort(key=lambda x: x.get('distance', 0), reverse=True)
        return all_results[:k]
    
    def search_with_scores(
        self,
        query: str,
        k: int = 3,
        filter_expr: Optional[str] = None,
        collection_name: Optional[str] = None,
        search_all_collections: bool = False
    ) -> List[Tuple[Document, float]]:
        """점수와 함께 검색 (LangChain 호환 형식)"""
        results = self.search(
            query, 
            k=k, 
            filter_expr=filter_expr,
            collection_name=collection_name,
            search_all_collections=search_all_collections
        )
        
        doc_score_pairs = []
        for hit in results:
            metadata = {
                'chunk_id': hit['entity'].get('chunk_id', ''),
                'source': hit['entity'].get('source', ''),
                'filename': hit['entity'].get('filename', ''),
                'language': hit['entity'].get('language', ''),
                'chunk_index': hit['entity'].get('chunk_index', 0),
                'total_chunks': hit['entity'].get('total_chunks', 0),
                'chunk_size_chars': hit['entity'].get('chunk_size_chars', 0),
                'chunk_size_tokens': hit['entity'].get('chunk_size_tokens', 0),
            }
            # 컬렉션 정보 추가 (복수 컬렉션 검색 시)
            if 'collection' in hit:
                metadata['collection'] = hit['collection']
            
            doc = Document(
                page_content=hit['entity'].get('text', ''),
                metadata=metadata
            )
            # Milvus distance를 similarity score로 변환 (COSINE의 경우)
            score = hit['distance']
            doc_score_pairs.append((doc, score))
        
        return doc_score_pairs
    
    def list_collections(self) -> List[str]:
        """모든 컬렉션 목록 반환"""
        return self.client.list_collections()
    
    def get_all_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """모든 컬렉션의 통계 반환"""
        collections = self.list_collections()
        stats = {}
        
        for coll_name in collections:
            coll_stats = self.client.get_collection_stats(coll_name)
            stats[coll_name] = {
                "row_count": coll_stats.get("row_count", 0),
            }
        
        return stats
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """컬렉션 통계 조회"""
        coll_name = collection_name or self.config.collection_name
        
        if not self.client.has_collection(coll_name):
            return {"exists": False, "collection_name": coll_name}
        
        # 최신 통계를 위해 flush와 load 수행
        try:
            self.client.flush(coll_name)
            self.client.load_collection(coll_name)
        except Exception as e:
            print(f"⚠️ flush/load 경고: {e}")
        
        stats = self.client.get_collection_stats(coll_name)
        return {
            "exists": True,
            "collection_name": coll_name,
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
