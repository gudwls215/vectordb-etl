"""
Search Utilities Module
"""
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document

from .config import PipelineConfig, get_config
from .milvus_store import MilvusVectorStore, get_vector_store


def detect_language(text: str) -> str:
    """텍스트의 언어 감지 (간단한 휴리스틱)"""
    korean_chars = len(re.findall(r'[가-힣]', text))
    vietnamese_chars = len(re.findall(
        r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', 
        text.lower()
    ))
    
    total_chars = len(text)
    if total_chars == 0:
        return 'english'
    
    korean_ratio = korean_chars / total_chars
    vietnamese_ratio = vietnamese_chars / total_chars
    
    if korean_ratio > 0.1:
        return 'korean'
    elif vietnamese_ratio > 0.05:
        return 'vietnamese'
    else:
        return 'english'


def search(
    query: str, 
    k: int = 3, 
    filter_language: Optional[str] = None, 
    auto_detect_language: bool = True,
    collection_name: Optional[str] = None,
    search_all_collections: bool = False,
    vectorstore: Optional[MilvusVectorStore] = None,
    config: Optional[PipelineConfig] = None
) -> List[Document]:
    """벡터 DB 검색 (언어 자동 감지 옵션)"""
    config = config or get_config()
    vs = vectorstore or get_vector_store()
    
    # 언어 자동 감지
    if auto_detect_language and filter_language is None:
        filter_language = detect_language(query)
        print(f"자동 감지된 언어: {filter_language}")
    
    # 필터 표현식 생성
    filter_expr = f'language == "{filter_language}"' if filter_language else None
    
    # 검색
    results = vs.search_with_scores(
        query, 
        k=k, 
        filter_expr=filter_expr,
        collection_name=collection_name,
        search_all_collections=search_all_collections
    )
    
    return [doc for doc, _ in results]


def search_with_scores(
    query: str, 
    k: int = 3, 
    filter_language: Optional[str] = None, 
    auto_detect_language: bool = True,
    collection_name: Optional[str] = None,
    search_all_collections: bool = False,
    vectorstore: Optional[MilvusVectorStore] = None,
    config: Optional[PipelineConfig] = None
) -> List[Tuple[Document, float]]:
    """점수와 함께 검색"""
    config = config or get_config()
    vs = vectorstore or get_vector_store()
    
    # 언어 자동 감지
    if auto_detect_language and filter_language is None:
        filter_language = detect_language(query)
        print(f"자동 감지된 언어: {filter_language}")
    
    # 필터 표현식 생성
    filter_expr = f'language == "{filter_language}"' if filter_language else None
    
    return vs.search_with_scores(
        query, 
        k=k, 
        filter_expr=filter_expr,
        collection_name=collection_name,
        search_all_collections=search_all_collections
    )


def create_rag_prompt(
    query: str, 
    k: int = 3, 
    auto_detect_language: bool = True,
    vectorstore: Optional[MilvusVectorStore] = None
) -> List[Dict[str, str]]:
    """RAG 기반 프롬프트 생성 (언어 자동 감지)"""
    results = search(
        query, 
        k=k, 
        auto_detect_language=auto_detect_language,
        vectorstore=vectorstore
    )
    
    # 검색된 문서 출력
    print("검색된 문서:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.metadata.get('filename', 'N/A')} ({doc.metadata.get('language', 'N/A')})")
        print(f"{doc.page_content[:300]}...")
    
    # 시스템 메시지 구성
    context = "\n\n".join([f"문서 {i+1}: {doc.page_content}" for i, doc in enumerate(results)])
    
    system_message = f"""당신은 훌륭한 상담원입니다. 아래 문서들은 질문과 관련된 참고 자료입니다.

{context}

위 문서들을 참고하여 질문에 답변해 주세요.
반드시 한국어로 답변해 주세요."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    return messages


def print_search_results(results: List[Tuple[Document, float]]) -> None:
    """검색 결과 출력"""
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Source: {doc.metadata.get('filename', 'N/A')}")
        print(f"Language: {doc.metadata.get('language', 'N/A')}")
        print(f"Content: {doc.page_content[:300]}...")
