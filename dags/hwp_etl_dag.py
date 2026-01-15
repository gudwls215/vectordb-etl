"""
VectorDB ETL Airflow DAG for HWP Documents
HWP 문서를 Milvus 벡터 DB에 저장하는 ETL 파이프라인
폴더별로 별도의 컬렉션에 저장

DAG 구조:
    extract_hwp_documents -> transform_to_chunks -> load_to_milvus_by_folder -> validate_quality

사용법:
    1. 이 파일을 Airflow dags 디렉토리에 복사
    2. VECTORDB_ETL_PATH 환경 변수 설정 (또는 기본 경로 사용)
    3. /hwp 디렉토리에 폴더별로 HWP 파일 배치
    4. Airflow UI에서 DAG 활성화
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List
import os
import pickle
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

# 프로젝트 경로 설정
VECTORDB_ETL_PATH = os.environ.get(
    "VECTORDB_ETL_PATH", 
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# 경로를 sys.path에 추가
import sys
if VECTORDB_ETL_PATH not in sys.path:
    sys.path.insert(0, VECTORDB_ETL_PATH)


# DAG 기본 인자
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def extract_hwp_documents(**context) -> str:
    """
    Extract 단계: HWP 파일에서 문서 추출
    
    Returns:
        저장된 문서 파일 경로
    """
    from modules import load_hwp_documents, get_config, DATA_DIR
    
    config = get_config()
    
    print(f"HWP 디렉토리: {config.hwp_dir}")
    
    # HWP 문서 로드
    documents = load_hwp_documents(
        directory=config.hwp_dir,
        recursive=True,
        config=config
    )
    
    print(f"로드된 문서 수: {len(documents)}")
    
    # 폴더별 문서 분류
    folder_documents = {}
    for doc in documents:
        folder_name = doc.metadata.get('folder_name', 'root')
        if folder_name not in folder_documents:
            folder_documents[folder_name] = []
        folder_documents[folder_name].append(doc)
    
    print("폴더별 문서 수:")
    for folder, docs in folder_documents.items():
        print(f"  - {folder}: {len(docs)}개")
    
    # 중간 결과 저장
    documents_path = os.path.join(DATA_DIR, "hwp_documents.pkl")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(documents_path, 'wb') as f:
        pickle.dump(documents, f)
    
    # XCom으로 경로 전달
    context['ti'].xcom_push(key='documents_path', value=documents_path)
    context['ti'].xcom_push(key='document_count', value=len(documents))
    context['ti'].xcom_push(key='folder_names', value=list(folder_documents.keys()))
    
    return documents_path


def transform_to_chunks(**context) -> str:
    """
    Transform 단계: 문서를 청크로 분할
    폴더별로 분류 유지
    
    Returns:
        저장된 청크 파일 경로
    """
    from modules import chunk_documents, get_config, DATA_DIR
    
    config = get_config()
    
    # 이전 단계 결과 로드
    documents_path = context['ti'].xcom_pull(
        key='documents_path', 
        task_ids='extract_hwp_documents'
    )
    
    with open(documents_path, 'rb') as f:
        documents = pickle.load(f)
    
    print(f"로드된 문서 수: {len(documents)}")
    
    # 청킹
    chunks = chunk_documents(
        documents,
        config=config.chunker,
        remove_duplicates=True,
        similarity_threshold=config.duplicate_similarity_threshold
    )
    
    print(f"생성된 청크 수: {len(chunks)}")
    
    # 폴더별 청크 분류
    folder_chunks = {}
    for chunk in chunks:
        folder_name = chunk.metadata.get('folder_name', 'root')
        if folder_name not in folder_chunks:
            folder_chunks[folder_name] = []
        folder_chunks[folder_name].append(chunk)
    
    print("폴더별 청크 수:")
    for folder, ch in folder_chunks.items():
        print(f"  - {folder}: {len(ch)}개")
    
    # 중간 결과 저장
    chunks_path = os.path.join(DATA_DIR, "hwp_chunks.pkl")
    
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    # XCom으로 경로 전달
    context['ti'].xcom_push(key='chunks_path', value=chunks_path)
    context['ti'].xcom_push(key='chunk_count', value=len(chunks))
    context['ti'].xcom_push(key='folder_chunk_counts', value={k: len(v) for k, v in folder_chunks.items()})
    
    return chunks_path


def load_to_milvus_by_folder(**context) -> Dict[str, Any]:
    """
    Load 단계: 폴더별로 별도의 Milvus 컬렉션에 저장
    
    컬렉션 이름 규칙: hwp_{폴더명}
    예: /hwp/contracts/ -> hwp_contracts
        /hwp/reports/   -> hwp_reports
    
    Returns:
        저장 결과 통계
    """
    from modules import get_config, DATA_DIR, MilvusVectorStore
    
    config = get_config()
    
    # 이전 단계 결과 로드
    chunks_path = context['ti'].xcom_pull(
        key='chunks_path', 
        task_ids='transform_to_chunks'
    )
    
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"로드된 청크 수: {len(chunks)}")
    
    # 폴더별 청크 분류
    folder_chunks = {}
    for chunk in chunks:
        folder_name = chunk.metadata.get('folder_name', 'root')
        if folder_name not in folder_chunks:
            folder_chunks[folder_name] = []
        folder_chunks[folder_name].append(chunk)
    
    # 폴더별로 별도 컬렉션에 저장
    results = {}
    
    for folder_name, folder_chunk_list in folder_chunks.items():
        # 컬렉션 이름 생성
        collection_name = f"hwp_{folder_name.lower().replace('-', '_').replace(' ', '_')}"
        
        print(f"\n폴더 '{folder_name}' -> 컬렉션 '{collection_name}'")
        print(f"  청크 수: {len(folder_chunk_list)}")
        
        # 해당 폴더용 벡터 저장소 생성
        vectorstore = MilvusVectorStore(
            collection_name=collection_name,
            uri=config.milvus.uri,
        )
        
        # 컬렉션 생성 (기존 삭제)
        vectorstore.create_collection(drop_existing=True)
        
        # 문서 삽입
        inserted_count = vectorstore.insert_documents(folder_chunk_list)
        
        # 통계 조회
        stats = vectorstore.get_collection_stats()
        
        results[folder_name] = {
            "collection_name": collection_name,
            "inserted_count": inserted_count,
            "total_vectors": stats.get("row_count"),
        }
        
        print(f"  저장 완료: {inserted_count}개")
    
    print(f"\n전체 저장 완료: {sum(r['inserted_count'] for r in results.values())}개")
    
    # XCom으로 결과 전달
    context['ti'].xcom_push(key='load_stats', value=results)
    
    return results


def validate_quality(**context) -> Dict[str, Any]:
    """
    Validate 단계: 폴더별 품질 검증
    
    Returns:
        검증 보고서
    """
    from modules import get_config, validate_pipeline, DATA_DIR, MilvusVectorStore
    
    config = get_config()
    
    # 청크 로드
    chunks_path = context['ti'].xcom_pull(
        key='chunks_path', 
        task_ids='transform_to_chunks'
    )
    
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # 저장 결과 로드
    load_stats = context['ti'].xcom_pull(
        key='load_stats', 
        task_ids='load_to_milvus_by_folder'
    )
    
    # 폴더별 검증
    reports = {}
    
    for folder_name, stats in load_stats.items():
        collection_name = stats['collection_name']
        
        # 해당 폴더의 청크만 필터링
        folder_chunks = [c for c in chunks if c.metadata.get('folder_name', 'root') == folder_name]
        
        # 벡터 저장소 연결
        vectorstore = MilvusVectorStore(
            collection_name=collection_name,
            uri=config.milvus.uri,
        )
        
        # 간단한 검증: 샘플 쿼리
        test_queries = ["내용 검색 테스트"]
        
        try:
            report = validate_pipeline(
                vectorstore=vectorstore,
                chunks=folder_chunks,
                test_queries=test_queries,
                sample_count=min(3, len(folder_chunks))
            )
            reports[folder_name] = report
        except Exception as e:
            print(f"폴더 '{folder_name}' 검증 실패: {e}")
            reports[folder_name] = {"error": str(e)}
    
    # 전체 보고서 저장
    report_path = os.path.join(
        DATA_DIR, 
        f"hwp_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    
    print(f"검증 보고서 저장: {report_path}")
    
    # XCom으로 결과 전달
    context['ti'].xcom_push(key='validation_reports', value=reports)
    context['ti'].xcom_push(key='report_path', value=report_path)
    
    return reports


def notify_completion(**context):
    """
    파이프라인 완료 알림
    """
    # 모든 단계의 결과 수집
    document_count = context['ti'].xcom_pull(
        key='document_count', 
        task_ids='extract_hwp_documents'
    )
    
    chunk_count = context['ti'].xcom_pull(
        key='chunk_count', 
        task_ids='transform_to_chunks'
    )
    
    folder_chunk_counts = context['ti'].xcom_pull(
        key='folder_chunk_counts', 
        task_ids='transform_to_chunks'
    )
    
    load_stats = context['ti'].xcom_pull(
        key='load_stats', 
        task_ids='load_to_milvus_by_folder'
    )
    
    # 요약 출력
    print("\n" + "=" * 60)
    print("📊 HWP VectorDB ETL 파이프라인 완료 요약")
    print("=" * 60)
    print(f"총 문서 수: {document_count}")
    print(f"총 청크 수: {chunk_count}")
    print("\n폴더별 컬렉션:")
    
    for folder_name, stats in load_stats.items():
        print(f"  📁 {folder_name}")
        print(f"     컬렉션: {stats['collection_name']}")
        print(f"     벡터 수: {stats['total_vectors']}")
    
    print("=" * 60)


# DAG 정의 - HWP 전체 파이프라인
with DAG(
    dag_id="vectordb_hwp_etl_pipeline",
    default_args=default_args,
    description="HWP 문서를 폴더별로 Milvus 벡터 DB에 저장하는 ETL 파이프라인",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "milvus", "hwp", "embedding"],
    doc_md=__doc__,
) as dag:
    
    # 시작 태스크
    start = EmptyOperator(task_id="start")
    
    # Extract 태스크
    extract_task = PythonOperator(
        task_id="extract_hwp_documents",
        python_callable=extract_hwp_documents,
        provide_context=True,
    )
    
    # Transform 태스크
    transform_task = PythonOperator(
        task_id="transform_to_chunks",
        python_callable=transform_to_chunks,
        provide_context=True,
    )
    
    # Load 태스크 (폴더별 컬렉션)
    load_task = PythonOperator(
        task_id="load_to_milvus_by_folder",
        python_callable=load_to_milvus_by_folder,
        provide_context=True,
    )
    
    # Validate 태스크
    validate_task = PythonOperator(
        task_id="validate_quality",
        python_callable=validate_quality,
        provide_context=True,
    )
    
    # 완료 알림 태스크
    notify_task = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
        provide_context=True,
    )
    
    # 종료 태스크
    end = EmptyOperator(task_id="end")
    
    # 태스크 의존성 정의
    start >> extract_task >> transform_task >> load_task >> validate_task >> notify_task >> end


# 개별 단계 실행을 위한 서브 DAG들
with DAG(
    dag_id="vectordb_hwp_extract_only",
    default_args=default_args,
    description="HWP 문서 추출만 실행",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "hwp", "extract"],
) as extract_dag:
    
    extract_only_task = PythonOperator(
        task_id="extract_hwp_documents",
        python_callable=extract_hwp_documents,
        provide_context=True,
    )


with DAG(
    dag_id="vectordb_hwp_load_only",
    default_args=default_args,
    description="HWP Milvus 저장만 실행 (transform 결과 필요)",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "hwp", "load"],
) as load_dag:
    
    load_only_task = PythonOperator(
        task_id="load_to_milvus_by_folder",
        python_callable=load_to_milvus_by_folder,
        provide_context=True,
    )
