"""
VectorDB ETL Airflow DAG
HTML 문서를 Milvus 벡터 DB에 저장하는 ETL 파이프라인

DAG 구조:
    extract_html_documents -> transform_to_chunks -> load_to_milvus -> validate_quality

사용법:
    1. 이 파일을 Airflow dags 디렉토리에 복사
    2. VECTORDB_ETL_PATH 환경 변수 설정 (또는 기본 경로 사용)
    3. Airflow UI에서 DAG 활성화
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


def extract_html_documents(**context) -> str:
    """
    Extract 단계: HTML 파일에서 문서 추출
    
    Returns:
        저장된 문서 파일 경로
    """
    from modules import load_html_documents, get_config, DATA_DIR
    
    config = get_config()
    
    print(f"HTML 디렉토리: {config.html_dir}")
    print(f"패턴: {config.html_glob_pattern}")
    
    # HTML 문서 로드
    documents = load_html_documents(
        directory=config.html_dir,
        glob_pattern=config.html_glob_pattern,
        config=config
    )
    
    print(f"로드된 문서 수: {len(documents)}")
    
    # 중간 결과 저장
    documents_path = os.path.join(DATA_DIR, "documents.pkl")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(documents_path, 'wb') as f:
        pickle.dump(documents, f)
    
    # XCom으로 경로 전달
    context['ti'].xcom_push(key='documents_path', value=documents_path)
    context['ti'].xcom_push(key='document_count', value=len(documents))
    
    return documents_path


def transform_to_chunks(**context) -> str:
    """
    Transform 단계: 문서를 청크로 분할
    
    Returns:
        저장된 청크 파일 경로
    """
    from modules import chunk_documents, get_config, DATA_DIR
    
    config = get_config()
    
    # 이전 단계 결과 로드
    documents_path = context['ti'].xcom_pull(
        key='documents_path', 
        task_ids='extract_html_documents'
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
    
    # 중간 결과 저장
    chunks_path = os.path.join(DATA_DIR, "chunks.pkl")
    
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    # XCom으로 경로 전달
    context['ti'].xcom_push(key='chunks_path', value=chunks_path)
    context['ti'].xcom_push(key='chunk_count', value=len(chunks))
    
    return chunks_path


def load_to_milvus(**context) -> Dict[str, Any]:
    """
    Load 단계: Milvus에 벡터 저장
    
    Returns:
        저장 결과 통계
    """
    from modules import get_vector_store, DATA_DIR
    
    # 이전 단계 결과 로드
    chunks_path = context['ti'].xcom_pull(
        key='chunks_path', 
        task_ids='transform_to_chunks'
    )
    
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"로드된 청크 수: {len(chunks)}")
    
    # Milvus에 저장
    vectorstore = get_vector_store()
    vectorstore.create_collection(drop_existing=True)
    inserted_count = vectorstore.insert_documents(chunks)
    
    # 통계 조회
    stats = vectorstore.get_collection_stats()
    
    result = {
        "inserted_count": inserted_count,
        "collection_name": stats.get("collection_name"),
        "total_vectors": stats.get("row_count"),
    }
    
    print(f"저장 완료: {result}")
    
    # XCom으로 결과 전달
    context['ti'].xcom_push(key='load_stats', value=result)
    
    return result


def validate_quality(**context) -> Dict[str, Any]:
    """
    Validate 단계: 품질 검증
    
    Returns:
        검증 보고서
    """
    from modules import get_vector_store, validate_pipeline, DATA_DIR
    
    # 청크 로드
    chunks_path = context['ti'].xcom_pull(
        key='chunks_path', 
        task_ids='transform_to_chunks'
    )
    
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    # 벡터 저장소
    vectorstore = get_vector_store()
    
    # 테스트 쿼리
    test_queries = [
        "서울 사무실 주소",
        "수강신청방법",
    ]
    
    # 검증
    report = validate_pipeline(
        vectorstore=vectorstore,
        chunks=chunks,
        test_queries=test_queries,
        sample_count=3
    )
    
    # 보고서 저장
    report_path = os.path.join(
        DATA_DIR, 
        f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"검증 보고서 저장: {report_path}")
    
    # XCom으로 결과 전달
    context['ti'].xcom_push(key='validation_report', value=report)
    context['ti'].xcom_push(key='report_path', value=report_path)
    
    return report


def notify_completion(**context):
    """
    파이프라인 완료 알림
    """
    # 모든 단계의 결과 수집
    document_count = context['ti'].xcom_pull(
        key='document_count', 
        task_ids='extract_html_documents'
    )
    
    chunk_count = context['ti'].xcom_pull(
        key='chunk_count', 
        task_ids='transform_to_chunks'
    )
    
    load_stats = context['ti'].xcom_pull(
        key='load_stats', 
        task_ids='load_to_milvus'
    )
    
    validation_report = context['ti'].xcom_pull(
        key='validation_report', 
        task_ids='validate_quality'
    )
    
    # 요약 출력
    print("\n" + "=" * 60)
    print("📊 VectorDB ETL 파이프라인 완료 요약")
    print("=" * 60)
    print(f"문서 수: {document_count}")
    print(f"청크 수: {chunk_count}")
    print(f"저장된 벡터 수: {load_stats.get('total_vectors', 'N/A')}")
    print(f"목표 범위 내 청크 비율: {validation_report.get('in_range_ratio', 'N/A'):.1f}%")
    print("=" * 60)
    
    # 여기에 Slack, 이메일 등 알림 로직 추가 가능


# DAG 정의
with DAG(
    dag_id="vectordb_etl_pipeline",
    default_args=default_args,
    description="HTML 문서를 Milvus 벡터 DB에 저장하는 ETL 파이프라인",
    schedule_interval="@daily",  # 매일 실행 (필요에 따라 조정)
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "milvus", "embedding"],
    doc_md=__doc__,
) as dag:
    
    # 시작 태스크
    start = EmptyOperator(task_id="start")
    
    # Extract 태스크
    extract_task = PythonOperator(
        task_id="extract_html_documents",
        python_callable=extract_html_documents,
        provide_context=True,
    )
    
    # Transform 태스크
    transform_task = PythonOperator(
        task_id="transform_to_chunks",
        python_callable=transform_to_chunks,
        provide_context=True,
    )
    
    # Load 태스크
    load_task = PythonOperator(
        task_id="load_to_milvus",
        python_callable=load_to_milvus,
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
# 필요시 별도 DAG로 분리하여 독립 실행 가능

with DAG(
    dag_id="vectordb_etl_extract_only",
    default_args=default_args,
    description="HTML 문서 추출만 실행",
    schedule_interval=None,  # 수동 실행
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "extract"],
) as extract_dag:
    
    extract_only_task = PythonOperator(
        task_id="extract_html_documents",
        python_callable=extract_html_documents,
        provide_context=True,
    )


with DAG(
    dag_id="vectordb_etl_transform_only",
    default_args=default_args,
    description="문서 청킹만 실행 (extract 결과 필요)",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "transform"],
) as transform_dag:
    
    transform_only_task = PythonOperator(
        task_id="transform_to_chunks",
        python_callable=transform_to_chunks,
        provide_context=True,
    )


with DAG(
    dag_id="vectordb_etl_load_only",
    default_args=default_args,
    description="Milvus 저장만 실행 (transform 결과 필요)",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "load"],
) as load_dag:
    
    load_only_task = PythonOperator(
        task_id="load_to_milvus",
        python_callable=load_to_milvus,
        provide_context=True,
    )


with DAG(
    dag_id="vectordb_etl_validate_only",
    default_args=default_args,
    description="품질 검증만 실행",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["vectordb", "etl", "validate"],
) as validate_dag:
    
    validate_only_task = PythonOperator(
        task_id="validate_quality",
        python_callable=validate_quality,
        provide_context=True,
    )
