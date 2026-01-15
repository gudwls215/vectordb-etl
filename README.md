# VectorDB ETL Pipeline

HTML 문서를 BGE-M3 임베딩으로 변환하여 Milvus 벡터 DB에 저장하는 ETL 파이프라인입니다.

## 📁 프로젝트 구조

```
vectordb-etl/
├── modules/
│   ├── __init__.py           # 모듈 초기화 및 exports
│   ├── config.py             # 설정 관리 (Milvus, Chunker, Embedding)
│   ├── embeddings.py         # BGE-M3 임베딩 클래스
│   ├── text_cleaner.py       # 텍스트 정제
│   ├── html_loader.py        # HTML/JSP 파일 로더 (재귀 로딩 지원)
│   ├── chunker.py            # 의미 기반 문서 분할
│   ├── milvus_store.py       # Milvus 벡터 저장소 (폴더별 컬렉션 분리)
│   ├── quality_monitor.py    # 품질 검증
│   └── search_utils.py       # 검색 유틸리티 (다중 컬렉션 지원)
├── dags/
│   └── vectordb_etl_dag.py   # Airflow DAG
├── html/                      # HTML/JSP 소스 파일
│   ├── lms/                   # LMS 관련 파일 → docs_lms 컬렉션
│   ├── compa/                 # 회사 관련 파일 → docs_compa 컬렉션
│   └── ...                    # 기타 폴더 → 각각의 컬렉션
├── data/                      # 중간 결과 및 DB 파일
├── main.py                    # CLI 실행 스크립트
├── requirements.txt           # 의존성
└── README.md                  # 문서
```

## ✨ 주요 기능

### 1. 폴더별 컬렉션 자동 분리
- `html/` 하위의 각 폴더가 별도의 Milvus 컬렉션으로 저장됩니다
- 예: `html/lms/` → `docs_lms` 컬렉션
- 예: `html/compa/` → `docs_compa` 컬렉션

### 2. HTML/JSP 파일 통합 지원
- HTML 파일뿐만 아니라 JSP 파일도 자동으로 로드
- 재귀적으로 하위 폴더 탐색

### 3. 다중 컬렉션 검색
- 모든 컬렉션에서 동시 검색 가능
- 특정 컬렉션만 지정하여 검색 가능
- 언어별 자동 필터링 지원

## 🚀 설치

### uv를 사용한 설치 (권장)

```bash
# uv 설치 (아직 설치하지 않은 경우)
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 pip로 설치
pip install uv

# 프로젝트 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --all-extras

# Airflow 의존성 포함 설치
uv sync --extra airflow
```

### 의존성 관리

```bash
# 의존성 추가
uv add package-name

# 개발 의존성 추가
uv add --dev package-name

# 의존성 제거
uv remove package-name

# 의존성 업데이트
uv sync --upgrade

# lock 파일 생성/업데이트
uv lock
```

### 기존 pip 방식 (레거시)

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -e .
```

## 💻 CLI 사용법

### 전체 파이프라인 실행

```bash
uv run python main.py --stage all

# HWP 전체 파이프라인
uv run python main.py --stage all-hwp
```

### 단계별 실행

```bash
# 1. Extract: HTML 파일 로드
uv run python main.py --stage extract

# 2. Transform: 문서 청킹
uv run python main.py --stage transform

# 3. Load: Milvus에 저장
uv run python main.py --stage load

# 4. Validate: 품질 검증
uv run python main.py --stage validate

# HWP 단계별 실행
uv run python main.py --stage extract-hwp
uv run python main.py --stage transform-hwp
uv run python main.py --stage load-hwp
```

### 검색 테스트

```bash
# 모든 컬렉션에서 검색 (언어 자동 감지)
uv run python main.py --stage search --query "서울 사무실 주소"

# 특정 컬렉션에서만 검색
uv run python main.py --stage search --query "딥사이언스 창업 활성화" --collection hwp_compa

# 언어 필터 지정
uv run python main.py --stage search --query "Seoul office address" --language english

# 결과 수 지정
uv run python main.py --stage search --query "수강신청" --k 5
```

### 컬렉션 목록 확인

```bash
# Python으로 컬렉션 확인
uv run python test/check_collections.py
```

### 벡터 DB 초기화

```bash
# 확인 후 삭제
uv run python main.py --stage reset --confirm
```

### 추가 옵션

```bash
# HTML 디렉토리 지정
uv run python main.py --stage all --html-dir /path/to/html

# Milvus URI 지정 (서버 연결)
uv run python main.py --stage all --milvus-uri "http://localhost:19530"

# 컬렉션 이름 지정
uv run python main.py --stage all --collection my_collection
```

## 🔄 Airflow 연동

### DAG 설정

1. `dags/vectordb_etl_dag.py`를 Airflow dags 디렉토리에 복사
2. 환경 변수 설정:
   ```bash
   export VECTORDB_ETL_PATH=/path/to/vectordb-etl
   ```
3. Airflow UI에서 `vectordb_etl_pipeline` DAG 활성화

### 제공되는 DAG

| DAG ID | 설명 | 스케줄 |
|--------|------|--------|
| `vectordb_etl_pipeline` | 전체 파이프라인 | @daily |
| `vectordb_etl_extract_only` | Extract만 실행 | 수동 |
| `vectordb_etl_transform_only` | Transform만 실행 | 수동 |
| `vectordb_etl_load_only` | Load만 실행 | 수동 |
| `vectordb_etl_validate_only` | Validate만 실행 | 수동 |

## 📊 파이프라인 단계

### 1. Extract (추출)
- HTML 파일 로드
- 구조 정보 추출 (제목, 헤딩, 테이블 등)
- 텍스트 정제 (템플릿 태그, 이모지, JS 코드 제거)
- 언어 감지 (파일명 기반)

### 2. Transform (변환)
- 의미 기반 청킹 (RecursiveCharacterTextSplitter)
- 토큰 수 추정 (한글/영어 고려)
- 중복 제거 (해시 기반)
- 메타데이터 보강

### 3. Load (적재)
- BGE-M3 임베딩 생성
- Milvus 컬렉션 생성/갱신
- 배치 단위 벡터 삽입
- 인덱스 생성 (IVF_FLAT/HNSW)

### 4. Validate (검증)
- 청크 크기 분포 분석
- 메타데이터 분석
- 검색 품질 테스트
- 종합 보고서 생성

## 🔧 설정

### 기본 설정 (modules/config.py)

```python
# Milvus 설정
MilvusConfig(
    uri="./data/milvus_vectordb.db",  # 로컬 파일 (Milvus Lite)
    collection_name="html_documents",
    index_type="IVF_FLAT",
    metric_type="COSINE",
)

# 임베딩 설정
EmbeddingConfig(
    model_name="BAAI/bge-m3",
    dimension=1024,
    batch_size=32,
)

# 청커 설정
ChunkerConfig(
    target_chunk_size=800,   # 문자 기준
    chunk_overlap=150,
)
```

### 커스텀 설정 사용

```python
from modules import create_config, PipelineRunner

config = create_config(
    milvus_uri="http://localhost:19530",  # Milvus 서버
    collection_name="my_docs",
    chunk_size=1000,
    chunk_overlap=200,
)

runner = PipelineRunner(config)
runner.run_all()
```

## 🔍 검색 API

```python
from modules import search_with_scores, create_rag_prompt

# 기본 검색
results = search_with_scores("서울 사무실 주소", k=3)
for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:200]}...")

# 언어 필터링
results = search_with_scores(
    "course information", 
    k=5, 
    filter_language="english"
)

# RAG 프롬프트 생성
messages = create_rag_prompt("수료 기준은 무엇인가요?")
# -> OpenAI API 등에 전달
```

## 📈 품질 모니터링

```python
from modules import QualityMonitor, get_vector_store

vectorstore = get_vector_store()
monitor = QualityMonitor(vectorstore, chunks)

# 분포 분석
monitor.analyze_chunk_distribution()
monitor.plot_distribution()  # matplotlib 필요

# 메타데이터 분석
monitor.analyze_metadata()

# 검색 품질 테스트
monitor.test_search_quality(["테스트 쿼리1", "테스트 쿼리2"])

# 종합 보고서
report = monitor.generate_report()
```

## 🐳 Docker 지원 (선택사항)

PowerShell 또는 Windows 명령 프롬프트에서 다음 명령을 실행하여 Milvus Standalone용 Docker Compose 구성 파일을 다운로드하고 Milvus를 시작

```bash
# Download the configuration file and rename it as docker-compose.yml
C:\>Invoke-WebRequest https://github.com/milvus-io/milvus/releases/download/v2.6.8/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml

# Start Milvus
C:\>docker compose up -d
Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```

Milvus 서버 실행:

```bash
# Docker Compose로 Milvus 실행
docker-compose up -d

# 또는 단독 실행
docker run -d --name milvus \
    -p 19530:19530 \
    -p 9091:9091 \
    milvusdb/milvus:latest
```


## � 개발 도구

프로젝트는 다음 도구를 사용합니다:

- **uv**: 빠른 Python 패키지 관리자
- **black**: 코드 포맷터
- **ruff**: 빠른 린터
- **pytest**: 테스트 프레임워크

```bash
# 코드 포맷팅
uv run black modules/ main.py

# 린팅
uv run ruff check modules/ main.py

# 테스트 실행
uv run pytest

# 커버리지 포함 테스트
uv run pytest --cov=modules
```

## �📝 라이선스

MIT License
