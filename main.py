#!/usr/bin/env python
"""
VectorDB ETL Pipeline CLI
각 단계를 독립적으로 실행할 수 있는 CLI 스크립트

Usage:
    # 전체 파이프라인
    python main.py --stage all
    
    # 단계별 실행
    python main.py --stage extract
    python main.py --stage transform
    python main.py --stage load
    python main.py --stage validate
    
    # 검색 테스트
    python main.py --stage search --query "서울 사무실 주소"
    
    # 벡터 DB 초기화
    python main.py --stage reset --confirm
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import (
    PipelineConfig,
    get_config,
    create_config,
    load_html_documents,
    chunk_documents,
    get_vector_store,
    reset_vector_store,
    get_embeddings,
    validate_pipeline,
    search_with_scores,
    print_search_results,
    create_rag_prompt,
    DATA_DIR,
)


class PipelineRunner:
    """ETL 파이프라인 실행기"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 중간 결과 저장 경로
        self.documents_path = self.data_dir / "documents.pkl"
        self.chunks_path = self.data_dir / "chunks.pkl"
        
    def extract(self) -> List:
        """
        Extract 단계: HTML 파일에서 문서 추출
        """
        print("\n" + "=" * 60)
        print("📂 EXTRACT: HTML 파일 로드")
        print("=" * 60)
        
        documents = load_html_documents(
            directory=self.config.html_dir,
            glob_pattern=self.config.html_glob_pattern,
            config=self.config
        )
        
        print(f"\n로드된 문서 수: {len(documents)}")
        
        if documents:
            print(f"\n첫 번째 문서 메타데이터:")
            for key, value in documents[0].metadata.items():
                print(f"  {key}: {value}")
        
        # 중간 결과 저장
        with open(self.documents_path, 'wb') as f:
            pickle.dump(documents, f)
        print(f"\n문서 저장 완료: {self.documents_path}")
        
        return documents
    
    def transform(self, documents: Optional[List] = None) -> List:
        """
        Transform 단계: 문서를 청크로 분할
        """
        print("\n" + "=" * 60)
        print("🔄 TRANSFORM: 문서 청킹")
        print("=" * 60)
        
        # 이전 단계 결과 로드
        if documents is None:
            if self.documents_path.exists():
                with open(self.documents_path, 'rb') as f:
                    documents = pickle.load(f)
                print(f"저장된 문서 로드: {len(documents)}개")
            else:
                raise FileNotFoundError(
                    f"문서 파일을 찾을 수 없습니다: {self.documents_path}\n"
                    "먼저 extract 단계를 실행하세요."
                )
        
        # 청킹
        chunks = chunk_documents(
            documents,
            config=self.config.chunker,
            remove_duplicates=True,
            similarity_threshold=self.config.duplicate_similarity_threshold
        )
        
        if chunks:
            print(f"\n샘플 청크 메타데이터:")
            for key, value in chunks[0].metadata.items():
                print(f"  {key}: {value}")
        
        # 중간 결과 저장
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"\n청크 저장 완료: {self.chunks_path}")
        
        return chunks
    
    def load(self, chunks: Optional[List] = None) -> None:
        """
        Load 단계: Milvus에 벡터 저장
        """
        print("\n" + "=" * 60)
        print("💾 LOAD: Milvus 벡터 저장")
        print("=" * 60)
        
        # 이전 단계 결과 로드
        if chunks is None:
            if self.chunks_path.exists():
                with open(self.chunks_path, 'rb') as f:
                    chunks = pickle.load(f)
                print(f"저장된 청크 로드: {len(chunks)}개")
            else:
                raise FileNotFoundError(
                    f"청크 파일을 찾을 수 없습니다: {self.chunks_path}\n"
                    "먼저 transform 단계를 실행하세요."
                )
        
        # Milvus에 저장
        vectorstore = get_vector_store()
        vectorstore.create_collection(drop_existing=True)
        vectorstore.insert_documents(chunks)
        
        # 통계 출력
        stats = vectorstore.get_collection_stats()
        print(f"\n저장 완료:")
        print(f"  컬렉션: {stats.get('collection_name', 'N/A')}")
        print(f"  벡터 수: {stats.get('row_count', 'N/A')}")
    
    def validate(self, chunks: Optional[List] = None) -> dict:
        """
        Validate 단계: 품질 검증
        """
        print("\n" + "=" * 60)
        print("✅ VALIDATE: 품질 검증")
        print("=" * 60)
        
        # 청크 로드
        if chunks is None:
            if self.chunks_path.exists():
                with open(self.chunks_path, 'rb') as f:
                    chunks = pickle.load(f)
                print(f"저장된 청크 로드: {len(chunks)}개")
            else:
                raise FileNotFoundError(
                    f"청크 파일을 찾을 수 없습니다: {self.chunks_path}\n"
                    "먼저 transform 단계를 실행하세요."
                )
        
        # 벡터 저장소
        vectorstore = get_vector_store()
        
        # 테스트 쿼리
        test_queries = [
            "서울 사무실 주소",
            "수강신청방법",
            "Seoul office address",
        ]
        
        # 검증
        report = validate_pipeline(
            vectorstore=vectorstore,
            chunks=chunks,
            test_queries=test_queries,
            sample_count=3
        )
        
        # 보고서 저장
        report_path = self.data_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n검증 보고서 저장: {report_path}")
        
        return report
    
    def search(self, query: str, k: int = 3, language: Optional[str] = None) -> None:
        """
        검색 테스트
        """
        print("\n" + "=" * 60)
        print("🔍 SEARCH: 검색 테스트")
        print("=" * 60)
        
        print(f"\n쿼리: '{query}'")
        if language:
            print(f"언어 필터: {language}")
        
        results = search_with_scores(
            query=query,
            k=k,
            filter_language=language,
            auto_detect_language=(language is None)
        )
        
        print_search_results(results)
    
    def reset(self, confirm: bool = False) -> None:
        """
        벡터 DB 초기화
        """
        print("\n" + "=" * 60)
        print("🗑️ RESET: 벡터 DB 초기화")
        print("=" * 60)
        
        vectorstore = get_vector_store()
        stats = vectorstore.get_collection_stats()
        
        if stats.get('exists'):
            print(f"\n컬렉션: {stats.get('collection_name')}")
            print(f"벡터 수: {stats.get('row_count', 0)}")
            
            if confirm:
                vectorstore.drop_collection()
                print("\n✅ 컬렉션이 삭제되었습니다.")
            else:
                print("\n⚠️ 삭제하려면 --confirm 옵션을 추가하세요.")
        else:
            print("ℹ️ 삭제할 컬렉션이 없습니다.")
        
        # 중간 파일 삭제
        if confirm:
            for path in [self.documents_path, self.chunks_path]:
                if path.exists():
                    path.unlink()
                    print(f"삭제됨: {path}")
    
    def run_all(self) -> None:
        """
        전체 파이프라인 실행
        """
        print("\n" + "=" * 60)
        print("🚀 전체 파이프라인 실행")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Extract
        documents = self.extract()
        
        # Transform
        chunks = self.transform(documents)
        
        # Load
        self.load(chunks)
        
        # Validate
        self.validate(chunks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print(f"✅ 파이프라인 완료! (소요 시간: {duration:.1f}초)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="VectorDB ETL Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 전체 파이프라인
    python main.py --stage all
    
    # 단계별 실행
    python main.py --stage extract
    python main.py --stage transform
    python main.py --stage load
    python main.py --stage validate
    
    # 검색 테스트
    python main.py --stage search --query "서울 사무실 주소"
    python main.py --stage search --query "address" --language english
    
    # 벡터 DB 초기화
    python main.py --stage reset --confirm
        """
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["all", "extract", "transform", "load", "validate", "search", "reset"],
        help="실행할 파이프라인 단계"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="검색 쿼리 (search 단계에서 사용)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        choices=["korean", "english", "vietnamese"],
        help="검색 언어 필터 (search 단계에서 사용)"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="검색 결과 수 (기본값: 3)"
    )
    
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="초기화 확인 (reset 단계에서 사용)"
    )
    
    parser.add_argument(
        "--html-dir",
        type=str,
        help="HTML 파일 디렉토리 경로"
    )
    
    parser.add_argument(
        "--milvus-uri",
        type=str,
        help="Milvus URI (기본값: 로컬 파일)"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="html_documents",
        help="Milvus 컬렉션 이름 (기본값: html_documents)"
    )
    
    args = parser.parse_args()
    
    # 설정 생성
    config_kwargs = {
        "milvus_uri": args.milvus_uri,
        "collection_name": args.collection,
    }
    if args.html_dir:
        config_kwargs["html_dir"] = args.html_dir
    
    config = create_config(**config_kwargs)
    
    # 파이프라인 실행기 생성
    runner = PipelineRunner(config)
    
    # 단계 실행
    if args.stage == "all":
        runner.run_all()
    elif args.stage == "extract":
        runner.extract()
    elif args.stage == "transform":
        runner.transform()
    elif args.stage == "load":
        runner.load()
    elif args.stage == "validate":
        runner.validate()
    elif args.stage == "search":
        if not args.query:
            parser.error("--query 옵션이 필요합니다.")
        runner.search(args.query, k=args.k, language=args.language)
    elif args.stage == "reset":
        runner.reset(confirm=args.confirm)


if __name__ == "__main__":
    main()
