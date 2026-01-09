"""
Quality Monitoring Module
"""
import random
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np

from langchain_core.documents import Document

from .milvus_store import MilvusVectorStore


class QualityMonitor:
    """벡터 DB 품질 검증 및 모니터링 클래스"""
    
    def __init__(
        self, 
        vectorstore: MilvusVectorStore, 
        chunks: List[Document]
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        
    def analyze_chunk_distribution(self) -> Tuple[List[int], List[int]]:
        """청크 크기 분포 분석"""
        char_sizes = [len(c.page_content) for c in self.chunks]
        token_sizes = [c.metadata.get('chunk_size_tokens', 0) for c in self.chunks]
        
        print("=" * 50)
        print("📊 청크 크기 분포 분석")
        print("=" * 50)
        print(f"\n[문자 수 기준]")
        print(f"  총 청크 수: {len(char_sizes)}")
        print(f"  최소: {min(char_sizes)}")
        print(f"  최대: {max(char_sizes)}")
        print(f"  평균: {np.mean(char_sizes):.1f}")
        print(f"  중앙값: {np.median(char_sizes):.1f}")
        print(f"  표준편차: {np.std(char_sizes):.1f}")
        
        print(f"\n[토큰 수 기준 (추정)]")
        print(f"  최소: {min(token_sizes)}")
        print(f"  최대: {max(token_sizes)}")
        print(f"  평균: {np.mean(token_sizes):.1f}")
        print(f"  중앙값: {np.median(token_sizes):.1f}")
        
        return char_sizes, token_sizes
    
    def plot_distribution(self, char_sizes: List[int], token_sizes: List[int]) -> None:
        """청크 크기 분포 시각화"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].hist(char_sizes, bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Characters')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Chunk Size Distribution (Characters)')
            axes[0].axvline(np.mean(char_sizes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(char_sizes):.0f}')
            axes[0].legend()
            
            axes[1].hist(token_sizes, bins=30, edgecolor='black', alpha=0.7, color='green')
            axes[1].set_xlabel('Tokens (Estimated)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Chunk Size Distribution (Tokens)')
            axes[1].axvline(np.mean(token_sizes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(token_sizes):.0f}')
            axes[1].axvline(300, color='blue', linestyle=':', label='Target Range: 300-500')
            axes[1].axvline(500, color='blue', linestyle=':')
            axes[1].legend()
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib가 설치되지 않아 시각화를 건너뜁니다.")
    
    def analyze_metadata(self) -> Tuple[Counter, Counter]:
        """메타데이터 분석"""
        print("\n" + "=" * 50)
        print("📋 메타데이터 분석")
        print("=" * 50)
        
        # 언어 분포
        languages = [c.metadata.get('language', 'unknown') for c in self.chunks]
        lang_counts = Counter(languages)
        print(f"\n[언어 분포]")
        for lang, count in lang_counts.most_common():
            print(f"  {lang}: {count} ({count/len(self.chunks)*100:.1f}%)")
        
        # 소스 파일 분포
        sources = [c.metadata.get('filename', 'unknown') for c in self.chunks]
        source_counts = Counter(sources)
        print(f"\n[소스 파일별 청크 수]")
        for source, count in source_counts.most_common(10):
            print(f"  {source}: {count}")
            
        return lang_counts, source_counts
    
    def test_search_quality(self, test_queries: List[str], k: int = 3) -> None:
        """검색 품질 테스트"""
        print("\n" + "=" * 50)
        print("🔍 검색 품질 테스트")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\n쿼리: '{query}'")
            print("-" * 40)
            
            results = self.vectorstore.search_with_scores(query, k=k)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n  [{i}] 유사도 점수: {score:.4f}")
                print(f"      소스: {doc.metadata.get('filename', 'N/A')}")
                print(f"      언어: {doc.metadata.get('language', 'N/A')}")
                print(f"      내용: {doc.page_content[:150]}...")
    
    def test_search_with_language_filter(
        self, 
        test_queries: List[Tuple[str, str]], 
        k: int = 3
    ) -> None:
        """언어별 필터링을 포함한 검색 품질 테스트"""
        print("\n" + "=" * 50)
        print("🔍 검색 품질 테스트 (언어 필터링)")
        print("=" * 50)
        
        for query, lang in test_queries:
            print(f"\n쿼리: '{query}' (언어: {lang})")
            print("-" * 40)
            
            # 언어 필터링 적용
            filter_expr = f'language == "{lang}"' if lang else None
            results = self.vectorstore.search_with_scores(query, k=k, filter_expr=filter_expr)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n  [{i}] 유사도 점수: {score:.4f}")
                print(f"      소스: {doc.metadata.get('filename', 'N/A')}")
                print(f"      언어: {doc.metadata.get('language', 'N/A')}")
                print(f"      내용: {doc.page_content[:200]}...")
    
    def sample_chunks_review(self, n: int = 5) -> None:
        """샘플 청크 검토"""
        print("\n" + "=" * 50)
        print("📝 샘플 청크 검토")
        print("=" * 50)
        
        sample_indices = random.sample(range(len(self.chunks)), min(n, len(self.chunks)))
        
        for i, idx in enumerate(sample_indices, 1):
            chunk = self.chunks[idx]
            print(f"\n[샘플 {i}]")
            print(f"  소스: {chunk.metadata.get('filename', 'N/A')}")
            print(f"  청크 인덱스: {chunk.metadata.get('chunk_index', 'N/A')}/{chunk.metadata.get('total_chunks', 'N/A')}")
            print(f"  크기: {len(chunk.page_content)} chars / {chunk.metadata.get('chunk_size_tokens', 'N/A')} tokens")
            print(f"  내용:\n{chunk.page_content[:300]}...")
            print("-" * 40)
    
    def generate_report(self) -> Dict[str, Any]:
        """종합 보고서 생성"""
        print("\n" + "=" * 60)
        print("📈 벡터 DB 품질 종합 보고서")
        print("=" * 60)
        
        stats = self.vectorstore.get_collection_stats()
        
        print(f"\n[기본 통계]")
        print(f"  총 청크 수: {len(self.chunks)}")
        print(f"  벡터 DB 문서 수: {stats.get('row_count', 'N/A')}")
        
        char_sizes = [len(c.page_content) for c in self.chunks]
        token_sizes = [c.metadata.get('chunk_size_tokens', 0) for c in self.chunks]
        
        # 목표 범위 (300~500 토큰) 내 청크 비율
        in_range = sum(1 for t in token_sizes if 300 <= t <= 500)
        in_range_ratio = in_range / len(token_sizes) * 100 if token_sizes else 0
        
        print(f"\n[품질 지표]")
        print(f"  목표 토큰 범위 (300-500) 내 청크 비율: {in_range_ratio:.1f}%")
        print(f"  평균 청크 크기: {np.mean(char_sizes):.0f} chars / {np.mean(token_sizes):.0f} tokens")
        
        cv = np.std(token_sizes)/np.mean(token_sizes)*100 if np.mean(token_sizes) > 0 else 0
        print(f"  청크 크기 일관성 (CV): {cv:.1f}%")
        
        # 권장 사항
        print(f"\n[권장 사항]")
        recommendations = []
        if in_range_ratio < 70:
            msg = "⚠️ 목표 범위 내 청크 비율이 낮습니다. chunk_size 파라미터 조정을 권장합니다."
            print(f"  {msg}")
            recommendations.append(msg)
        else:
            msg = "✅ 청크 크기 분포가 양호합니다."
            print(f"  {msg}")
            recommendations.append(msg)
            
        if cv > 50:
            msg = "⚠️ 청크 크기 변동이 큽니다. 분할 전략 검토를 권장합니다."
            print(f"  {msg}")
            recommendations.append(msg)
        else:
            msg = "✅ 청크 크기가 일관적입니다."
            print(f"  {msg}")
            recommendations.append(msg)
        
        return {
            "total_chunks": len(self.chunks),
            "vector_count": stats.get('row_count', 0),
            "in_range_ratio": in_range_ratio,
            "avg_char_size": np.mean(char_sizes),
            "avg_token_size": np.mean(token_sizes),
            "cv": cv,
            "recommendations": recommendations,
        }


def validate_pipeline(
    vectorstore: MilvusVectorStore,
    chunks: List[Document],
    test_queries: Optional[List[str]] = None,
    sample_count: int = 3
) -> Dict[str, Any]:
    """파이프라인 검증 편의 함수"""
    monitor = QualityMonitor(vectorstore, chunks)
    
    # 분포 분석
    char_sizes, token_sizes = monitor.analyze_chunk_distribution()
    
    # 메타데이터 분석
    monitor.analyze_metadata()
    
    # 샘플 검토
    monitor.sample_chunks_review(n=sample_count)
    
    # 검색 테스트
    if test_queries:
        monitor.test_search_quality(test_queries)
    
    # 보고서 생성
    report = monitor.generate_report()
    
    return report
