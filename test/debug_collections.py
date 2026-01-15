"""컬렉션별 검색 디버깅 스크립트"""
from modules import get_vector_store, get_embeddings

vs = get_vector_store()

print("\n=== 컬렉션 목록 및 데이터 확인 ===")
collections = vs.list_collections()
stats = vs.get_all_collection_stats()

for coll_name in collections:
    count = stats.get(coll_name, {}).get('row_count', 0)
    print(f"\n{coll_name}: {count}개")
    
    # 각 컬렉션에서 샘플 검색
    try:
        vs.client.load_collection(coll_name)
        results = vs.search(
            query="회사",
            k=2,
            collection_name=coll_name,
            search_all_collections=False
        )
        print(f"  검색 결과: {len(results)}개")
        for i, hit in enumerate(results[:2], 1):
            filename = hit['entity'].get('filename', 'N/A')
            print(f"  [{i}] {filename}")
    except Exception as e:
        print(f"  검색 오류: {e}")
