"""언어 필터 없이 검색"""
from modules import search_with_scores

results = search_with_scores(
    query='조직도',
    k=3,
    collection_name='docs_compa',
    search_all_collections=False,
    filter_language=None,
    auto_detect_language=False
)

print(f"\n검색 결과: {len(results)}개\n")

for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] Score: {score:.4f}")
    print(f"File: {doc.metadata.get('filename')}")
    print(f"Language: {doc.metadata.get('language')}")
    print(f"Content: {doc.page_content[:150]}...")
    print()
