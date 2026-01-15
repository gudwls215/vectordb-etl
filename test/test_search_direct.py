"""직접 검색 테스트"""
from modules import search_with_scores

print("=== 직접 검색 테스트 ===\n")

# docs_compa에서만 검색
results = search_with_scores(
    query="조직도",
    k=3,
    collection_name="docs_compa",
    search_all_collections=False
)

print(f"\n검색 결과: {len(results)}개\n")

for i, (doc, score) in enumerate(results, 1):
    print(f"[{i}] Score: {score:.4f}")
    print(f"Source: {doc.metadata.get('filename', 'N/A')}")
    print(f"Folder: {doc.metadata.get('folder_name', 'N/A')}")
    if 'collection' in doc.metadata:
        print(f"Collection: {doc.metadata['collection']}")
    print(f"Content: {doc.page_content[:200]}...")
    print()
