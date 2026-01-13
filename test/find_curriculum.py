"""curriculumKo3.html 파일 위치 확인"""
from modules import get_vector_store
import os

vs = get_vector_store()

print("\n=== curriculumKo3.html 파일 검색 ===")

collections = vs.list_collections()
for coll_name in collections:
    print(f"\n[{coll_name}] 검색 중...")
    
    try:
        vs.client.load_collection(coll_name)
        
        # filename 필터로 검색
        results = vs.client.query(
            collection_name=coll_name,
            filter='filename == "curriculumKo3.html"',
            output_fields=["filename", "source", "chunk_id"],
            limit=10
        )
        
        if results:
            print(f"  ✓ 발견! {len(results)}개 청크")
            for hit in results[:3]:
                print(f"    - chunk_id: {hit.get('chunk_id')}")
                print(f"    - source: {hit.get('source')}")
        else:
            print(f"  없음")
            
    except Exception as e:
        print(f"  오류: {e}")

# 파일 시스템에서도 확인
print("\n\n=== 파일 시스템 확인 ===")
html_dir = "C:\\dev\\vectordb-etl\\html"
for root, dirs, files in os.walk(html_dir):
    for file in files:
        if file == "curriculumKo3.html":
            full_path = os.path.join(root, file)
            folder = os.path.basename(root)
            print(f"파일 위치: {full_path}")
            print(f"폴더명: {folder}")
