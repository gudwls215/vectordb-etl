"""HWP 청크 데이터 디버깅"""
import pickle
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
hwp_chunks_path = data_dir / "hwp_chunks.pkl"

if hwp_chunks_path.exists():
    with open(hwp_chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"총 청크 수: {len(chunks)}")
    print(f"\n청크 타입: {type(chunks)}")
    
    # 모든 청크 정보 출력
    for idx, chunk in enumerate(chunks):
        print(f"\n청크 {idx + 1}:")
        print(f"  타입: {type(chunk)}")
        print(f"  page_content 길이: {len(chunk.page_content)}")
        print(f"  메타데이터:")
        for key, value in chunk.metadata.items():
            print(f"    {key}: {value}")    
    
    if chunks:
        print(f"\n첫 번째 청크:")
        print(f"  타입: {type(chunks[0])}")
        print(f"  page_content 길이: {len(chunks[0].page_content)}")
        print(f"  page_content[:200]: {chunks[0].page_content[:200]}")
        print(f"\n  메타데이터:")
        for key, value in chunks[0].metadata.items():
            print(f"    {key}: {value}")
    else:
        print("청크가 비어있습니다!")
else:
    print(f"파일이 없습니다: {hwp_chunks_path}")
