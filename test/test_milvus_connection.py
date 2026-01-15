"""Milvus 연결 테스트 스크립트"""
from pymilvus import MilvusClient, connections
import sys

def test_connection():
    try:
        print("Connecting to Milvus at http://localhost:19530...")
        
        # MilvusClient로 연결
        client = MilvusClient(uri="http://localhost:19530")
        
        print("✓ Connected successfully!")
        
        # 서버 정보
        try:
            # connections를 통한 버전 확인
            connections.connect(
                alias="default",
                host="localhost",
                port="19530"
            )
            from pymilvus import utility
            version = utility.get_server_version()
            print(f"\nMilvus version: {version}")
        except Exception as e:
            print(f"\nVersion check skipped: {e}")
        
        # 컬렉션 목록
        collections = client.list_collections()
        print(f"\nCollections: {collections if collections else '(No collections yet)'}")
        
        # 연결 종료
        client.close()
        if connections.has_connection("default"):
            connections.disconnect("default")
        
        print("\n✓ Milvus is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
