"""HWP 텍스트 추출 테스트"""
import subprocess
from pathlib import Path
import shutil

hwp_file = r"C:\dev\vectordb-etl\hwp\compa\붙임2. 홈페이지 사업내용_취합.hwp"

print("1. shutil.which 테스트:")
print(f"   hwp5txt: {shutil.which('hwp5txt')}")
print(f"   hwp5txt.exe: {shutil.which('hwp5txt.exe')}")

print("\n2. subprocess로 hwp5txt 실행 테스트:")
try:
    result = subprocess.run(
        ['hwp5txt', hwp_file],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=60
    )
    
    print(f"   반환 코드: {result.returncode}")
    print(f"   stdout 길이: {len(result.stdout)}")
    print(f"   stdout 앞부분 200자:\n{result.stdout[:200]}")
    print(f"   stderr: {result.stderr[:200] if result.stderr else 'None'}")
except Exception as e:
    print(f"   에러: {e}")
