"""
HWP File Loader Module
한글(HWP) 문서를 로드하고 텍스트를 추출하는 모듈

지원 형식:
- .hwp (HWP 5.0 이상)
- .hwpx (HWPX - Open Document Format 기반)
"""
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import re

from langchain_core.documents import Document

from .text_cleaner import TextCleaner
from .config import PipelineConfig, get_config, CUR_DIR

# HWP 처리 라이브러리 (선택적 import)
import subprocess
import shutil

olefile = None
pyhwp_available = False

# hwp5txt CLI 도구가 있는지 확인
hwp5txt_path = shutil.which('hwp5txt')
if hwp5txt_path:
    pyhwp_available = True
    HWP_LIBRARY = "pyhwp"
else:
    try:
        from pyhwpx import Hwp
        HWP_LIBRARY = "pyhwpx"
    except ImportError:
        try:
            import olefile as _olefile
            olefile = _olefile
            HWP_LIBRARY = "olefile"
        except ImportError:
            HWP_LIBRARY = None


class HWPTextExtractor:
    """HWP 파일에서 텍스트를 추출하는 클래스"""
    
    @staticmethod
    def extract_with_pyhwp(file_path: Path) -> Dict[str, Any]:
        """pyhwp CLI 도구(hwp5txt)를 사용한 텍스트 추출 (가장 정확)"""
        
        text = ""
        metadata = {
            'title': '',
            'author': '',
            'subject': '',
            'keywords': '',
        }
        
        try:
            # hwp5txt CLI 도구 실행
            result = subprocess.run(
                ['hwp5txt', str(file_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=60  # 60초 타임아웃
            )
            
            if result.returncode == 0:
                text = result.stdout
            else:
                print(f"hwp5txt 에러: {result.stderr}")
                # fallback to olefile
                return HWPTextExtractor.extract_with_olefile(file_path)
                
        except subprocess.TimeoutExpired:
            print(f"hwp5txt 타임아웃: {file_path}")
            return HWPTextExtractor.extract_with_olefile(file_path)
        except Exception as e:
            print(f"pyhwp 텍스트 추출 실패: {e}")
            return HWPTextExtractor.extract_with_olefile(file_path)
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    @staticmethod
    def extract_with_pyhwpx(file_path: Path) -> Dict[str, Any]:
        """pyhwpx 라이브러리를 사용한 텍스트 추출"""
        hwp = Hwp()
        hwp.open(str(file_path))
        
        # 전체 텍스트 추출
        text = hwp.get_text()
        
        # 메타데이터 추출 시도
        metadata = {
            'title': '',
            'author': '',
            'subject': '',
            'keywords': '',
        }
        
        try:
            # HWP 문서 정보 추출
            doc_info = hwp.get_doc_info() if hasattr(hwp, 'get_doc_info') else {}
            metadata.update({
                'title': doc_info.get('title', ''),
                'author': doc_info.get('author', ''),
                'subject': doc_info.get('subject', ''),
                'keywords': doc_info.get('keywords', ''),
            })
        except:
            pass
        
        hwp.quit()
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    @staticmethod
    def extract_with_olefile(file_path: Path) -> Dict[str, Any]:
        """olefile 라이브러리를 사용한 텍스트 추출 (기본 방식)"""
        import olefile as ole_module
        import zlib
        
        text_parts = []
        metadata = {
            'title': '',
            'author': '',
            'subject': '',
            'keywords': '',
        }
        
        ole = ole_module.OleFileIO(str(file_path))
        try:
            # 문서 메타데이터 추출
            meta = ole.get_metadata()
            if meta:
                metadata['title'] = getattr(meta, 'title', '') or ''
                metadata['author'] = getattr(meta, 'author', '') or ''
                metadata['subject'] = getattr(meta, 'subject', '') or ''
                metadata['keywords'] = getattr(meta, 'keywords', '') or ''
            
            # HWP 파일 구조에서 텍스트 추출
            if ole.exists('BodyText/Section0'):
                # 섹션 스트림 읽기
                for i in range(100):  # 최대 100개 섹션
                    section_name = f'BodyText/Section{i}'
                    if not ole.exists(section_name):
                        break
                    
                    section_data = ole.openstream(section_name).read()
                    
                    # 압축 해제 시도
                    try:
                        decompressed = zlib.decompress(section_data, -15)
                        # 텍스트 추출 (간단한 방식)
                        text = HWPTextExtractor._extract_text_from_section(decompressed)
                        if text:
                            text_parts.append(text)
                    except zlib.error:
                        # 압축되지 않은 경우
                        text = HWPTextExtractor._extract_text_from_section(section_data)
                        if text:
                            text_parts.append(text)
        finally:
            ole.close()
        
        return {
            'text': '\n\n'.join(text_parts),
            'metadata': metadata
        }
    
    @staticmethod
    def _extract_text_from_section(data: bytes) -> str:
        """섹션 바이너리 데이터에서 텍스트 추출 (개선된 버전)"""
        text_parts = []
        
        # HWP 텍스트 레코드 파싱
        try:
            # UTF-16LE로 디코딩 시도
            i = 0
            while i < len(data) - 1:
                char = data[i:i+2]
                try:
                    decoded = char.decode('utf-16le')
                    code_point = ord(decoded)
                    
                    # 허용할 문자 범위 정의
                    is_valid = (
                        # 기본 ASCII 출력 문자 (공백 ~ ~)
                        (0x20 <= code_point <= 0x7E) or
                        # 한글 자모
                        (0x1100 <= code_point <= 0x11FF) or
                        # 한글 호환 자모
                        (0x3130 <= code_point <= 0x318F) or
                        # 한글 음절
                        (0xAC00 <= code_point <= 0xD7AF) or
                        # 한글 확장
                        (0xA960 <= code_point <= 0xA97F) or
                        (0xD7B0 <= code_point <= 0xD7FF) or
                        # CJK 통합 한자 (일부 한자 포함 문서용)
                        (0x4E00 <= code_point <= 0x9FFF) or
                        # 줄바꿈, 탭
                        decoded in '\n\t\r'
                    )
                    
                    if is_valid:
                        text_parts.append(decoded)
                    elif code_point == 0:
                        pass  # NULL 문자 무시
                    else:
                        # 그 외 제어/특수 문자는 공백으로 (연속 방지)
                        if text_parts and text_parts[-1] != ' ':
                            text_parts.append(' ')
                except:
                    pass
                i += 2
        except:
            pass
        
        # 텍스트 정리
        text = ''.join(text_parts)
        
        # 연속 공백 정리
        text = re.sub(r'[ \t]+', ' ', text)
        # 연속 줄바꿈 정리
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 줄 시작/끝 공백 정리
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text.strip()
    
    @staticmethod
    def extract_with_hwp5txt(file_path: Path) -> Dict[str, Any]:
        """hwp5txt 명령줄 도구를 사용한 텍스트 추출 (fallback)"""
        import subprocess
        
        try:
            result = subprocess.run(
                ['hwp5txt', str(file_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            text = result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            text = ""
        
        return {
            'text': text,
            'metadata': {
                'title': '',
                'author': '',
                'subject': '',
                'keywords': '',
            }
        }


class StructuredHWPLoader:
    """HWP 파일을 로드하는 클래스"""
    
    def __init__(
        self,
        directory: Optional[str] = None,
        glob_pattern: str = "**/*.hwp",
        recursive: bool = True,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or get_config()
        
        # 디렉토리 설정
        if directory:
            self.directory = Path(directory)
        elif self.config.hwp_dir:
            self.directory = Path(self.config.hwp_dir)
        else:
            # 기본값: 프로젝트 루트의 hwp 폴더
            self.directory = Path(CUR_DIR) / 'hwp'
        
        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self.extractor = HWPTextExtractor()
        
        # 사용 가능한 라이브러리 확인
        if HWP_LIBRARY is None:
            print("Warning: HWP 처리 라이브러리가 설치되지 않았습니다.")
            print("  pip install pyhwpx 또는 pip install olefile 설치 권장")
    
    def load(self) -> List[Document]:
        """HWP 파일들을 로드하고 Document 리스트로 반환"""
        documents = []
        
        if not self.directory.exists():
            print(f"HWP 디렉토리가 존재하지 않습니다: {self.directory}")
            return documents
        
        # HWP 및 HWPX 파일 검색
        if self.recursive:
            hwp_files = list(self.directory.rglob('*.hwp')) + list(self.directory.rglob('*.hwpx'))
        else:
            hwp_files = list(self.directory.glob('*.hwp')) + list(self.directory.glob('*.hwpx'))
        
        print(f"발견된 HWP 파일 수: {len(hwp_files)}")
        
        # 폴더별로 그룹화하여 출력
        folder_counts = {}
        for file_path in hwp_files:
            folder_name = file_path.parent.name if file_path.parent != self.directory else 'root'
            folder_counts[folder_name] = folder_counts.get(folder_name, 0) + 1
        
        for folder, count in sorted(folder_counts.items()):
            print(f"  - {folder}: {count}개")
        
        for file_path in hwp_files:
            try:
                doc = self._load_single_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> Optional[Document]:
        """단일 HWP 파일 로드"""
        # 텍스트 추출
        extracted = self._extract_text(file_path)
        
        if not extracted['text']:
            print(f"텍스트 추출 실패: {file_path}")
            return None
        
        # 텍스트 정제 (HWP 전용 강력한 필터링 사용)
        cleaned_text = TextCleaner.clean_hwp_text(extracted['text'])
        
        if not cleaned_text or len(cleaned_text) < 10:
            print(f"유효한 텍스트 없음: {file_path}")
            return None
        
        # 파일 메타데이터
        file_stat = file_path.stat()
        
        # 폴더명 추출 (컬렉션 분리용)
        folder_name = file_path.parent.name if file_path.parent != self.directory else 'root'
        
        # 파일 확장자
        file_extension = file_path.suffix.lower()
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'folder_name': folder_name,
            'file_type': file_extension,
            'file_size': file_stat.st_size,
            'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'title': extracted['metadata'].get('title', '') or file_path.stem,
            'author': extracted['metadata'].get('author', ''),
            'subject': extracted['metadata'].get('subject', ''),
            'keywords': extracted['metadata'].get('keywords', ''),
            'char_count': len(cleaned_text),
            'language': self._detect_language_from_content(cleaned_text),
        }
        
        return Document(page_content=cleaned_text, metadata=metadata)
    
    def _extract_text(self, file_path: Path) -> Dict[str, Any]:
        """파일에서 텍스트 추출 (사용 가능한 방법 시도)"""
        # HWPX 파일인 경우 XML 기반 처리
        if file_path.suffix.lower() == '.hwpx':
            return self._extract_hwpx(file_path)
        
        # HWP 파일 처리 - pyhwp 우선 시도 (가장 정확)
        if HWP_LIBRARY == "pyhwp":
            try:
                print(f"🔧 pyhwp(hwp5txt) 사용하여 텍스트 추출: {file_path.name}")
                result = HWPTextExtractor.extract_with_pyhwp(file_path)
                if result['text']:
                    print(f"✅ pyhwp 추출 성공: {len(result['text'])}자")
                    return result
                else:
                    print(f"⚠️ pyhwp 추출 결과 텍스트 없음, olefile 시도")
            except Exception as e:
                print(f"❌ pyhwp 추출 실패, olefile 시도: {e}")
        
        if HWP_LIBRARY == "pyhwpx":
            try:
                return HWPTextExtractor.extract_with_pyhwpx(file_path)
            except Exception as e:
                print(f"pyhwpx 추출 실패, olefile 시도: {e}")
        
        # olefile 방식 시도
        print(f"🔧 olefile 사용하여 텍스트 추출 시도")
        try:
            result = HWPTextExtractor.extract_with_olefile(file_path)
            print(f"⚠️ olefile 추출: {len(result['text'])}자 (품질 낮음)")
            return result
        except Exception as e:
            print(f"olefile 추출 실패: {e}")
        
        # 최후의 수단: hwp5txt 명령줄 도구
        try:
            return HWPTextExtractor.extract_with_hwp5txt(file_path)
        except:
            pass
        
        return {'text': '', 'metadata': {}}
    
    def _extract_hwpx(self, file_path: Path) -> Dict[str, Any]:
        """HWPX 파일 추출 (ZIP 기반 XML)"""
        import zipfile
        import xml.etree.ElementTree as ET
        
        text_parts = []
        metadata = {
            'title': '',
            'author': '',
            'subject': '',
            'keywords': '',
        }
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # 콘텐츠 XML 파일 찾기
                for name in zf.namelist():
                    if 'section' in name.lower() and name.endswith('.xml'):
                        with zf.open(name) as f:
                            content = f.read().decode('utf-8')
                            # XML에서 텍스트 추출
                            root = ET.fromstring(content)
                            for elem in root.iter():
                                if elem.text:
                                    text_parts.append(elem.text.strip())
                    
                    # 메타데이터 파일
                    if 'meta' in name.lower() and name.endswith('.xml'):
                        with zf.open(name) as f:
                            content = f.read().decode('utf-8')
                            root = ET.fromstring(content)
                            for elem in root.iter():
                                tag_name = elem.tag.split('}')[-1].lower()
                                if tag_name == 'title' and elem.text:
                                    metadata['title'] = elem.text
                                elif tag_name == 'creator' and elem.text:
                                    metadata['author'] = elem.text
        except Exception as e:
            print(f"HWPX 추출 오류: {e}")
        
        return {
            'text': '\n'.join(text_parts),
            'metadata': metadata
        }
    
    def _detect_language_from_content(self, text: str) -> str:
        """텍스트 내용 기반 언어 감지"""
        # 영어 문자 카운트
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        # 한글 문자 카운트
        korean_chars = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')
        # 베트남어 특수문자 카운트
        vietnamese_chars = sum(1 for c in text if c in 'ăâđêôơưĂÂĐÊÔƠƯàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ')
        
        total_chars = len(text)
        if total_chars == 0:
            return 'korean'
        
        korean_ratio = korean_chars / total_chars
        vietnamese_ratio = vietnamese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if korean_ratio > 0.10:
            return 'korean'
        elif vietnamese_ratio > 0.02:
            return 'vietnamese'
        elif english_ratio > 0.30:
            return 'english'
        else:
            max_ratio = max(korean_ratio, vietnamese_ratio, english_ratio)
            if max_ratio == korean_ratio:
                return 'korean'
            elif max_ratio == vietnamese_ratio:
                return 'vietnamese'
            else:
                return 'english'


def load_hwp_documents(
    directory: Optional[str] = None,
    glob_pattern: str = "**/*.hwp",
    recursive: bool = True,
    config: Optional[PipelineConfig] = None
) -> List[Document]:
    """HWP 문서 로드 편의 함수"""
    loader = StructuredHWPLoader(
        directory=directory,
        glob_pattern=glob_pattern,
        recursive=recursive,
        config=config
    )
    return loader.load()


def get_hwp_folders(directory: Optional[str] = None) -> List[str]:
    """HWP 디렉토리 내의 폴더 목록 반환 (컬렉션 분리용)"""
    config = get_config()
    hwp_dir = Path(directory) if directory else Path(config.hwp_dir)
    
    if not hwp_dir.exists():
        return []
    
    folders = ['root']  # 루트 폴더
    
    for item in hwp_dir.iterdir():
        if item.is_dir():
            folders.append(item.name)
    
    return folders
