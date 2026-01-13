"""
HTML File Loader Module
"""
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document
from bs4 import BeautifulSoup

from .text_cleaner import TextCleaner
from .config import PipelineConfig, get_config


class StructuredHTMLLoader:
    """구조를 보존하면서 HTML/JSP 파일을 로드하는 클래스"""
    
    def __init__(
        self, 
        directory: Optional[str] = None, 
        glob_pattern: str = "**/*.{html,jsp}",
        recursive: bool = True,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or get_config()
        
        # 디렉토리 설정
        if directory:
            self.directory = Path(directory)
        elif self.config.html_dir:
            self.directory = Path(self.config.html_dir)
        else:
            raise ValueError("html_dir must be specified in config or as directory parameter")
            
        self.glob_pattern = glob_pattern or self.config.html_glob_pattern
        self.recursive = recursive
        
    def load(self) -> List[Document]:
        """HTML/JSP 파일들을 로드하고 Document 리스트로 반환"""
        documents = []
        
        # 재귀적으로 HTML과 JSP 파일 검색
        if self.recursive:
            html_files = list(self.directory.rglob('*.html')) + list(self.directory.rglob('*.jsp'))
        else:
            html_files = list(self.directory.glob('*.html')) + list(self.directory.glob('*.jsp'))
        
        print(f"발견된 HTML/JSP 파일 수: {len(html_files)}")
        
        # 폴더별로 그룹화하여 출력
        folder_counts = {}
        for file_path in html_files:
            folder_name = file_path.parent.name if file_path.parent != self.directory else 'root'
            folder_counts[folder_name] = folder_counts.get(folder_name, 0) + 1
        
        for folder, count in sorted(folder_counts.items()):
            print(f"  - {folder}: {count}개")
        
        for file_path in html_files:
            try:
                doc = self._load_single_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return documents
    
    def _load_single_file(self, file_path: Path) -> Document:
        """단일 HTML 파일 로드"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 스크립트, 스타일 태그 제거
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # 구조 정보 추출
        structure = TextCleaner.extract_structure(html_content)
        
        # 본문 텍스트 추출 및 정제
        text = soup.get_text(separator='\n')
        cleaned_text = TextCleaner.clean_text(text)
        
        # 파일 메타데이터
        file_stat = file_path.stat()
        
        # 폴더명 추출 (컬렉션 분리용)
        folder_name = file_path.parent.name if file_path.parent != self.directory else 'root'
        
        # 파일 확장자
        file_extension = file_path.suffix.lower()
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'folder_name': folder_name,  # 폴더명 추가
            'file_type': file_extension,  # .html, .jsp 등
            'file_size': file_stat.st_size,
            'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'title': structure['title'],
            'heading_count': len(structure['headings']),
            'paragraph_count': structure['paragraphs'],
            'table_count': structure['tables'],
            'list_count': structure['lists'],
            'char_count': len(cleaned_text),
        }
        
        # 언어 감지 ( 내용 기반 )
        metadata['language'] = self._detect_language_from_content(cleaned_text)
        
        return Document(page_content=cleaned_text, metadata=metadata)
    
    def _detect_language_from_content(self, text: str) -> str:
        """텍스트 내용 기반 언어 감지 (간단한 휴리스틱)"""
        # 영어 문자 카운트
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        # 한글 문자 카운트
        korean_chars = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')
        # 베트남어 특수문자 카운트 (ă, â, đ, ê, ô, ơ, ư 및 성조)
        vietnamese_chars = sum(1 for c in text if c in 'ăâđêôơưĂÂĐÊÔƠƯàáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵ')
        
        total_chars = len(text)
        if total_chars == 0:
            return 'korean'  # 기본값
        
        korean_ratio = korean_chars / total_chars
        vietnamese_ratio = vietnamese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # 한글이 10% 이상이면 한국어
        if korean_ratio > 0.10:
            return 'korean'
        # 베트남어 특수문자가 2% 이상이면 베트남어
        elif vietnamese_ratio > 0.02:
            return 'vietnamese'
        # 영어 문자가 30% 이상이면 영어
        elif english_ratio > 0.30:
            return 'english'
        # 그 외는 혼합 또는 불명확하므로 가장 높은 비율의 언어 선택
        else:
            max_ratio = max(korean_ratio, vietnamese_ratio, english_ratio)
            if max_ratio == korean_ratio:
                return 'korean'
            elif max_ratio == vietnamese_ratio:
                return 'vietnamese'
            else:
                return 'english'


def load_html_documents(
    directory: Optional[str] = None,
    glob_pattern: str = "**/*.{html,jsp}",
    recursive: bool = True,
    config: Optional[PipelineConfig] = None
) -> List[Document]:
    """HTML/JSP 문서 로드 편의 함수"""
    loader = StructuredHTMLLoader(
        directory=directory,
        glob_pattern=glob_pattern,
        recursive=recursive,
        config=config
    )
    return loader.load()
