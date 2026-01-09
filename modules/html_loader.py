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
    """구조를 보존하면서 HTML 파일을 로드하는 클래스"""
    
    def __init__(
        self, 
        directory: Optional[str] = None, 
        glob_pattern: str = "*.html",
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
        
    def load(self) -> List[Document]:
        """HTML 파일들을 로드하고 Document 리스트로 반환"""
        documents = []
        html_files = list(self.directory.glob(self.glob_pattern))
        
        print(f"발견된 HTML 파일 수: {len(html_files)}")
        
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
        
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'file_size': file_stat.st_size,
            'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'title': structure['title'],
            'heading_count': len(structure['headings']),
            'paragraph_count': structure['paragraphs'],
            'table_count': structure['tables'],
            'list_count': structure['lists'],
            'char_count': len(cleaned_text),
        }
        
        # 언어 감지 (파일명 기반)
        filename = file_path.name
        if 'En' in filename or '_en' in filename.lower():
            metadata['language'] = 'english'
        elif 'Ko' in filename or '_ko' in filename.lower():
            metadata['language'] = 'korean'
        elif 'Vi' in filename or '_vi' in filename.lower():
            metadata['language'] = 'vietnamese'
        else:
            metadata['language'] = 'unknown'
        
        return Document(page_content=cleaned_text, metadata=metadata)


def load_html_documents(
    directory: Optional[str] = None,
    glob_pattern: str = "*.html",
    config: Optional[PipelineConfig] = None
) -> List[Document]:
    """HTML 문서 로드 편의 함수"""
    loader = StructuredHTMLLoader(
        directory=directory,
        glob_pattern=glob_pattern,
        config=config
    )
    return loader.load()
