"""
Text Cleaning Module
"""
import re
from typing import Dict, Any
from bs4 import BeautifulSoup


class TextCleaner:
    """텍스트 정제 클래스"""
    
    # 이모지 패턴 - 한글 범위(AC00-D7AF) 제외
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+", 
        flags=re.UNICODE
    )
    
    # 템플릿 태그 패턴 (Handlebars, Jinja, Mustache 등)
    TEMPLATE_TAG_PATTERNS = [
        r'\{\{#?\/?[^}]+\}\}',      # {{#layout}}, {{/layout}}, {{ layout }}
        r'\{%[^%]+%\}',              # {% block %}, {% endblock %}
        r'\$\{[^}]+\}',              # ${variable}
        r'<%[^%]+%>',                # <% erb %>
        r'\[\[[^\]]+\]\]',           # [[wiki style]]
    ]
    
    # 헤더/푸터 패턴
    HEADER_FOOTER_PATTERNS = [
        r'Copyright.*?\d{4}',
        r'All [Rr]ights [Rr]eserved',
        r'Page\s*\d+\s*(of\s*\d+)?',
        r'^\s*\d+\s*$',  # 단독 페이지 번호
        r'\|\s*Page\s*\d+',
        r'www\..*?\.com',
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    ]
    
    # JavaScript 관련 패턴
    JS_PATTERNS = [
        r'javascript:\s*void\s*\([^)]*\)',  # javascript:void(0)
        r'onclick\s*=\s*["\'][^"\']+["\']',  # onclick="..."
        r'PageScript\.[a-zA-Z]+\([^)]*\)',   # PageScript.clickMap(...)
        r'function\s*\([^)]*\)\s*\{[^}]*\}', # function() {...}
    ]
    
    # 특수문자 패턴 (한글, 영문, 숫자, 기본 문장부호 제외)
    SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9.,!?;:\'\"\-\(\)\[\]\{\}+@]')
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """텍스트 정제 메인 함수"""
        if not text:
            return ""
        
        # 1. HTML 태그 제거
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 2. 템플릿 태그 제거 ({{layout}}, {%block%} 등)
        for pattern in cls.TEMPLATE_TAG_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 3. JavaScript 관련 텍스트 제거
        for pattern in cls.JS_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 4. 이모지 제거
        text = cls.EMOJI_PATTERN.sub('', text)
        
        # 5. 헤더/푸터/페이지 번호 제거
        for pattern in cls.HEADER_FOOTER_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 6. 특수문자 제거 (기본 문장부호는 유지)
        text = cls.SPECIAL_CHAR_PATTERN.sub(' ', text)
        
        # 7. 중복 공백 정규화
        text = re.sub(r'[ \t]+', ' ', text)
        
        # 8. 중복 줄바꿈 정규화
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 9. 앞뒤 공백 제거 및 각 줄 정리
        lines = [line.strip() for line in text.split('\n')]
        # 빈 줄이 아닌 줄만 유지하되, 문단 구분을 위해 빈 줄 하나는 허용
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True
        
        text = '\n'.join(cleaned_lines).strip()
        
        return text
    
    @classmethod
    def extract_structure(cls, html_content: str) -> Dict[str, Any]:
        """HTML 구조 정보 추출"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        structure = {
            'title': '',
            'headings': [],
            'sections': [],
            'lists': 0,
            'tables': 0,
            'paragraphs': 0
        }
        
        # 타이틀 추출
        title_tag = soup.find('title')
        if title_tag:
            structure['title'] = cls.clean_text(title_tag.get_text())
        
        # 헤딩 추출
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                heading_text = cls.clean_text(heading.get_text())
                if heading_text:
                    structure['headings'].append({
                        'level': i,
                        'text': heading_text
                    })
        
        # 구조 요소 카운트
        structure['lists'] = len(soup.find_all(['ul', 'ol']))
        structure['tables'] = len(soup.find_all('table'))
        structure['paragraphs'] = len(soup.find_all('p'))
        
        return structure


def clean_text(text: str) -> str:
    """텍스트 정제 편의 함수"""
    return TextCleaner.clean_text(text)


def extract_structure(html_content: str) -> Dict[str, Any]:
    """HTML 구조 추출 편의 함수"""
    return TextCleaner.extract_structure(html_content)
