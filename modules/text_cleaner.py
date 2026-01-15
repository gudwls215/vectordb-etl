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
    SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9.,!?;:\'\"\-\(\)\[\]\{\}+@/·•]')
    
    # HWP 바이너리에서 자주 나오는 깨진 문자 범위 (제거 대상)
    HWP_GARBAGE_PATTERN = re.compile(
        r'[\u0080-\u00FF'    # Latin-1 Supplement 중 제어 문자
        r'\u0100-\u017F'     # Latin Extended-A
        r'\u0180-\u024F'     # Latin Extended-B
        r'\u0250-\u02AF'     # IPA Extensions
        r'\u0300-\u036F'     # Combining Diacritical Marks
        r'\u0370-\u03FF'     # Greek
        r'\u0400-\u04FF'     # Cyrillic (러시아어 - HWP 노이즈)
        r'\u0500-\u052F'     # Cyrillic Supplement
        r'\u0530-\u058F'     # Armenian
        r'\u0590-\u05FF'     # Hebrew
        r'\u0600-\u06FF'     # Arabic
        r'\u0700-\u074F'     # Syriac
        r'\u0900-\u097F'     # Devanagari
        r'\u0980-\u09FF'     # Bengali
        r'\u0B00-\u0B7F'     # Oriya
        r'\u0B80-\u0BFF'     # Tamil
        r'\u0C00-\u0C7F'     # Telugu
        r'\u0D00-\u0D7F'     # Malayalam
        r'\u0E00-\u0E7F'     # Thai
        r'\u1000-\u109F'     # Myanmar
        r'\u10A0-\u10FF'     # Georgian
        r'\u1100-\u11FF'     # Hangul Jamo (조합형 - 완성형만 유지)
        r'\u1200-\u137F'     # Ethiopic
        r'\u1400-\u167F'     # Canadian Aboriginal
        r'\u2000-\u206F'     # General Punctuation
        r'\u2070-\u209F'     # Superscripts and Subscripts
        r'\u20A0-\u20CF'     # Currency Symbols
        r'\u2100-\u214F'     # Letterlike Symbols
        r'\u2150-\u218F'     # Number Forms
        r'\u2190-\u21FF'     # Arrows
        r'\u2200-\u22FF'     # Mathematical Operators
        r'\u2300-\u23FF'     # Miscellaneous Technical
        r'\u2400-\u243F'     # Control Pictures
        r'\u2440-\u245F'     # Optical Character Recognition
        r'\u2460-\u24FF'     # Enclosed Alphanumerics
        r'\u2500-\u257F'     # Box Drawing
        r'\u2580-\u259F'     # Block Elements
        r'\u25A0-\u25FF'     # Geometric Shapes
        r'\u2600-\u26FF'     # Miscellaneous Symbols
        r'\u2700-\u27BF'     # Dingbats
        r'\u3000-\u303F'     # CJK Symbols and Punctuation
        r'\u3040-\u309F'     # Hiragana
        r'\u30A0-\u30FF'     # Katakana
        r'\u3100-\u312F'     # Bopomofo
        r'\u3200-\u32FF'     # Enclosed CJK Letters
        r'\u3300-\u33FF'     # CJK Compatibility
        r'\uFE00-\uFEFF'     # Variation Selectors
        r'\uFF00-\uFFEF'     # Halfwidth and Fullwidth Forms
        r'\uFFF0-\uFFFF'     # Specials
        r'\U00010000-\U0001FFFF'  # SMP (Supplementary Multilingual Plane)
        r']+'
    )
    
    # 허용할 문자만 남기는 패턴 (whitelist 방식)
    # 한글, 영문, 숫자, 기본 구두점, 공백만 유지
    ALLOWED_CHARS_PATTERN = re.compile(
        r'[^\sa-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣ\.\,\!\?\:\;\'\"\-\(\)\[\]\{\}\/\n\r\t@#$%&*+=~`<>|\\^·•※◎○●◆■□]'
    )
    
    # HWP 노이즈 패턴: 반복되는 무의미 문자열
    HWP_NOISE_PATTERNS = [
        # HWP 제어 문자로 자주 나오는 한글 (드문 조합)
        r'[밼밾뀀뀜럑됀쀀쀜쀌쟑쮜뛵픀븀휀렀낭갊뗈퐀팀햀쐀쐐썀썐찀쨀쩐짐쪠짤팜팠]\s*',
        r'[엀움은윀쁀쁘뻘뺘빀삐삘쌤씀썼쎄쐬쒀쓔쓰씌앜얘옜웨윔읨윙읭욀]\s*',
        r'[낗삓삙낸쓅맂곂탗탉랺곅섀쓇먈쇑눀뤀엌얮쓍샅헒밀곇딀솳쒬겼쓀킭봀쀄탅쀠뒭탇듅랬]\s*',
        r'[냖멎넀슻췀븷쀔쀐쀘뜀늲]\s*',  # 추가 발견된 노이즈
        r'(?:[A-Z]\s+){3,}',      # 연속된 대문자 + 공백
        r'\b[A-Z]\b(?:\s+\b[A-Z]\b){2,}',  # 단독 대문자 연속
        r'耀[^가-힣]*',           # 耀로 시작하는 노이즈
        r'[而戀肮]\s*',           # CJK 노이즈
        r'(?:\s[a-zA-Z]\s){2,}',  # 공백으로 둘러싸인 단일 문자 연속
        r'저\s*\n\s*저\s*\n',     # "저\n저\n" 패턴
        r'원본 그림의 이름:[^\n]*',  # 그림 메타데이터
        r'원본 그림의 크기:[^\n]*',  # 그림 크기 정보
        r'\d+pixel',              # pixel 정보
        r'(?<![가-힣])[a-zA-Z]{1,2}\d+(?![가-힣])',  # 짧은 코드 (한글 사이 아닌 경우)
        r'저\d*\s*저?\d*',        # "저0 저0" 패턴
        r'톱니모양의[^\n]*',      # 도형 설명
        r'화살표입니다[^\n]*',    # 도형 설명
        r'그림입니다[^\n]*',      # 그림 설명
    ]
    
    @classmethod
    def clean_hwp_text(cls, text: str) -> str:
        """HWP 전용 텍스트 정제 (더 강력한 필터링)"""
        if not text:
            return ""
        
        # 1. 먼저 일반 정제
        text = cls.clean_text(text)
        
        # 2. 의미 없는 짧은 줄 제거 (3글자 이하)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 한글이 포함된 의미있는 줄만 유지
            korean_chars = sum(1 for c in line if '가' <= c <= '힣')
            if korean_chars >= 3 or len(line) >= 10:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # 3. 연속된 공백과 줄바꿈 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """텍스트 정제 메인 함수"""
        if not text:
            return ""
        
        # 0. HWP 바이너리 노이즈 문자 제거 (가장 먼저 실행)
        text = cls.HWP_GARBAGE_PATTERN.sub(' ', text)
        
        # 0.1 허용되지 않는 문자 제거 (whitelist 방식)
        text = cls.ALLOWED_CHARS_PATTERN.sub(' ', text)
        
        # 0.2 HWP 특유의 노이즈 패턴 제거
        for pattern in cls.HWP_NOISE_PATTERNS:
            text = re.sub(pattern, ' ', text)
        
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
