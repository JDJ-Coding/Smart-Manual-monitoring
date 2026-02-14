# Smart Manual Assistant

설비 매뉴얼 기반 AI 질의응답 시스템 (Hugging Face 임베딩 사용)

## 🚀 빠른 시작

### 1. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
Gemini API 키를 환경 변수로 설정하세요:

**Windows:**
```cmd
set MY_API_KEY_GOOGLE=your_gemini_api_key
```

**Linux/Mac:**
```bash
export MY_API_KEY_GOOGLE=your_gemini_api_key
```

### 3. 벡터 DB 구축
처음 한 번만 실행하세요:
```bash
python build_db.py
```

### 4. 앱 실행

**방법 1: 배치 파일 사용 (Windows 권장)**
```cmd
start.bat
```

**방법 2: 직접 실행**
```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 을 열어주세요.

## ⚠️ 주의사항

### ❌ 잘못된 실행 방법
```bash
python app.py  # 이렇게 하면 안됩니다!
```

### ✅ 올바른 실행 방법
```bash
streamlit run app.py  # 이렇게 실행해야 합니다!
```

또는 `start.bat`을 더블클릭하세요.

## 📁 프로젝트 구조

```
Smart-Manual-monitoring/
├── app.py                 # Streamlit 웹 앱
├── build_db.py           # 벡터 DB 구축 스크립트
├── requirements.txt      # 필수 패키지 목록
├── start.bat            # Windows 시작 스크립트
├── README.md            # 이 파일
├── manuals/             # PDF 매뉴얼 저장 폴더
└── manual_db/           # 벡터 DB 저장 폴더
    └── index.pkl        # 임베딩 인덱스 (pickle 형식)
```

## 🔧 기능

- **PDF 매뉴얼 업로드**: 관리자 모드에서 PDF 파일 업로드
- **질의응답**: 매뉴얼 내용 기반 AI 답변
- **검색 범위 선택**: 특정 매뉴얼 또는 전체 검색
- **DB 재구축**: 매뉴얼 변경 시 벡터 DB 재생성

## 🔑 관리자 기능

- 비밀번호: `posco` (기본값)
- `ADMIN_PASSWORD` 환경 변수로 변경 가능

## 🛠️ 기술 스택

- **Streamlit**: 웹 UI
- **Hugging Face**: 임베딩 모델 (paraphrase-multilingual-MiniLM-L12-v2)
- **FAISS**: 벡터 검색
- **LangChain**: 문서 처리
- **Google Gemini**: AI 답변 생성

## 📝 문제 해결

### "ScriptRunContext" 오류 발생 시
- `python app.py` 대신 `streamlit run app.py`를 사용하세요
- 또는 `start.bat`을 실행하세요

### "벡터 DB 로드 오류" 발생 시
- `python build_db.py`를 먼저 실행하세요
- `manual_db` 폴더를 삭제하고 다시 구축하세요

### PDF 파싱 오류 발생 시
- `cryptography` 패키지가 설치되어 있는지 확인하세요
- `pip install cryptography`

## 📄 라이선스

MIT License
