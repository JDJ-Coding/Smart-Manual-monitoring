import sys
import os

# 1. 파이썬이 라이브러리를 찾는 모든 가능성 있는 경로를 강제로 추가
paths = [
    r"C:\Users\장덕진\AppData\Local\Programs\Python\Python313\Lib\site-packages",
    r"C:\Users\장덕진\AppData\Roaming\Python\Python313\site-packages",
    os.path.join(os.getcwd(), ".venv", "Lib", "site-packages")
]

for p in paths:
    if p not in sys.path:
        sys.path.append(p)

# 이제서야 원래 코드 시작

import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- [설정 및 초기화] ---
DB_PATH = "manual_db"
MANUAL_DIR = "./manuals"
if not os.path.exists(MANUAL_DIR): os.makedirs(MANUAL_DIR)

# 1. 무료 오픈소스 한국어 임베딩 모델 (CPU에서도 잘 돌아갑니다)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 2. 벡터 DB 로드 또는 생성 함수
def load_manual_db():
    embeddings = get_embeddings()
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# 3. POSCO GPT 호출 함수
def ask_posco_gpt(context, user_input):
    api_key = os.environ.get("POSCO_GPT_KEY")
    url = "http://aigpt.posco.net/gpgpta01-gpt/gptApi/personalApi"
    
    prompt = f"아래 매뉴얼 내용을 참고하여 답변하세요.\n\n[매뉴얼 내용]\n{context}\n\n[질문]\n{user_input}"
    
    payload = {
        "messages": [
            {"role": "system", "content": "너는 포스코 퓨처엠의 설비 전문가야. 매뉴얼에 근거하여 정확하게 답변해."},
            {"role": "user", "content": prompt}
        ],
        "model": "gpt-4o"
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    # 응답 형태가 JSON이라면 response.json().get('choices')[0]['message']['content'] 등으로 수정 필요
    return response.text

# --- [UI 레이아웃] ---
st.set_page_config(page_title="설비 매뉴얼 어시스턴트", layout="wide")

# 사이드바: 매뉴얼 관리
with st.sidebar:
    st.header("📂 매뉴얼 관리")
    files = os.listdir(MANUAL_DIR)
    st.write(f"현재 등록된 파일: {len(files)}개")
    for f in files[:10]: st.text(f"• {f}")
    
    if st.button("🔄 전체 매뉴얼 새로고침/학습"):
        with st.spinner("매뉴얼 분석 중... (약 수 분 소요)"):
            all_docs = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            for f in files:
                if f.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(MANUAL_DIR, f))
                    all_docs.extend(loader.load_and_split(splitter))
            
            vectorstore = FAISS.from_documents(all_docs, get_embeddings())
            vectorstore.save_local(DB_PATH)
            st.success("학습 완료!")
            st.rerun()

# 메인 화면: 채팅 인터페이스
st.title("🤖 POSCO 퓨처엠 설비 매뉴얼 챗봇")
st.caption("장치 모델명과 고장 증상을 입력하면 매뉴얼에서 정답을 찾아드립니다.")

# 세션 상태 초기화 (채팅 기록 저장용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 질문 입력
if prompt := st.chat_input("에러 코드나 점검 방법을 물어보세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 답변 생성 로직
    vectorstore = load_manual_db()
    if vectorstore:
        # 1. 검색 (유사 문서 3개)
        docs = vectorstore.similarity_search(prompt, k=3)
        
        # 2. 컨텍스트와 출처 정리
        context_list = []
        sources = set()
        for d in docs:
            context_list.append(d.page_content)
            # 메타데이터에서 파일명과 페이지 추출
            source_info = f"{d.metadata.get('source', 'Unknown')} (P. {d.metadata.get('page', '0') + 1})"
            sources.add(source_info)
        
        context_text = "\n---\n".join(context_list)
        
        # 3. AI 호출
        with st.chat_message("assistant"):
            with st.spinner("매뉴얼을 확인하고 있습니다..."):
                answer = ask_posco_gpt(context_text, prompt)
                
                full_response = f"{answer}\n\n---\n**📍 참고 출처:**\n" + "\n".join([f"- {s}" for s in sources])
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("사이드바에서 '전체 매뉴얼 새로고침'을 먼저 눌러주세요!")