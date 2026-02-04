import streamlit as st
import os
import requests
import json
import shutil
import streamlit.components.v1 as components 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- [1. 기본 설정 및 비밀번호] ---
st.set_page_config(page_title="설비 매뉴얼 챗봇", layout="wide", page_icon="🏭")

# ★ 관리자 비밀번호 설정 (원하는대로 변경하세요)
ADMIN_PASSWORD = "posco" 

# --- [2. CSS: 스크롤바 유지 + UI 스타일] ---
st.markdown("""
    <style>
    /* 1. 전체 폰트 */
    .stApp { font-family: 'Pretendard', sans-serif; }

    /* 2. 메인 화면 스크롤 강제 허용 */
    div[data-testid="stAppViewContainer"] {
        overflow-y: auto !important;
        overflow-x: hidden !important;
        height: 100vh !important;
    }

    /* 3. 입력창 가림 방지 여백 */
    .main .block-container {
        padding-bottom: 150px !important; 
        max-width: 100% !important;
    }

    /* 4. 입력창 스타일 */
    div[data-testid="stBottom"] {
        background-color: white !important;
        z-index: 99999;
        border-top: 1px solid #ddd;
    }
    
    /* 5. 스크롤바 디자인 */
    ::-webkit-scrollbar {
        width: 12px !important;
        display: block !important;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #bbb;
        border-radius: 6px;
    }
    
    /* 6. 원문 보기 텍스트박스 */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# --- [3. 경로 설정] ---
DB_PATH = "manual_db"
MANUAL_DIR = "./manuals"
CACHE_DIR = "./manual_cache"
for d in [MANUAL_DIR, CACHE_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# --- [4. 백엔드 로직] ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="./model")

def load_manual_db():
    embeddings = get_embeddings()
    if os.path.exists(DB_PATH):
        try:
            return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        except:
            return None
    return None

def update_vector_db():
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # 청크 사이즈 미세 조정
    
    current_files = [f for f in os.listdir(MANUAL_DIR) if f.endswith(".pdf")]
    cached = os.listdir(CACHE_DIR)
    
    # 삭제된 파일 캐시 제거
    for c in cached:
        if c not in current_files: shutil.rmtree(os.path.join(CACHE_DIR, c))
            
    # 신규 파일 학습
    files_to_process = [f for f in current_files if not os.path.exists(os.path.join(CACHE_DIR, f))]
    
    if files_to_process:
        bar = st.sidebar.progress(0)
        for idx, f in enumerate(files_to_process):
            st.toast(f"학습 중: {f}")
            try:
                loader = PyPDFLoader(os.path.join(MANUAL_DIR, f))
                docs = loader.load_and_split(text_splitter)
                for doc in docs:
                    doc.metadata["source"] = f 
                    if "page" in doc.metadata: doc.metadata["page"] += 1
                
                if docs:
                    temp_db = FAISS.from_documents(docs, embeddings)
                    temp_db.save_local(os.path.join(CACHE_DIR, f))
            except Exception as e:
                st.error(f"오류 ({f}): {e}")
            bar.progress((idx + 1) / len(files_to_process))
        bar.empty()

    # DB 병합
    valid_caches = [f for f in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, f))]
    if not valid_caches:
        if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
        return

    st.toast("DB 통합 중...")
    base_db = FAISS.load_local(os.path.join(CACHE_DIR, valid_caches[0]), embeddings, allow_dangerous_deserialization=True)
    for cache_name in valid_caches[1:]:
        sub_db = FAISS.load_local(os.path.join(CACHE_DIR, cache_name), embeddings, allow_dangerous_deserialization=True)
        base_db.merge_from(sub_db)
    base_db.save_local(DB_PATH)
    st.success("업데이트 완료!")

def ask_posco_gpt(context, user_input):
    api_key = os.environ.get("POSCO_GPT_KEY")
    url = "http://aigpt.posco.net/gpgpta01-gpt/gptApi/personalApi"
    
    prompt = f"""
    당신은 포스코 퓨처엠 설비 유지보수 전문가입니다.
    아래 제공된 [매뉴얼 내용]만을 바탕으로 사용자의 질문에 답변하세요.
    내용에 없는 사실을 지어내지 마세요.
    
    [매뉴얼 내용]
    {context}
    
    [질문]
    {user_input}
    
    [답변 형식]
    1. 핵심 조치 내용 (번호 매기기)
    2. 참고 문서와 페이지 (정확하게 명시)
    """
    
    payload = {"messages": [{"role": "user", "content": prompt}], "model": "gpt-4o"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        return response.text
    except Exception as e:
        return f"GPT Error: {str(e)}"

# 스크롤 자동 이동 (JS)
def scroll_to_bottom():
    js = """
    <script>
        var body = window.parent.document.querySelector(".main");
        if (body) { body.scrollTop = body.scrollHeight; }
    </script>
    """
    components.html(js, height=0, width=0)

# --- [5. 사이드바 로직 (핵심 변경)] ---
with st.sidebar:
    st.header("⚙️ 설정")
    
    # (1) 검색 대상 선택 (정확도 향상용)
    file_list = [f for f in os.listdir(MANUAL_DIR) if f.endswith(".pdf")]
    search_options = ["전체 매뉴얼 검색"] + file_list
    selected_manual = st.selectbox(
        "검색 대상 선택", 
        search_options, 
        index=0,
        help="특정 매뉴얼을 선택하면 답변의 정확도가 높아집니다."
    )
    
    st.divider()
    
    # (2) 관리자 로그인 (보안 기능)
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        admin_input = st.text_input("관리자 암호", type="password")
        if admin_input == ADMIN_PASSWORD:
            st.session_state.is_admin = True
            st.rerun()
    
    # (3) 관리자 전용 메뉴 (로그인 시에만 보임)
    if st.session_state.is_admin:
        st.success("🔓 관리자 모드")
        
        st.subheader("파일 관리")
        for f in file_list:
            c1, c2 = st.columns([0.8, 0.2])
            c1.text(f"{f}")
            if c2.button("x", key=f):
                os.remove(os.path.join(MANUAL_DIR, f))
                st.rerun()
        
        if st.button("🔄 DB 업데이트 (학습)", type="primary", use_container_width=True):
            update_vector_db()
            st.rerun()
            
        if st.button("🚪 로그아웃"):
            st.session_state.is_admin = False
            st.rerun()
    else:
        st.info("관리자만 파일을 수정할 수 있습니다.")

    st.divider()
    if st.button("🧹 대화 내용 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- [6. 메인 채팅 화면] ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 랜딩 페이지
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div style="text-align: center; margin-top: 15vh;">
            <div style="font-size: 5rem; margin-bottom: 20px;">🏭</div>
            <h1 style="color: #005eb8;">POSCO FUTURE M<br>Smart Assistant</h1>
            <p style="color: #555; font-size: 1.2rem;">설비 매뉴얼 및 트러블슈팅 AI</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.caption("🚀 POSCO FUTURE M 설비 매뉴얼 챗봇")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 입력창
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# 답변 로직 (필터링 적용)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    vectorstore = load_manual_db()
    
    if vectorstore:
        with st.chat_message("assistant"):
            with st.spinner(f"'{selected_manual}'에서 검색 중..."):
                
                # ★ 핵심: 필터링 로직 추가
                # 전체 검색이 아니면 metadata의 'source'가 일치하는 것만 검색함
                search_kwargs = {}
                if selected_manual != "전체 매뉴얼 검색":
                    search_kwargs["filter"] = {"source": selected_manual}
                
                # 검색 수행
                try:
                    docs = vectorstore.similarity_search(
                        st.session_state.messages[-1]["content"], 
                        k=5, 
                        **search_kwargs
                    )
                except Exception as e:
                    st.error("검색 중 오류가 발생했습니다. DB를 업데이트해주세요.")
                    docs = []

                if not docs:
                    answer = f"⚠️ '{selected_manual}' 내에서 관련 내용을 찾을 수 없습니다. 다른 매뉴얼을 선택하거나 질문을 구체적으로 해주세요."
                    sources = []
                    full_context = ""
                else:
                    context_parts = []
                    sources = set()
                    for d in docs:
                        content = d.page_content 
                        src = d.metadata.get('source', 'Unknown')
                        page = d.metadata.get('page', 0)
                        context_parts.append(f"📄 [파일: {src} | p.{page}]\n{content}")
                        sources.add(f"{src} (p.{page})")
                    
                    full_context = "\n\n".join(context_parts)
                    answer = ask_posco_gpt(full_context, st.session_state.messages[-1]["content"])
                
                st.markdown(answer)
                
                if sources:
                    st.divider()
                    st.caption(f"**📚 참조 문서:** {', '.join(sources)}")
                    with st.expander("🔍 원문 보기"):
                        st.text_area("원문", value=full_context, height=300, label_visibility="collapsed")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                scroll_to_bottom()
    else:
        st.error("학습된 DB가 없습니다. 관리자에게 문의하여 DB를 업데이트하세요.")
        st.session_state.messages.append({"role": "assistant", "content": "DB가 없습니다."})

# 하단 여백 확보
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
if len(st.session_state.messages) > 0:
    scroll_to_bottom()
