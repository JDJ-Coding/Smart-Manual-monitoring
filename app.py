import streamlit as st
import os
import requests
import json
import shutil
import time  # time 모듈 추가
import streamlit.components.v1 as components 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- [1. 기본 설정] ---
st.set_page_config(page_title="설비 매뉴얼 챗봇", layout="wide", page_icon="🏭")
ADMIN_PASSWORD = "posco" 

# --- [2. CSS 스타일] ---
st.markdown("""
    <style>
    .stApp { font-family: 'Pretendard', sans-serif; }
    div[data-testid="stAppViewContainer"] { overflow-y: auto !important; overflow-x: hidden !important; height: 100vh !important; }
    .main .block-container { padding-bottom: 150px !important; max-width: 100% !important; }
    div[data-testid="stBottom"] { background-color: white !important; z-index: 99999; border-top: 1px solid #ddd; }
    ::-webkit-scrollbar { width: 12px !important; display: block !important; }
    ::-webkit-scrollbar-thumb { background-color: #bbb; border-radius: 6px; }
    .stTextArea textarea { font-family: 'Courier New', monospace; font-size: 0.9rem; background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

# --- [3. 경로 설정] ---
DB_PATH = "manual_db"
MANUAL_DIR = "./manuals"
CACHE_DIR = "./manual_cache"
for d in [MANUAL_DIR, CACHE_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# --- [4. 함수 로직] ---
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
    
    # ★ [중요] 표 인식을 위해 Chunk Size 대폭 증가
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    
    current_files = [f for f in os.listdir(MANUAL_DIR) if f.endswith(".pdf")]
    
    # 기존 캐시 중 삭제된 파일 정리
    cached = os.listdir(CACHE_DIR)
    for c in cached:
        if c not in current_files: 
            shutil.rmtree(os.path.join(CACHE_DIR, c), ignore_errors=True)
            
    files_to_process = [f for f in current_files if not os.path.exists(os.path.join(CACHE_DIR, f))]
    
    if files_to_process:
        bar = st.sidebar.progress(0)
        status = st.sidebar.empty()
        for idx, f in enumerate(files_to_process):
            status.text(f"학습 중... {f}")
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
        status.empty()

    # DB 병합
    valid_caches = [f for f in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, f))]
    if not valid_caches:
        if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH, ignore_errors=True)
        return

    base_db = FAISS.load_local(os.path.join(CACHE_DIR, valid_caches[0]), embeddings, allow_dangerous_deserialization=True)
    for cache_name in valid_caches[1:]:
        sub_db = FAISS.load_local(os.path.join(CACHE_DIR, cache_name), embeddings, allow_dangerous_deserialization=True)
        base_db.merge_from(sub_db)
    base_db.save_local(DB_PATH)

def ask_posco_gpt(context, user_input):
    api_key = os.environ.get("POSCO_GPT_KEY")
    url = "http://aigpt.posco.net/gpgpta01-gpt/gptApi/personalApi"
    
    # ★ 프롬프트 강화
    prompt = f"""
    당신은 포스코 퓨처엠 설비 유지보수 전문가입니다.
    사용자는 설비의 알람 코드나 고장 증상에 대해 묻고 있습니다.
    
    아래 [매뉴얼 내용]을 분석하여 답변하세요.
    - 내용이 표(Table) 형태로 되어 있다면 행/열을 주의 깊게 연결하여 해석하세요.
    - '알람 코드', '원인', '조치 방법'을 명확히 구분해서 설명하세요.
    - 내용에 없는 사실은 지어내지 마세요.
    
    [매뉴얼 내용]
    {context}
    
    [질문]
    {user_input}
    
    [답변 형식]
    1. 🚨 증상/알람 의미: (간략 설명)
    2. 🛠️ 원인 및 조치 방법: (번호 매겨서 상세 설명)
    3. 📄 참고 문서: (파일명, 페이지)
    """
    
    payload = {"messages": [{"role": "user", "content": prompt}], "model": "gpt-5-chat-latest"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        return response.text
    except Exception as e:
        return f"GPT Error: {str(e)}"

def scroll_to_bottom():
    js = """<script>
        var body = window.parent.document.querySelector(".main");
        if (body) { body.scrollTop = body.scrollHeight; }
    </script>"""
    components.html(js, height=0, width=0)

# --- [5. 사이드바 (여기에 에러 수정됨)] ---
with st.sidebar:
    st.header("⚙️ 설정")
    file_list = [f for f in os.listdir(MANUAL_DIR) if f.endswith(".pdf")]
    search_options = ["전체 매뉴얼 검색"] + file_list
    selected_manual = st.selectbox("검색 대상 선택", search_options, index=0)
    
    st.divider()
    
    if "is_admin" not in st.session_state: st.session_state.is_admin = False
    
    # ★ 에러 해결: key="admin_login_pw" 추가하여 고유 ID 부여
    if not st.session_state.is_admin:
        if st.text_input("관리자 암호", type="password", key="admin_login_pw") == ADMIN_PASSWORD:
            st.session_state.is_admin = True
            st.rerun()
    
    if st.session_state.is_admin:
        st.success("🔓 관리자 모드")
        st.subheader("파일 관리")
        for f in file_list:
            c1, c2 = st.columns([0.8, 0.2])
            c1.text(f"{f}")
            if c2.button("x", key=f"del_{f}"): # key에 파일명 붙여서 고유화
                os.remove(os.path.join(MANUAL_DIR, f))
                if os.path.exists(os.path.join(CACHE_DIR, f)):
                    shutil.rmtree(os.path.join(CACHE_DIR, f))
                st.rerun()
        
        st.markdown("---")
        # ★ [초기화 및 재학습 버튼]
        if st.button("🔄 DB 전체 초기화 및 재학습", type="primary", use_container_width=True):
            with st.status("데이터베이스 재구축 중...", expanded=True) as status:
                st.write("1. 기존 데이터 삭제 중...")
                if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR)
                if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
                os.makedirs(CACHE_DIR, exist_ok=True)
                
                st.write("2. 새로운 설정(Chunk 1200)으로 학습 시작...")
                update_vector_db()
                status.update(label="완료!", state="complete", expanded=False)
            
            st.success("재학습 완료! 이제 질문해보세요.")
            time.sleep(1.5)
            st.rerun()
            
        if st.button("🚪 로그아웃"):
            st.session_state.is_admin = False
            st.rerun()

    st.divider()
    if st.button("🧹 대화 내용 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- [6. 메인 로직] ---
if "messages" not in st.session_state: st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("""
        <div style="text-align: center; margin-top: 15vh;">
            <div style="font-size: 5rem; margin-bottom: 20px;">🏭</div>
            <h1 style="color: #005eb8;">POSCO FUTURE M<br>Smart Assistant</h1>
        </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    vectorstore = load_manual_db()
    
    if vectorstore:
        with st.chat_message("assistant"):
            with st.spinner(f"'{selected_manual}'에서 정밀 검색 중..."):
                search_kwargs = {}
                if selected_manual != "전체 매뉴얼 검색":
                    search_kwargs["filter"] = {"source": selected_manual}
                
                try:
                    # 검색 범위 k=10
                    docs = vectorstore.similarity_search(st.session_state.messages[-1]["content"], k=10, **search_kwargs)
                except:
                    docs = []

                if not docs:
                    answer = f"⚠️ '{selected_manual}' 내에서 관련 내용을 찾을 수 없습니다."
                    full_context = ""
                    sources = []
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
                        st.text_area("Context", value=full_context, height=200)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                scroll_to_bottom()
    else:
        st.error("학습된 DB가 없습니다. 관리자 로그인 후 DB를 업데이트하세요.")
        st.session_state.messages.append({"role": "assistant", "content": "DB가 없습니다."})

st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
if len(st.session_state.messages) > 0: scroll_to_bottom()
