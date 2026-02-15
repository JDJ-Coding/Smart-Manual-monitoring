import streamlit as st
import os
import shutil
import time
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "posco")
POSCO_GPT_URL = "http://aigpt.posco.net/gpgpta01-gpt/gptApi/personalApi"
POSCO_GPT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-" \
"v2"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_db")
MANUAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuals")
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
SEARCH_K = 10

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Manual Assistant",
    layout="wide",
    page_icon="\u2699\ufe0f",
)

# ──────────────────────────────────────────────
# CSS Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    /* Global */
    .stApp {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main content */
    .main .block-container {
        max-width: 880px !important;
        padding: 1.5rem 1rem 160px 1rem !important;
    }

    /* Chat input bottom bar */
    div[data-testid="stBottom"] {
        background: linear-gradient(to top, #ffffff 85%, rgba(255,255,255,0)) !important;
        border-top: none !important;
        padding-top: 0.5rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #d1d5db;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f9fafb;
    }
    section[data-testid="stSidebar"] label {
        color: #9ca3af !important;
    }
    section[data-testid="stSidebar"] .stDivider {
        border-color: rgba(255,255,255,0.08);
    }

    /* Chat bubbles */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.4rem;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* Source citation */
    .source-box {
        background: linear-gradient(135deg, #eff6ff, #f0fdf4);
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #374151;
    }
    .source-box strong { color: #1e40af; }

    /* Welcome */
    .welcome-wrap {
        text-align: center;
        margin-top: 13vh;
        animation: fadeUp 0.6s ease-out;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .welcome-icon { font-size: 3.2rem; margin-bottom: 0.6rem; }
    .welcome-title {
        font-size: 2rem; font-weight: 800; color: #111827;
        margin-bottom: 0.3rem;
    }
    .welcome-sub {
        font-size: 1.05rem; color: #6b7280; margin-bottom: 2rem;
        line-height: 1.6;
    }
    .chip-row {
        display: flex; flex-wrap: wrap; justify-content: center;
        gap: 0.5rem; max-width: 620px; margin: 0 auto;
    }
    .chip {
        background: #fff; border: 1px solid #e5e7eb; border-radius: 999px;
        padding: 0.45rem 1rem; font-size: 0.88rem; color: #374151;
        transition: all 0.2s;
    }
    .chip:hover {
        border-color: #3b82f6; color: #2563eb;
        box-shadow: 0 2px 8px rgba(59,130,246,0.12);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 7px; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* Expander */
    .streamlit-expanderHeader { font-size: 0.85rem; color: #6b7280; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Directory Setup
# ──────────────────────────────────────────────
os.makedirs(MANUAL_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        st.error(
            "Hugging Face 임베딩 초기화에 실패했습니다. "
            "`sentence-transformers` 설치 후 다시 실행하세요.\n\n"
            f"오류: {e}"
        )
        st.stop()


def load_manual_db():
    embeddings = get_embeddings()
    pkl_path = os.path.join(DB_PATH, "index.pkl")

    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, "rb") as f:
                serialized = pickle.load(f)
            return FAISS.deserialize_from_bytes(
                serialized, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"벡터 DB 로드 오류: {e}")
            return None
    return None


def update_vector_db():
    import pickle

    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    pdf_files = [f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH, ignore_errors=True)
        return

    all_docs = []
    bar = st.sidebar.progress(0)
    status = st.sidebar.empty()

    for idx, f in enumerate(pdf_files):
        status.text(f"처리 중: {f} ({idx + 1}/{len(pdf_files)})")
        try:
            loader = PyPDFLoader(os.path.join(MANUAL_DIR, f))
            docs = loader.load_and_split(text_splitter)
            for doc in docs:
                doc.metadata["source"] = f
                if "page" in doc.metadata:
                    doc.metadata["page"] += 1
            if docs:
                all_docs.extend(docs)
        except Exception as e:
            st.error(f"PDF 처리 오류 ({f}): {e}")
        bar.progress((idx + 1) / len(pdf_files))

    bar.empty()
    status.empty()

    if not all_docs:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH, ignore_errors=True)
        return

    # Create vector DB from all documents
    status.text("벡터 DB 생성 중...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # Save using pickle to avoid encoding issues
    os.makedirs(DB_PATH, exist_ok=True)
    with open(os.path.join(DB_PATH, "index.pkl"), "wb") as f:
        pickle.dump(vectorstore.serialize_to_bytes(), f)

    status.empty()


def ask_posco_gpt(context: str, user_input: str) -> str:
    api_key = os.environ.get("POSCO_GPT_KEY")
    if not api_key:
        return (
            "**API 키 오류**\n\n"
            "환경변수 `POSCO_GPT_KEY`가 설정되지 않았습니다.\n\n"
            "```bash\nexport POSCO_GPT_KEY=your_key\n```"
        )

    system_prompt = "당신은 산업 설비 유지보수 전문가입니다."

    user_prompt = f"""사용자는 설비의 알람 코드나 고장 증상에 대해 묻고 있습니다.

아래 [매뉴얼 내용]을 분석하여 답변하세요.
- 내용이 표(Table) 형태로 되어 있다면 행/열을 주의 깊게 연결하여 해석하세요.
- '알람 코드', '원인', '조치 방법'을 명확히 구분해서 설명하세요.
- 내용에 없는 사실은 지어내지 마세요.
- 답변은 한국어로 작성하세요.

[매뉴얼 내용]
{context}

[질문]
{user_input}

[답변 형식]
1. 증상/알람 의미: (간략 설명)
2. 원인 및 조치 방법: (번호 매겨서 상세 설명)
3. 참고 문서: (파일명, 페이지)"""

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": POSCO_GPT_MODEL,
        }
        response = requests.post(POSCO_GPT_URL, headers=headers, data=json.dumps(payload))
        return response.text
    except Exception as e:
        return f"**POSCO GPT API 오류:** {str(e)}"


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    # Branding
    st.markdown("""
        <div style="text-align:center; padding:1rem 0 0.4rem 0;">
            <div style="font-size:2.2rem;">&#9881;&#65039;</div>
            <div style="font-size:1.15rem; font-weight:700; color:#f9fafb;">Smart Manual</div>
            <div style="font-size:0.72rem; color:#6b7280; margin-top:2px;">v2.0 &middot; Powered by Gemini</div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Search scope
    st.markdown("##### 검색 설정")
    file_list = sorted([f for f in os.listdir(MANUAL_DIR) if f.lower().endswith(".pdf")])
    search_options = ["전체 매뉴얼 검색"] + file_list
    selected_manual = st.selectbox(
        "검색 대상", search_options, index=0,
        help="특정 매뉴얼만 검색하려면 선택하세요.",
    )
    st.divider()

    # Admin section
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        st.markdown("##### 관리자")
        admin_pw = st.text_input(
            "비밀번호", type="password", key="admin_login_pw",
            placeholder="관리자 비밀번호",
        )
        if admin_pw == ADMIN_PASSWORD and admin_pw != "":
            st.session_state.is_admin = True
            st.rerun()

    if st.session_state.is_admin:
        st.success("관리자 모드 활성화")

        # PDF upload
        st.markdown("##### 매뉴얼 관리")
        uploaded = st.file_uploader(
            "PDF 업로드", type=["pdf"], accept_multiple_files=True,
            key="pdf_uploader", help="매뉴얼 PDF를 드래그하거나 클릭하여 업로드",
        )
        if uploaded:
            for uf in uploaded:
                with open(os.path.join(MANUAL_DIR, uf.name), "wb") as wf:
                    wf.write(uf.getbuffer())
            st.success(f"{len(uploaded)}개 파일 업로드 완료")
            time.sleep(1)
            st.rerun()

        # File list
        if file_list:
            st.caption(f"등록된 매뉴얼 ({len(file_list)}개)")
            for f in file_list:
                c1, c2 = st.columns([0.85, 0.15])
                c1.markdown(f"<small style='color:#d1d5db'>{f}</small>", unsafe_allow_html=True)
                if c2.button("\u2715", key=f"del_{f}", help=f"{f} 삭제"):
                    os.remove(os.path.join(MANUAL_DIR, f))
                    st.rerun()
        else:
            st.info("등록된 매뉴얼이 없습니다.")

        st.markdown("---")

        # DB rebuild
        if st.button("DB 전체 재구축", type="primary", use_container_width=True):
            with st.status("데이터베이스 재구축 중...", expanded=True) as status_box:
                st.write("기존 데이터 삭제 중...")
                if os.path.exists(DB_PATH):
                    shutil.rmtree(DB_PATH)
                st.write("벡터 DB 구축 중...")
                update_vector_db()
                status_box.update(label="재구축 완료!", state="complete", expanded=False)
            st.success("학습 완료!")
            time.sleep(1.5)
            st.rerun()

        if st.button("로그아웃", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()

    st.divider()

    # DB status
    db_exists = os.path.exists(DB_PATH)
    if db_exists:
        st.markdown("**DB 상태:** <span style='color:#34d399'>정상</span>", unsafe_allow_html=True)
    else:
        st.markdown("**DB 상태:** <span style='color:#f87171'>미구축</span>", unsafe_allow_html=True)

    # Clear chat
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # API key warning
    if not os.environ.get("POSCO_GPT_KEY"):
        st.warning("POSCO GPT API 키 미설정\n\n`export POSCO_GPT_KEY=키`")


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome screen
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-icon">&#9881;&#65039;</div>
            <div class="welcome-title">Smart Manual Assistant</div>
            <div class="welcome-sub">
                설비 매뉴얼 기반 AI 질의응답 시스템<br>
                알람 코드, 고장 진단, 유지보수 절차를 물어보세요.
            </div>
            <div class="chip-row">
                <div class="chip">FR-E800 인버터 알람 E.OC1 원인은?</div>
                <div class="chip">MR-J4 서보 AL.16 조치 방법</div>
                <div class="chip">파라미터 초기화 절차</div>
                <div class="chip">과전류 보호 기능 설명</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("참조 문서 보기"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Chat input
if prompt := st.chat_input("설비 관련 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Process last user message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_question = st.session_state.messages[-1]["content"]
    vectorstore = load_manual_db()

    if not vectorstore:
        with st.chat_message("assistant"):
            error_msg = "벡터 DB가 아직 구축되지 않았습니다. 관리자 로그인 후 **DB 전체 재구축** 버튼을 눌러주세요."
            st.warning(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
    else:
        with st.chat_message("assistant"):
            with st.spinner("매뉴얼에서 관련 내용을 검색하고 있습니다..."):
                # Build search kwargs
                search_kwargs = {"k": SEARCH_K}
                if selected_manual != "전체 매뉴얼 검색":
                    search_kwargs["filter"] = {"source": selected_manual}

                try:
                    docs = vectorstore.similarity_search(user_question, **search_kwargs)
                except Exception as e:
                    st.error(f"검색 오류: {e}")
                    docs = []

                if not docs:
                    no_result = f"'{selected_manual}' 범위에서 관련 내용을 찾지 못했습니다. 다른 키워드로 시도해보세요."
                    st.info(no_result)
                    st.session_state.messages.append({"role": "assistant", "content": no_result, "sources": []})
                else:
                    # Build context
                    context_parts = []
                    sources = []
                    seen = set()
                    for d in docs:
                        src = d.metadata.get("source", "Unknown")
                        page = d.metadata.get("page", 0)
                        context_parts.append(f"[파일: {src} | p.{page}]\n{d.page_content}")
                        key = f"{src} (p.{page})"
                        if key not in seen:
                            sources.append(key)
                            seen.add(key)

                    full_context = "\n\n---\n\n".join(context_parts)
                    answer = ask_posco_gpt(full_context, user_question)

                    st.markdown(answer)

                    if sources:
                        st.markdown("---")
                        sources_html = "<br>".join(f"&bull; {s}" for s in sources)
                        st.markdown(
                            f'<div class="source-box"><strong>참조 문서</strong><br>{sources_html}</div>',
                            unsafe_allow_html=True,
                        )
                        with st.expander("원문 보기"):
                            st.code(full_context, language=None)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
