import os
import json
import uuid
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from embedding_generator import generate_and_save_embeddings

# 페이지 설정 (맨 위에 추가)
st.set_page_config(
    page_title="Bifrost GPT",
    page_icon=os.path.join(os.path.dirname(__file__), "resources", "img", "logo.webp"),
    layout="centered"
)

# OpenAI API Key 설정
def get_openai_api_key():
    # 환경변수에서 API 키를 우선적으로 가져옴
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("Using OPENAI_API_KEY from environment variable.")
        return api_key

    # 환경변수가 없으면 config.json 파일에서 API 키를 가져옴
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        api_key = config.get("OPENAI_API_KEY")
        if api_key:
            print("Using OPENAI_API_KEY from config.json.")
            return api_key
    except FileNotFoundError:
        print("config.json not found.")

    raise ValueError("OPENAI_API_KEY is not set in environment variables or config.json.")

OPENAI_API_KEY = get_openai_api_key()

# api key 유효성 체크 (그대로 실행하세요)
def check_openai_api_key(key):
    try:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=key)
        llm.invoke('key테스트입니다.')
        del llm
        return True
    except Exception as e:
        if e.code == 'invalid_api_key':
            st.sidebar.error('OPENAI API 키를 확인해주세요.')
        else:
            st.sidebar.error(e.code)
        del llm
        return False

# 로고 파일 경로 설정
logo_path = os.path.join(os.path.dirname(__file__), "resources", "img", "logo.webp")
if not os.path.exists(logo_path):
    #st.sidebar.warning("⚠️ 'logo.webp' 파일이 존재하지 않습니다. 같은 폴더에 추가해주세요.")
    logo_path = None  # 로고가 없으면 기본 아이콘 사용

# FAISS 폴더 경로 설정
faiss_folder_path = os.path.join(os.path.dirname(__file__), 'faiss')

# FAISS 벡터 스토어 불러오기
if os.path.exists(faiss_folder_path):
    vectorstore = FAISS.load_local(
        faiss_folder_path,
        embeddings=OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY),
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = generate_and_save_embeddings(OPENAI_API_KEY, faiss_folder_path)

#검색기 생성
retriever = vectorstore.as_retriever(search_kwargs={'k': 7})

# 인덱스와 ID 출력
print(vectorstore.index_to_docstore_id)

# key 값이 있는 경우만 출력
if OPENAI_API_KEY:
    # key가 유효한 경우만 제목 출력
    if check_openai_api_key(OPENAI_API_KEY):
        # 프롬프트 생성
        template = """다음 컨텍스트를 사용하여 질문에 답변해주세요. 바이프로스트 네트워크의 서비스들에 대한 고객지원을 담당하는 챗봇으로서, 친절하고 상세하게 답변해주세요. 바이프로스트 서비스와 관련없는 질문은 답하지 마세요.

컨텍스트: {context}

이전 대화 기록:
{history}

질문: {question}

답변:"""
        prompt = PromptTemplate.from_template(template)
        
        # LLM 생성
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        # 파서 생성
        parser = StrOutputParser()
        # 체인 생성
        chain = (
            {
                "context": lambda x: retriever.get_relevant_documents(x["question"]), 
                "question": lambda x: x["question"],
                "history": lambda x: x["history"]
            }
            | prompt 
            | llm 
            | parser
        )

        st.title("Bifrost Network Assistant")
        st.subheader("무엇을 도와드릴까요?")

        # 세션 상태(session_state)에 메시지 항목 생성
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 세션 상태에 저장한 이전 챗 기록을 불러와서 챗 메시지로 출력
        for message in st.session_state.messages:
            # 역할에 맞춰서 chat_message로 출력
            with st.chat_message(message["role"], avatar=logo_path if message["role"] == "assistant" else None):
                st.write(message["content"])

        # 챗 유저 메시지 입력
        if question := st.chat_input("바이프로스트 네트워크에 대해 궁금하신사항들을 물어보세요 :)"):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant", avatar=logo_path):
                history = '\n'.join([':'.join(entry.values()) for entry in st.session_state.messages])
                # 입력값을 딕셔너리로 전달
                message = st.write_stream(chain.stream({
                    "question": question,
                    "history": history
                }))
            # 현재 메시지(question, message)를 세션 상태에 저장
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": message})

# 자동 스크롤 스크립트
js = '''
<script>
    var body = window.parent.document.querySelector(".main");
    console.log(body);
    body.scrollTop = 0;
</script>
'''

st.components.v1.html(js)