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
import config  # config.py에서 OPENAI_API_KEY를 가져옴

# 페이지 설정 (맨 위에 추가)
st.set_page_config(
    page_title="Bifrost GPT",
    page_icon=os.path.join(os.path.dirname(__file__), "resources", "img", "logo.webp"),
    layout="wide"
)

# OpenAI API Key 설정
OPENAI_API_KEY = config.OPENAI_API_KEY

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

# 사이드 바 추가 및 텍스트 입력기 추가
key = OPENAI_API_KEY
#key = st.sidebar.text_input('OPENAI API KEY', type='password', value=OPENAI_API_KEY)

# 로고 파일 경로 설정
logo_path = os.path.join(os.path.dirname(__file__), "resources", "img", "logo.webp")
if not os.path.exists(logo_path):
    #st.sidebar.warning("⚠️ 'logo.webp' 파일이 존재하지 않습니다. 같은 폴더에 추가해주세요.")
    logo_path = None  # 로고가 없으면 기본 아이콘 사용

# JSON 파일 로드 및 텍스트 로더 생성
def load_and_process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 리스트의 각 항목에서 'content'를 추출하여 하나의 문자열로 결합
    return ' '.join(item['content'] for item in data)

# 텍스트 로드 및 분할
file_path_medium = os.path.join(os.path.dirname(__file__), 'data', 'medium', 'medium_articles.json')
content_medium = load_and_process_json(file_path_medium)

# gitbook 폴더의 모든 JSON 파일 로드
gitbook_dir = os.path.join(os.path.dirname(__file__), 'data', 'gitbook')
content_gitbook = ''
for filename in os.listdir(gitbook_dir):
    if filename.endswith('.json'):
        file_path_gitbook = os.path.join(gitbook_dir, filename)
        content_gitbook += load_and_process_json(file_path_gitbook) + " "

# 두 콘텐츠를 결합
content = content_medium + " " + content_gitbook

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
txt_docs = text_splitter.split_text(content)

# Document 객체로 변환
documents = [Document(page_content=doc) for doc in txt_docs]

# 임베딩 생성
embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=key)

# UUID를 이용한 ID 목록 생성
txt_ids = ['rag-nobel-text-' + str(uuid.uuid1()) for _ in range(len(documents))]

# 벡터 스토어 생성
vectorstore = FAISS.from_documents(documents, embeddings, ids=txt_ids)

#검색기 생성
retriever = vectorstore.as_retriever(search_kwargs={'k': 7})

# 인덱스와 ID 출력
print(vectorstore.index_to_docstore_id)

# key 값이 있는 경우만 출력
if key:
    # key가 유효한 경우만 제목 출력
    if check_openai_api_key(key):
        # 프롬프트 생성
        template = """다음 컨텍스트를 사용하여 질문에 답변해주세요. 바이프로스트 네트워크의 서비스들에 대한 고객지원을 담당하는 챗봇으로서, 친절하고 상세하게 답변해주세요. 바이프로스트 서비스와 관련없는 질문은 답하지 마세요.

컨텍스트: {context}

이전 대화 기록:
{history}

질문: {question}

답변:"""
        prompt = PromptTemplate.from_template(template)
        
        # LLM 생성
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=key)
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

        st.title("Bifrost Network Support")

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

# 자동 스크롤 스크립트 (그대로 사용하세요.)
js = '''
<script>
    var body = window.parent.document.querySelector(".main");
    console.log(body);
    body.scrollTop = 0;
</script>
'''

st.components.v1.html(js)