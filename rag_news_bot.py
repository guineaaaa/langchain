from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
import uuid

# 1. 환경 설정 및 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2. 뉴스 문서 로드 및 전처리
loader = TextLoader("news.txt", encoding="utf-8")
documents = loader.load() # 문서 로드드

# 텍스트 분할: 500자 단위, 50자 중첩첩
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents) # 문서, 리스트로 분할됨됨

# 임베딩 모델 준비 (OPEN AI)
embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    openai_api_key=openai_api_key
)

# 벡터 스토어 생성 (FAISS)
vectordb = FAISS.from_documents(docs, embedding)
print(vectordb)

# 리트리버 (벡터 검색기 생성성)
retriever = vectordb.as_retriever()

# 3. 세션 히스토리 저장소
store = {} # 세션 별로 히스토리를 저장하는 딕셔너리 

# 특정 세션 ID에 대한 히스토리 반환 (없으면 새로 생성)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. 프롬프트 템플릿 설정정
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about news."),
    MessagesPlaceholder(variable_name="chat_history"), # 이전 대화 히스토리 포함함
    ("human", "{input}"), # 사용자의 질문문
    ("system", "Context:\n{context}") # 검색된 문서 내용 포함
])

# 5. 검색기 + 생성기 구성
llm = ChatOpenAI(openai_api_key=openai_api_key)

# 히스토리를 고려한 검색기 구성 (이전 대화를 기반으로 관련 문서 검색 강화화)
retriever_with_memory = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=prompt
)

# 검색된 문서를 기반으로 답변 생성하는 체인인
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    output_parser=StrOutputParser() #결과를 문자열로 파싱싱
)

# 검색기 + 문서 기반 생성기 조합합
rag_chain = create_retrieval_chain(retriever_with_memory, document_chain)

# 6. 히스토리 포함 체인 구성
chain_with_history = RunnableWithMessageHistory(
    rag_chain, # RAG 체인 (검색+생성)
    get_session_history, # 세션 기반 히스토리 불러오기 함수수
    input_messages_key="input",  # 사용자 질문이 input으로 전달됨 (사용자 질문 입력 메세지 키)
    history_messages_key="chat_history",# 대화 히스토리 키
)

# 7. 사용자 입력 루프
print("뉴스 기반 Q&A 시스템입니다. 질문을 입력하세요 (exit 입력 시 종료됩니다).")

session_id = str(uuid.uuid4())  # 세션 ID 생성

# 무한 루프 생성성
while True:
    query = input("❓ 질문: ")
    if query.lower() == "exit":
        break

    try:
        # 질문을 히스토리 포함 체인에 전달하여 응답 받기
        response = chain_with_history.invoke(
            {"input": query},  # input key를 명시적으로 지정해야 함
            config={"configurable": {"session_id": session_id}}
        )
        # print("🔎 전체 응답:", response), 디버깅용 

        print("🧠 답변:", response.get("answer", "답변을 생성하지 못했습니다"))
    except Exception as e:
        print("❌ 오류 발생:", e)
