from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import TextLoader  # ✅ 여기 수정됨
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# 🌍 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY가 .env 파일에 없거나 비어 있습니다!")

# 문서 로딩
loader = TextLoader("news.txt", encoding="utf-8")
documents = loader.load()

# 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# 임베딩 및 벡터 DB 생성
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = FAISS.from_documents(docs, embedding)

# RAG 체인 구성
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=openai_api_key),
    retriever=vectordb.as_retriever()
)

# 질문 루프
while True:
    query=input("뉴스에 대해 궁금한 점을 물어보세요 (exit 입력 시 종료): ")
    if query.lower()=="exit":
        break
    answer=qa.invoke({"query":query})
    print(f"\n 답변: {answer}\n")