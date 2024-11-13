import os
import json
import pickle
import openai
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel

# OpenAI API 키 설정
openai.api_key = "api"

# FastAPI 애플리케이션 초기화
app = FastAPI()

# 챔피언 및 아이템 데이터 파일 경로
CHAMP_EMBEDDING_FILE = '/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/14.22_embeddings_counter.pkl'
ITEM_EMBEDDING_FILE = '/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/14.22_embeddings_item.pkl'

# 챔피언 데이터 로드
with open('/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/14.22/14.22_merged_champions_data.json', 'r', encoding='utf-8') as f:
    champions_data = json.load(f)

# 챔피언 데이터를 Document로 변환
documents = []
for champion_name, champion_info in champions_data.items():
    champion_text = f"챔피언: {champion_info['name']}\n" \
                    f"ddragon_key: {champion_info['ddragon_key']}\n" \
                    f"설명: {champion_info['description']}\n" \
                    f"스탯: {json.dumps(champion_info['stats'], ensure_ascii=False)}\n" \
                    f"스킬: {json.dumps(champion_info['skills'], ensure_ascii=False)}\n" \
                    f"상대하기 어려움(Counter): {json.dumps(champion_info['counter'], ensure_ascii=False)}\n" \
                    f"상대하기 쉬움(Easy): {json.dumps(champion_info['easy'], ensure_ascii=False)}"
    doc = Document(page_content=champion_text, metadata={"champion_name": champion_name, "key": champion_info['ddragon_key']})
    documents.append(doc)

# 아이템 데이터 로드
with open('/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/14.22/14.22_items.json', 'r', encoding='utf-8') as f:
    items_data = json.load(f)

# 아이템 데이터를 Document로 변환
item_documents = []
for item in items_data:
    item_text = f"아이템 이름: {item['name']}\n" \
                f"가격: {item['price']}\n" \
                f"설명: {item['description']}"
    doc = Document(page_content=item_text, metadata={"item_name": item['name']})
    item_documents.append(doc)

# 임베딩 저장 및 불러오기 함수
def save_embeddings(embedding_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embedding_dict, f)

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# OpenAI 임베딩 모델 설정
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

# 챔피언 임베딩 로드 또는 생성
embedding_dict = load_embeddings(CHAMP_EMBEDDING_FILE)
if embedding_dict is None:
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.embed_documents(texts)
    embedding_dict = {doc.metadata['champion_name']: embedding for doc, embedding in zip(documents, embeddings)}
    save_embeddings(embedding_dict, CHAMP_EMBEDDING_FILE)
    print("챔피언 임베딩을 생성하고 저장했습니다.")
else:
    print("챔피언 임베딩을 불러왔습니다.")

# 아이템 임베딩 로드 또는 생성
item_embedding_dict = load_embeddings(ITEM_EMBEDDING_FILE)
if item_embedding_dict is None:
    item_texts = [doc.page_content for doc in item_documents]
    item_embeddings = embedding_model.embed_documents(item_texts)
    item_embedding_dict = {doc.metadata['item_name']: embedding for doc, embedding in zip(item_documents, item_embeddings)}
    save_embeddings(item_embedding_dict, ITEM_EMBEDDING_FILE)
    print("아이템 임베딩을 생성하고 저장했습니다.")
else:
    print("아이템 임베딩을 불러왔습니다.")

# 벡터 스토어 생성 및 검색기 설정
all_documents = documents + item_documents
vector_store = FAISS.from_documents(all_documents, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.0,
    max_tokens=256,
    top_p=1.0,
    openai_api_key=openai.api_key,
    streaming=True
)

# 프롬프트 설정
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 세션 관리 및 대화 이력 관리 함수
store = {}
MAX_CHAT_HISTORY_LENGTH = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    chat_history = store[session_id]
    if MAX_CHAT_HISTORY_LENGTH is not None and len(chat_history.messages) > MAX_CHAT_HISTORY_LENGTH:
        chat_history.messages = chat_history.messages[-MAX_CHAT_HISTORY_LENGTH:]
    return chat_history

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# POST 요청으로 질문 받기
class QuestionRequest(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question
        session_id = request.session_id

        response = ""
        for chunk in conversational_rag_chain.stream(
            {"input": question},
            config={
                "configurable": {"session_id": session_id}
            }
        ):
            if 'answer' in chunk and chunk['answer']:
                response += chunk["answer"]
        
        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket을 통한 실시간 스트리밍 응답
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = "abc123"
    
    while True:
        data = await websocket.receive_text()
        if data.lower() in ["exit", "quit", "종료"]:
            break
        
        response = ""
        for chunk in conversational_rag_chain.stream(
            {"input": data},
            config={
                "configurable": {"session_id": session_id}
            }
        ):
            if 'answer' in chunk and chunk['answer']:
                await websocket.send_text(chunk["answer"])
        
    await websocket.close()

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
