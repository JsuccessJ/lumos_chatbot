import os
import json
import pickle
import openai
import asyncio
import uuid  # session_id 자동 생성에 사용
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import concurrent.futures

openai.api_key = "api"

CHAMP_EMBEDDING_FILE = '/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/embeddings_counter.pkl'
ITEM_EMBEDDING_FILE = '/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/embeddings_item.pkl'

# 챔피언 데이터 및 아이템 데이터를 Document로 변환
async def load_champion_and_item_data():
    documents, item_documents = [], []
    
    # 챔피언 데이터 로드
    with open('/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/merged_champions_data.json', 'r', encoding='utf-8') as f:
        champions_data = json.load(f)
    for champion_name, champion_info in champions_data.items():
        champion_text = (
            f"챔피언: {champion_info['name']}\n"
            f"ddragon_key: {champion_info['ddragon_key']}\n"
            f"설명: {champion_info['description']}\n"
            f"스탯: {json.dumps(champion_info['stats'], ensure_ascii=False)}\n"
            f"스킬: {json.dumps(champion_info['skills'], ensure_ascii=False)}\n"
            f"상대하기 어려움(Counter): {json.dumps(champion_info['counter'], ensure_ascii=False)}\n"
            f"상대하기 쉬움(Easy): {json.dumps(champion_info['easy'], ensure_ascii=False)}"
        )
        doc = Document(page_content=champion_text, metadata={"champion_name": champion_name, "key": champion_info['ddragon_key']})
        documents.append(doc)
    
    # 아이템 데이터 로드
    with open('/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/items.json', 'r', encoding='utf-8') as f:
        items_data = json.load(f)
    for item in items_data:
        item_text = f"아이템 이름: {item['name']}\n가격: {item['price']}\n설명: {item['description']}"
        doc = Document(page_content=item_text, metadata={"item_name": item['name']})
        item_documents.append(doc)

    return documents, item_documents

# 비동기적으로 임베딩 파일 로드
async def async_load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# 비동기적으로 임베딩 파일 저장
async def async_save_embeddings(embedding_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embedding_dict, f)

# 임베딩 모델 및 벡터 스토어 설정
async def initialize_embeddings(documents, item_documents):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

    # 챔피언 임베딩 로드 또는 생성
    embedding_dict = await async_load_embeddings(CHAMP_EMBEDDING_FILE)
    if embedding_dict is None:
        texts = [doc.page_content for doc in documents]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            embeddings = await asyncio.get_event_loop().run_in_executor(
                executor, embedding_model.embed_documents, texts
            )
        embedding_dict = {doc.metadata['champion_name']: embedding for doc, embedding in zip(documents, embeddings)}
        await async_save_embeddings(embedding_dict, CHAMP_EMBEDDING_FILE)
        print("챔피언 임베딩을 생성하고 저장했습니다.")
    else:
        print("챔피언 임베딩을 불러왔습니다.")

    # 아이템 임베딩 로드 또는 생성
    item_embedding_dict = await async_load_embeddings(ITEM_EMBEDDING_FILE)
    if item_embedding_dict is None:
        item_texts = [doc.page_content for doc in item_documents]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            item_embeddings = await asyncio.get_event_loop().run_in_executor(
                executor, embedding_model.embed_documents, item_texts
            )
        item_embedding_dict = {doc.metadata['item_name']: embedding for doc, embedding in zip(item_documents, item_embeddings)}
        await async_save_embeddings(item_embedding_dict, ITEM_EMBEDDING_FILE)
        print("아이템 임베딩을 생성하고 저장했습니다.")
    else:
        print("아이템 임베딩을 불러왔습니다.")

    # 벡터 스토어 생성
    all_documents = documents + item_documents
    vector_store = FAISS.from_documents(all_documents, embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 1})

# 세션 ID를 통한 대화 이력 관리
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    chat_history = store[session_id]
    if MAX_CHAT_HISTORY_LENGTH is not None and len(chat_history.messages) > MAX_CHAT_HISTORY_LENGTH:
        chat_history.messages = chat_history.messages[-MAX_CHAT_HISTORY_LENGTH:]
    return chat_history

# 질문에 대한 비동기 응답 스트리밍
async def ask_question_conversational_streaming(session_id, question, conversational_rag_chain):
    async for chunk in conversational_rag_chain.stream(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    ):
        if 'answer' in chunk and chunk['answer']:
            print(chunk["answer"], end='', flush=True)

async def main():
    # 챔피언 및 아이템 데이터를 로드하고 임베딩을 초기화
    documents, item_documents = await load_champion_and_item_data()
    retriever = await initialize_embeddings(documents, item_documents)

    # LLM 설정 및 RAG 체인 구성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, max_tokens=256, top_p=1.0, openai_api_key=openai.api_key, streaming=True)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history."),
        MessagesPlaceholder("chat_history"), ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n\n{context}"),
        MessagesPlaceholder("chat_history"), ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
    )

    # 사용자 입력 대기 및 처리
    while True:
        session_id = str(uuid.uuid4())  # 매 접속마다 새로운 세션 ID 생성
        user_input = input("질문을 입력하세요 ('종료' 입력 시 종료): ")
        if user_input.lower() in ['종료', 'exit', 'quit']:
            break
        await ask_question_conversational_streaming(session_id, user_input, conversational_rag_chain)
        print("\n")

# 비동기 실행
if __name__ == "__main__":
    store = {}
    MAX_CHAT_HISTORY_LENGTH = None
    asyncio.run(main())
