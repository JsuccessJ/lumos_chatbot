import os
import json
import pickle  # 임베딩 결과 저장 및 로드에 사용
import openai
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

openai.api_key = "api"

EMBEDDING_FILE = '/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/embeddings_counter.pkl'
with open('/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot/data/merged_champions_data.json', 'r', encoding='utf-8') as f:
    champions_data = json.load(f)

# 챔피언 별로 Document로 변환
documents = []
for champion_name, champion_info in champions_data.items():
    champion_text = f"챔피언: {champion_info['name']}\n" \
                    f"ddragon_key: {champion_info['ddragon_key']}\n" \
                    f"설명: {champion_info['description']}\n" \
                    f"스탯: {json.dumps(champion_info['stats'], ensure_ascii=False)}\n" \
                    f"스킬: {json.dumps(champion_info['skills'], ensure_ascii=False)}\n" \
                    f"상대하기 어려움(Counter): {json.dumps(champion_info['counter'], ensure_ascii=False)}" \
                    f"상대하기 쉬움(Easy): {json.dumps(champion_info['easy'], ensure_ascii=False)}"
    doc = Document(page_content=champion_text, metadata={"champion_name": champion_name, "key": champion_info['ddragon_key']})
    documents.append(doc)

# items.json 파일 로드
with open('/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot/data/items.json', 'r', encoding='utf-8') as f:
    items_data = json.load(f)

# 아이템 데이터를 Document로 변환
item_documents = []
for item in items_data:
    item_text = f"아이템 이름: {item['name']}\n" \
                f"가격: {item['price']}\n" \
                f"설명: {item['description']}"
    doc = Document(page_content=item_text, metadata={"item_name": item['name']})
    item_documents.append(doc)


def save_embeddings(embedding_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embedding_dict, f)

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

embedding_dict = load_embeddings(EMBEDDING_FILE)
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

if embedding_dict is None:
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.embed_documents(texts)
    embedding_dict = {doc.metadata['champion_name']: embedding for doc, embedding in zip(documents, embeddings)}
    save_embeddings(embedding_dict, EMBEDDING_FILE)
    print("임베딩을 생성하고 저장했습니다.")
else:
    print("임베딩을 불러왔습니다.")

vector_store = FAISS.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.0,
    max_tokens=256,
    top_p=0.95,
    openai_api_key=openai.api_key,
    streaming=True  # 스트리밍 활성화
)

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
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

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

store = {}
MAX_CHAT_HISTORY_LENGTH = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    chat_history = store[session_id]
    if MAX_CHAT_HISTORY_LENGTH is not None and len(chat_history.messages) > MAX_CHAT_HISTORY_LENGTH:
        chat_history.messages = chat_history.messages[-MAX_CHAT_HISTORY_LENGTH:]
    return chat_history

def get_recent_chat_history(session_id: str, n: int):
    chat_history = get_session_history(session_id)
    return chat_history.messages[-n:]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history, # BaseChatMessageHistory 객체임
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# baseline prompt 와 달라진 부분
def ask_question_conversational_streaming(session_id, question):
    # 체인 실행 중 데이터 필터링
    for chunk in conversational_rag_chain.stream(
        {"input": question},
        config={
            "configurable": {"session_id": session_id}
        }
    ):
        # 'answer' 키가 있을 경우에만 출력
        if 'answer' in chunk and chunk['answer']:
            print(chunk["answer"], end='', flush=True)

# 대화형 스트리밍 실행 코드
if __name__ == "__main__":
    session_id = "abc123"
    while True:
        user_input = input("질문을 입력하세요 ('종료' 입력 시 종료): ")
        if user_input.lower() in ['종료', 'exit', 'quit']:
            break
        ask_question_conversational_streaming(session_id, user_input)
        print("\n")


