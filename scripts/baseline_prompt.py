import os
import json
import pickle  # 임베딩 결과 저장 및 로드에 사용
import openai
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



openai.api_key = "## API ##"


EMBEDDING_FILE = 'embeddings.pkl'


with open('/data/jaesunghwang/llmtest/dataset/champions_data_final.json', 'r', encoding='utf-8') as f:
    champions_data = json.load(f)

# 챔피언 별로 Document로 변환
documents = []
for champion_name, champion_info in champions_data.items():
    champion_text = f"챔피언: {champion_info['name']}\n" \
                    f"키: {champion_info['key']}\n" \
                    f"설명: {champion_info['description']}\n" \
                    f"스탯: {json.dumps(champion_info['stats'], ensure_ascii=False)}\n" \
                    f"스킬: {json.dumps(champion_info['skills'], ensure_ascii=False)}"
    doc = Document(page_content=champion_text, metadata={"champion_name": champion_name, "key": champion_info['key']})
    documents.append(doc)

# print(documents[0])
# print(documents[0].page_content)
# print(len(documents))
def save_embeddings(embedding_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embedding_dict, f)

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None


embedding_dict = load_embeddings(EMBEDDING_FILE)

# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

if embedding_dict is None:
    # 임베딩이 없다면 새로 생성
    texts = [doc.page_content for doc in documents]
    
    # 임베딩 생성 및 저장
    embeddings = embedding_model.embed_documents(texts)
    embedding_dict = {doc.metadata['champion_name']: embedding for doc, embedding in zip(documents, embeddings)}
    
    # 임베딩 결과 저장
    save_embeddings(embedding_dict, EMBEDDING_FILE)
    print("임베딩을 생성하고 저장했습니다.")
else:
    print("임베딩을 불러왔습니다.")


vector_store = FAISS.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

## k:2로 했을 때 티모와 가렌의 정보를 제대로 가져옴!
# retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# docs = retriever.invoke("티모와 가렌의 정보")
# print(len(docs))
# print(docs)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1.0,
    max_tokens=1024,
    top_p=0.95,
    openai_api_key=openai.api_key
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history."
    # "without the chat history. Do NOT answer the question, "
    # "just reformulate it if needed and otherwise return it as is."
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
    # "the question. If you don't know the answer, say that you "
    # "don't know. Use three sentences maximum and keep the "
    # "answer concise."
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

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# 사용자 질문 처리 함수
def ask_question_conversational(session_id, question):
    # 사용자가 입력한 질문을 conversational_rag_chain을 통해 처리
    result = conversational_rag_chain.invoke(
        {"input": question},
        config={
            "configurable": {"session_id": session_id}
        }
    )
    return result["answer"]



if __name__ == "__main__":
    session_id = "abc123"  # 특정 사용자 세션 ID를 설정 (필요에 따라 동적으로 설정 가능)
    
    while True:
        user_input = input("질문을 입력하세요 ('종료' 입력 시 종료): ")
        if user_input.lower() in ['종료', 'exit', 'quit']:
            break
        
        # 사용자 질문을 처리하여 답변 생성
        answer = ask_question_conversational(session_id, user_input)
        print(f"답변: {answer}\n")

