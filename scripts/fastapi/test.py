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
from fastapi.middleware.cors import CORSMiddleware  # 추가
from pydantic import BaseModel

# OpenAI API 키 설정
openai.api_key = "api"

# FastAPI 애플리케이션 초기화
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://lolpago.gg"], # Todo: 배포할때 localhost:3000 지우기 
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 챔피언 및 아이템 데이터 파일 경로
CHAMP_EMBEDDING_FILE = 'C:/lumos_chatbot/data/14.22_embeddings_counter.pkl'
ITEM_EMBEDDING_FILE = 'C:/lumos_chatbot/data/14.22_embeddings_item.pkl'

# 챔피언 데이터 로드
with open('C:/lumos_chatbot/data/14.22/14.22_merged_champions_data.json', 'r', encoding='utf-8') as f:
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
with open('C:/lumos_chatbot/data/14.22/14.22_items.json', 'r', encoding='utf-8') as f:
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
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # 문서 3개로 변경

# LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=1.0,
    max_tokens=512, # 512 변경
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

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question."
#     "\n\n"
#     "{context}"
# )

# 프롬프트 변경
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "You just have to answer what I asked. "
    "Provide the response in the following organized exampled format:\n\n"
"""
챔피언: Akali  
==========================

기본 정보:
- 이름: Akali
- 설명: 아칼리는 은신 능력을 가진 암살 챔피언으로 적들 사이를 종횡무진 누빌 수 있는 챔피언입니다. 아칼리의 스킬들은 대부분 거리를 벌리거나 좁히는데 특화되어 있으며, 이를 활용하여 추가 대미지를 가하는 것이 중요합니다. 스킬을 활용하면 특정 지역에 지속적으로 은신할 수 있으며, 어그로 핑퐁으로 전투 승리에 기여할 수 있습니다.

스탯:
- 체력: 기본값 600, 성장 119, 최종값 2623
- 마나: 기본값 200, 성장 0, 최종값 200
- 공격력: 기본값 62, 성장 3, 최종값 113
- 공격 속도: 기본값 0.625, 성장 3.2%, 최종값 0.965
- 5초당 체력 회복: 기본값 5, 성장 0, 최종값 5
- 5초당 마나 회복: 기본값 50, 성장 0, 최종값 50
- 방어력: 기본값 23, 성장 4, 최종값 91
- 마법 저항력: 기본값 37, 성장 2, 최종값 71
- 이동 속도: 기본값 345, 성장 없음, 최종값 345
- 사정 거리: 기본값 125, 성장 없음, 최종값 125

스킬:
1. Passive - 암살자의 표식 (Assassin's Mark):
   - 스킬 공격으로 챔피언에게 피해를 입히면 해당 챔피언의 주변에 원이 생깁니다.
   - 아칼리가 이 원의 경계를 넘어가면 다음 공격의 사거리가 두 배로 증가하며, 35~182 (+0.6 추가AD) (+0.55AP)의 추가 마법 피해를 입힙니다.

2. Q - 오연투척검 (Five Point Strike):
   - 단검을 부채꼴 모양으로 던져 45/70/95/120/145 (+0.65 AD) (+0.6 AP)의 마법 피해를 입힙니다.
   - 사거리 끝에 있는 적들은 잠시 둔화됩니다. (사정거리: 500)

3. W - 황혼의 장막 (Twilight Shroud):
   - 연막탄을 떨어뜨려 5/5.5/6/6.5/7초 동안 지속되는 연막을 퍼뜨립니다.
   - 연막 안에 있는 동안 아칼리는 투명해지며, 아칼리의 이동 속도가 30/35/40/45/50% 증가했다가 2초에 걸쳐 원래대로 돌아옵니다.
   - 황혼의 장막이 활성화된 동안 아칼리의 기력이 100 증가합니다. 은신 - 투명: 근처의 적 포탑 또는 절대 시야만이 아칼리의 모습을 드러낼 수 있습니다. (사정거리: 250)

4. E - 표창곡예 (Shuriken Flip):
   - 뒤로 공중제비를 돌며 전방으로 표창을 던져 21/42/63/84/105 (+0.3AD) (+0.33AP)의 마법 피해를 입히고, 표창에 맞은 첫 번째 적이나 연막에 표식을 남깁니다.
   - 재사용 시: 표식을 남긴 대상에게 돌진해 49/98/147/196/245 (+0.7AD) (+0.77AP)의 마법 피해를 줍니다. (사정거리: 650)

5. R - 무결처형 (Perfect Execution):
   - 두 번의 돌진: 첫 번째 돌진은 지정한 적을 뛰어넘어 아칼리가 돌진하는 모든 적에게 110/220/330 (+0.5 추가 AD) (+0.3AP)의 물리 피해를 입힙니다.
   - 2.5초가 지나면 다시 돌진할 수 있습니다. 두 번째 돌진은 적들을 관통하여 대상이 잃은 체력에 비례해 70/140/210 (+0.3 AP) ~ 210/420/630 (+0.9 AP)의 마법 피해를 입힙니다. (사정거리: 750)

카운터 챔피언:
- 베이가:
  - 베이가의 광역 스킬들은 아칼리의 장막을 무력화할 수 있으며, 지평선의 스킬 적중으로 인해 아칼리의 움직임을 제한할 수 있습니다. 중후반에 아칼리의 접근을 제한하면서 효과적인 카운터로 작용합니다.
- 아리:
  - 아리는 아칼리의 장막을 무시하는 스킬을 보유하고 있으며, 기동성 있는 스킬들로 아칼리의 접근을 피하기 좋습니다. 라인 푸쉬와 로밍을 통해 아칼리를 압박할 수 있는 챔피언입니다.
- 갈리오:
  - 갈리오는 돌진 후 장막을 무시하는 광역 도발로 아칼리의 기습을 막아낼 수 있으며, 강력한 갱 호응과 맵 전역에 영향을 미치는 궁극기로 아칼리에게 큰 위협이 됩니다.

쉬운 상대:
- 칼리스타:
  - 아칼리의 장막은 칼리스타의 평타를 막아내기 매우 유리하며, 아칼리는 칼리스타를 상대로 전 구간에서 우세한 편입니다. 필밴을 추천합니다.
- 사이온:
  - CS를 챙기며 극도로 사리면 6레벨 이후부터 딜교환에서 우위를 점할 수 있습니다. 사이온의 Q를 뺀 타이밍에 들어가면 효과적인 딜교환이 가능합니다.

"""
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
