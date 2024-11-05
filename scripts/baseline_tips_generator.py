from openai import OpenAI
import json

client = OpenAI(api_key='api')

champions = ["가렌", "티모", "그라가스"]  # 전체 챔피언 목록 입력

def generate_tip(champion, opponent):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", 
             "content": "너는 리그 오브 레전드 게임의 전문가야. 나의 질문에 대해 웹 기반의 가장 정확한 정보를 알려줘야해."},
            {"role": "user", 
             "content": f"내가 {champion}로 {opponent}를 상대할 때의 팁을 30단어로 요약해줘."}
        ],
        max_tokens=128,
        temperature=1.0
    )
    return response.choices[0].message.content

champion_tips = {}

for champ in champions:
    tips_for_champ = {}
    for opponent in champions:
        if champ != opponent:
            tip = generate_tip(champ, opponent)
            tips_for_champ[opponent] = tip
    champion_tips[champ] = tips_for_champ

with open('champion_tips2.json', 'w', encoding='utf-8') as json_file:
    json.dump(champion_tips, json_file, ensure_ascii=False, indent=4)

print("JSON 파일 저장완료")
