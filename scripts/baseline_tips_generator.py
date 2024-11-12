# from openai import OpenAI
# import json

# client = OpenAI(api_key='api')

# champions = ["가렌", "티모"]  # 전체 챔피언 목록 입력

# def generate_tip(champion, opponent):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", 
#              "content": "너는 리그 오브 레전드 게임의 전문가야. 나의 질문에 대해 웹 기반의 가장 정확한 정보를 알려줘야해."},
#             {"role": "user", 
#              "content": f"내가 {champion}로 {opponent}를 상대할 때의 팁을 30단어로 요약해줘."}
#         ],
#         max_tokens=128,
#         temperature=1.0
#     )
#     return response.choices[0].message.content

# champion_tips = {}

# for champ in champions:
#     tips_for_champ = {}
#     for opponent in champions:
#         if champ != opponent:
#             tip = generate_tip(champ, opponent)
#             tips_for_champ[opponent] = tip
#     champion_tips[champ] = tips_for_champ

# with open('champion_tips2.json', 'w', encoding='utf-8') as json_file:
#     json.dump(champion_tips, json_file, ensure_ascii=False, indent=4)

# print("JSON 파일 저장완료")


from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI(api_key='api')

champions = ['가렌', '갈리오', '갱플랭크', '그라가스', '그레이브즈', '그웬', '나르', '나미', '나서스', '나피리', 
             '노틸러스', '녹턴', '누누와 윌럼프', '니달리', '니코', '닐라', '다리우스', '다이애나', '드레이븐', '라이즈', 
             '라칸', '람머스', '럭스', '럼블', '레나타 글라스크', '레넥톤', '레오나', '렉사이', '렐', '렝가', 
             '루시안', '룰루', '르블랑', '리 신', '리븐', '리산드라', '릴리아', '마스터 이', '마오카이', '말자하', 
             '말파이트', '모데카이저', '모르가나', '문도 박사', '미스 포츈', '밀리오', '바드', '바루스', '바이', '베이가', 
             '베인', '벡스', '벨베스', '벨코즈', '볼리베어', '브라움', '브라이어', '브랜드', '블라디미르', '블리츠크랭크', 
             '비에고', '빅토르', '뽀삐', '사미라', '사이온', '사일러스', '샤코', '세나', '세라핀', '세주아니', 
             '세트', '소나', '소라카', '쉔', '쉬바나', '스몰더', '스웨인', '스카너', '시비르', '신 짜오', 
             '신드라', '신지드', '쓰레쉬', '아리', '아무무', '아우렐리온 솔', '아이번', '아지르', '아칼리', '아크샨', 
             '아트록스', '아펠리오스', '알리스타', '애니', '애니비아', '애쉬', '야스오', '에코', '엘리스', '오공', 
             '오로라', '오른', '오리아나', '올라프', '요네', '요릭', '우디르', '우르곳', '워윅', '유미', 
             '이렐리아', '이블린', '이즈리얼', '일라오이', '자르반 4세', '자야', '자이라', '자크', '잔나', '잭스', 
             '제드', '제라스', '제리', '제이스', '조이', '직스', '진', '질리언', '징크스', '초가스', 
             '카르마', '카밀', '카사딘', '카서스', '카시오페아', '카이사', '카직스', '카타리나', '칼리스타', '케넨', 
             '케이틀린', '케인', '케일', '코그모', '코르키', '퀸', '크산테', '클레드', '키아나', '킨드레드', 
             '타릭', '탈론', '탈리야', '탐 켄치', '트런들', '트리스타나', '트린다미어', '트위스티드 페이트', '트위치', '티모', 
             '파이크', '판테온', '피들스틱', '피오라', '피즈', '하이머딩거', '헤카림', '흐웨이']

print(len(champions))
# 토큰 사용량을 추적할 딕셔너리
token_usage = {}

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
    
    # 토큰 사용량 저장
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    
    # 문자열 키로 변경하여 저장
    token_usage[f"{champion} vs {opponent}"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }

    return response.choices[0].message.content

champion_tips = {}

with tqdm(total=len(champions) * (len(champions) - 1), desc="Generating tips") as pbar:
    for champ in champions:
        tips_for_champ = {}
        for opponent in champions:
            if champ != opponent:
                tip = generate_tip(champ, opponent)
                tips_for_champ[opponent] = tip
                pbar.set_description(f"{champ} vs {opponent}")  # 현재 상태 표시
                pbar.update(1)  # 진행 바 업데이트
        champion_tips[champ] = tips_for_champ


with open('champion_tips.json', 'w', encoding='utf-8') as json_file:
    json.dump(champion_tips, json_file, ensure_ascii=False, indent=4)

# 최종 토큰 사용량 합계
total_tokens_sum = sum(item["total_tokens"] for item in token_usage.values())
# 합계를 token_usage 딕셔너리에 추가
token_usage["total_tokens_sum"] = total_tokens_sum

with open('token_usage.json', 'w', encoding='utf-8') as json_file:
    json.dump(token_usage, json_file, ensure_ascii=False, indent=4)

print("JSON 파일 저장완료")

