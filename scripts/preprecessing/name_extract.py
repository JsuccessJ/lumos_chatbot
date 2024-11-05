import json

# JSON 파일 열기
with open('/Users/hwangjaesung/jaesung/StudyRoom/Study/lumos_chatbot_fork/lumos_chatbot/data/champion_data_counter_favorable.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# champion_name 필드만 추출하여 리스트 생성
champion_names = [item["champion_name"] for item in data]

print(champion_names)