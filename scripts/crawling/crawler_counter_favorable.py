import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import re  # 정규 표현식 모듈 임포트

# Step 1: 웹 페이지에서 counter와 easy 챔피언 ID 각각 추출
def get_champion_ids(page_url, icon_type='counter'):
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Error fetching page: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # icon_type에 맞는 "countericon_숫자" 또는 "easyicon_숫자" 형식의 <img> 태그에서 챔피언 ID 추출
    champ_ids = []
    prefix = f'{icon_type}icon_'
    for img_tag in soup.find_all('img', id=lambda x: x and x.startswith(prefix)):
        champ_id = img_tag['id'].split('_')[-1]  # ID에서 숫자 부분만 추출
        champ_ids.append(champ_id)
    
    return champ_ids

# Step 2: Counter API 호출 및 텍스트만 추출
def select_counter(champnum, schamp):
    url = 'https://lol.inven.co.kr/dataninfo/counter/list_ajax.php'
    params = {
        'wchamp': champnum,  # wchamp는 champnum으로 고정
        'schamp': schamp  # schamp는 추출된 counter 챔피언 ID 사용
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # HTML 태그를 제거하고 텍스트만 출력
        text = soup.get_text(separator="\n", strip=True)
        return text
    else:
        # 오류 발생 시 상태 코드와 오류 메시지 출력
        return f"Error {response.status_code}: {response.reason}"

# Step 3: Easy API 호출 및 텍스트만 추출
def select_easy(champnum, wchamp):
    url = 'https://lol.inven.co.kr/dataninfo/counter/list_ajax.php'
    params = {
        'wchamp': wchamp,  # easy 챔피언 ID가 wchamp로 들어감
        'schamp': champnum  # schamp는 champnum으로 고정
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # HTML 태그를 제거하고 텍스트만 출력
        text = soup.get_text(separator="\n", strip=True)
        return text
    else:
        # 오류 발생 시 상태 코드와 오류 메시지 출력
        return f"Error {response.status_code}: {response.reason}"

# 수정된 parse_text 함수
def parse_text(text, our_champion_name, is_easy=False):
    entries = {}
    if is_easy:
        # Easy 데이터에 맞는 정규 표현식 수정
        pattern = re.compile(
            re.escape(our_champion_name) +
            r'\s*>\s*([^\n]+?)\n(.*?)\s*-.*?x\s*\d+',
            re.DOTALL
        )
    else:
        # Counter 데이터에 맞는 정규 표현식 수정
        pattern = re.compile(
            r'([^\n]+?)\s*>\s*' +
            re.escape(our_champion_name) +
            r'\s*(.*?)\s*-.*?x\s*\d+',
            re.DOTALL
        )

    matches = pattern.findall(text)
    for match in matches:
        opponent_name = match[0].strip()
        advice = match[1].strip()
        # 불필요한 내용 제거
        advice = re.sub(r'【.*?】', '', advice)
        advice = advice.strip()
        if opponent_name not in entries:
            entries[opponent_name] = []
        entries[opponent_name].append(advice)
    return entries




# Step 4: 웹 페이지에서 챔피언 번호와 이름을 추출하고 Counter, Easy API 호출
def fetch_champion_data():
    url = 'https://lol.inven.co.kr/dataninfo/counter/list.php?code=19'
    
    # 웹 페이지를 가져와 파싱
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 챔피언 리스트를 담고 있는 ul 태그 선택
    champ_list = soup.find('ul', {'class': 'champList'})

    # 각 챔피언의 번호와 이름 추출
    champion_data = []
    for li in tqdm(champ_list.find_all('li')):
        # id에서 CHAMPNUM 추출
        champ_id = li['id']
        champ_num = champ_id.replace('champList', '').split('_')[0]
        
        # 챔피언 이름 추출
        champ_name_tag = li.select_one('div.champName > nobr > a')
        champ_name = champ_name_tag.text.strip() if champ_name_tag else 'Unknown'
        
        # Step 5: counter와 easy 아이콘이 있는 페이지에서 챔피언 ID 추출
        counter_champ_ids = get_champion_ids(f'https://lol.inven.co.kr/dataninfo/counter/list.php?code={champ_num}', icon_type='counter')
        easy_champ_ids = get_champion_ids(f'https://lol.inven.co.kr/dataninfo/counter/list.php?code={champ_num}', icon_type='easy')

        # Counter 및 Easy 정보 가져오기
        counter_info_raw = [select_counter(champ_num, counter_id) for counter_id in counter_champ_ids]
        easy_info_raw = [select_easy(champ_num, easy_id) for easy_id in easy_champ_ids]
        
        # 텍스트 파싱하여 딕셔너리 형태로 저장
        counter_entries = {}
        for text in counter_info_raw:
            entries = parse_text(text, champ_name, is_easy=False)
            for opponent_name, advices in entries.items():
                if opponent_name not in counter_entries:
                    counter_entries[opponent_name] = []
                counter_entries[opponent_name].extend(advices)
        
        easy_entries = {}
        for text in easy_info_raw:
            entries = parse_text(text, champ_name, is_easy=True)
            for opponent_name, advices in entries.items():
                if opponent_name not in easy_entries:
                    easy_entries[opponent_name] = []
                easy_entries[opponent_name].extend(advices)
        
        # 챔피언 데이터 저장
        champion_data.append({
            'champion_id': champ_num,
            'champion_name': champ_name,
            'counter': counter_entries,
            'easy': easy_entries
        })
    
    return champion_data


# 실행
if __name__ == "__main__":
    champ_data = fetch_champion_data()
    
    # 챔피언 데이터를 JSON 파일로 저장
    with open('champion_data.json', 'w', encoding='utf-8') as f:
        json.dump(champ_data, f, ensure_ascii=False, indent=4)
    
    print("챔피언 데이터가 'champion_data.json' 파일에 저장되었습니다.")
