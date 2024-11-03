import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import re

# Step 1: 기본 정보 크롤링

def crawl(url, ddragon):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # 챔피언 기본 정보 가져오기
        name = soup.find(class_="engName").get_text().split(",")[0]
        name = name.replace("Kai’Sa", "Kai'Sa").replace("Khazix", "Kha'Zix")  # 예외 처리
        for short_name in ddragon["data"]:
            if ddragon["data"][short_name]["name"] == name:
                name = short_name
                break
        else:
            return None, f"Name {name} not in ddragon. url : {url}"

        key = ddragon["data"][name]["key"]
        desc = soup.find(class_="descText").get_text()
        stat_table = soup.find(class_="statTable")

        # 통계 테이블 파싱
        stats = {}
        if stat_table:
            rows = stat_table.find_all("tr")
            for row in rows[1:]:  # 첫 번째 행은 헤더
                cols = row.find_all("td")
                if len(cols) == 3:
                    stat_name = cols[0].text.strip()
                    base_value = cols[1].text.split()[0].strip()
                    growth = cols[1].find("span")
                    growth_value = growth.text.strip("()+-") if growth else None
                    final_value = cols[2].text.strip()

                    stats[stat_name] = {"기본값": base_value, "성장": growth_value, "최종값": final_value}
        else:
            return None, f"stat_table not found for {name}"

        # 스킬 이름과 설명 파싱
        skill_names = [div.get_text().strip() for div in soup.find_all("div", class_="skillName")]
        skill_descs = [div.get_text().strip() for div in soup.find_all("div", class_="skillDesc")]

        # Aphelios의 스킬 처리 (특수 구조)
        combined_skills = {}
        if name == "Aphelios":
            if len(skill_names) == 13 and len(skill_descs) == 13:
                combined_skills["Passive"] = f"{skill_names[0]}: {skill_descs[0]}"
                combined_skills["Weapons"] = [f"{skill_names[i]}: {skill_descs[i]}" for i in range(1, 6)]
                combined_skills["Q"] = [f"{skill_names[i]}: {skill_descs[i]}" for i in range(6, 11)]
                combined_skills["W"] = f"{skill_names[11]}: {skill_descs[11]}"
                combined_skills["R"] = f"{skill_names[12]}: {skill_descs[12]}"
            else:
                return None, f"Aphelios skill count mismatch for {name}. Expected 13 skills, found {len(skill_names)} names and {len(skill_descs)} descriptions."
        else:
            # 일반 챔피언 스킬 처리
            skill_order = ["Passive", "Q", "W", "E", "R"]
            for i in range(min(len(skill_names), len(skill_descs))):
                combined_skills[skill_order[i]] = f"{skill_names[i]}: {skill_descs[i]}"

        champion_info = {
            "name": name,
            "key": key,
            "description": desc,
            "stats": stats,
            "skills": combined_skills
        }

        return champion_info, None

    except requests.exceptions.RequestException as e:
        return None, str(e)

# Step 2: Counter와 Easy 정보 가져오기

def get_champion_ids(page_url, icon_type='counter'):
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Error fetching page: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.content, 'html.parser')
    champ_ids = []
    prefix = f'{icon_type}icon_'
    for img_tag in soup.find_all('img', id=lambda x: x and x.startswith(prefix)):
        champ_id = img_tag['id'].split('_')[-1]
        champ_ids.append(champ_id)
    
    return champ_ids

def select_counter_or_easy(champnum, other_champ, is_easy=False):
    url = 'https://lol.inven.co.kr/dataninfo/counter/list_ajax.php'
    params = {
        'wchamp': champnum if not is_easy else other_champ,
        'schamp': other_champ if not is_easy else champnum
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator="\n", strip=True)
    else:
        return f"Error {response.status_code}: {response.reason}"

def parse_text(text, our_champion_name, is_easy=False):
    entries = {}
    if is_easy:
        pattern = re.compile(re.escape(our_champion_name) + r'\s*>\s*([^\n]+?)\n(.*?)\s*-.*?x\s*\d+', re.DOTALL)
    else:
        pattern = re.compile(r'([^\n]+?)\s*>\s*' + re.escape(our_champion_name) + r'\s*(.*?)\s*-.*?x\s*\d+', re.DOTALL)
    matches = pattern.findall(text)
    for match in matches:
        opponent_name = match[0].strip()
        advice = match[1].strip()
        advice = re.sub(r'【.*?】', '', advice).strip()
        if opponent_name not in entries:
            entries[opponent_name] = []
        entries[opponent_name].append(advice)
    return entries

# Step 3: 데이터 결합 및 통합 저장

def fetch_all_champion_data():
    base_url = "https://lol.inven.co.kr/dataninfo/champion/detail.php?code={}"
    ddragon = json.load(open("ddragon.json", "r", encoding="utf-8"))
    champion_data = {}

    # 기본 정보 수집
    for i in range(1, 3 + 1):
        print(f"Crawling {i} / 3")
        url = base_url.format(i)
        champion_info, error = crawl(url, ddragon)
        if champion_info:
            champion_data[champion_info["name"]] = champion_info
        else:
            print(f"오류 발생: {error}")

    # Counter와 Easy 정보 추가
    for champ_name, champ_info in champion_data.items():
        champ_num = champ_info["key"]
        counter_champ_ids = get_champion_ids(f'https://lol.inven.co.kr/dataninfo/counter/list.php?code={champ_num}', icon_type='counter')
        easy_champ_ids = get_champion_ids(f'https://lol.inven.co.kr/dataninfo/counter/list.php?code={champ_num}', icon_type='easy')

        counter_entries = {}
        for counter_id in counter_champ_ids:
            text = select_counter_or_easy(champ_num, counter_id, is_easy=False)
            entries = parse_text(text, champ_name, is_easy=False)
            for opponent_name, advices in entries.items():
                if opponent_name not in counter_entries:
                    counter_entries[opponent_name] = []
                counter_entries[opponent_name].extend(advices)

        easy_entries = {}
        for easy_id in easy_champ_ids:
            text = select_counter_or_easy(champ_num, easy_id, is_easy=True)
            entries = parse_text(text, champ_name, is_easy=True)
            for opponent_name, advices in entries.items():
                if opponent_name not in easy_entries:
                    easy_entries[opponent_name] = []
                easy_entries[opponent_name].extend(advices)

        champ_info["counter"] = counter_entries
        champ_info["easy"] = easy_entries

    return champion_data

if __name__ == "__main__":
    all_champ_data = fetch_all_champion_data()
    with open('merged_champion_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_champ_data, f, ensure_ascii=False, indent=4)
    print("챔피언 데이터가 'merged_champion_data.json' 파일에 저장되었습니다.")
