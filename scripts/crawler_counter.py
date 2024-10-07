import requests
from bs4 import BeautifulSoup
import json

def crawl_counter_info(counter_url):
    """
    카운터 정보 페이지에서 카운터 챔피언과 그 설명을 크롤링하는 함수
    """
    try:
        response = requests.get(counter_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        counter_info = []

        # 챔피언 이름 리스트 크롤링
        champ_name_elements = soup.select("div.champName nobr")
        if not champ_name_elements:
            print(f"Champ names not found for {counter_url}")
            return counter_info, None

        champ_names = [nobr.get_text(strip=True) for nobr in champ_name_elements]

        # 카운터 설명 정보 크롤링 (weakand 섹션)
        counter_content_section = soup.find("div", id="counterContentWrap")
        if not counter_content_section:
            print(f"Counter content not found for {counter_url}")
            return counter_info, None

        descriptions = counter_content_section.find_all("div", class_="weakand")

        # 챔피언 이름과 대응되는 설명을 연결 (각각의 챔피언에 대해 설명이 여러 개 있을 수 있음)
        for i in range(min(len(champ_names), len(descriptions))):
            champ_name = champ_names[i]
            description = descriptions[i].get_text(strip=True)
            counter_info.append({
                "name": champ_name,
                "content": description
            })

        return counter_info, None

    except requests.exceptions.RequestException as e:
        return None, str(e)

def crawl(url, counter_url, ddragon):
    """
    에러 반환 : None, 에러 메시지
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # find .engName (className)
        name = soup.find(class_="engName").get_text().split(",")[0]
        name = name.replace("Kai’Sa", "Kai'Sa").replace("Khazix", "Kha'Zix")  # exceptions
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
        skill_classes = []
        skill_lists = soup.find(class_="skillLists")
        if skill_lists:
            for div in skill_lists.find_all("div", class_=True):
                class_name = div["class"][0]
                skill_classes.append(class_name.replace("skills", "").strip())
        else:
            return None, f"skill_lists not found for {name}"

        skill_descs = []
        for div in soup.find_all("div", class_="skillDesc"):
            skill_descs.append(div.get_text().strip())
            
        skill_names = []
        for div in soup.find_all("div", class_="skillName"):
            skill_names.append(div.get_text().strip())

        # Aphelios의 스킬 처리 (특수 구조: passive, 5개의 무기 설명, Q, W, R)
        combined_skills = {}
        if name == "Aphelios":
            if len(skill_names) == 13 and len(skill_descs) == 13:
                # Aphelios는 스킬이 1개, 5개(무기), 5개(무기 Q), 1개(W), 1개(R)
                combined_skills["Passive"] = f"{skill_names[0]}: {skill_descs[0]}"
                combined_skills["Weapons"] = [f"{skill_names[i]}: {skill_descs[i]}" for i in range(1, 6)]
                combined_skills["Q"] = [f"{skill_names[i]}: {skill_descs[i]}" for i in range(6, 11)]
                combined_skills["W"] = f"{skill_names[11]}: {skill_descs[11]}"
                combined_skills["R"] = f"{skill_names[12]}: {skill_descs[12]}"
            else:
                return None, f"Aphelios skill count mismatch for {name}. Expected 13 skills, found {len(skill_names)} names and {len(skill_descs)} descriptions."

        else:
            # 일반 챔피언들의 스킬 처리 (Passive, Q, W, E, R)
            skill_order = ["Passive", "Q", "W", "E", "R"]
            for i in range(min(len(skill_names), len(skill_descs))):
                combined_skills[skill_order[i]] = f"{skill_names[i]}: {skill_descs[i]}"

        # 카운터 정보 추가
        counter_info, error = crawl_counter_info(counter_url)
        if error:
            return None, error

        champion_info = {
            "name": name,
            "key": key,
            "description": desc,
            "stats": stats,
            "skills": combined_skills,
            "counter_info": counter_info  # 카운터 정보 추가
        }

        return champion_info, None

    except requests.exceptions.RequestException as e:
        return None, str(e)


if __name__ == "__main__":
    base_url = "https://lol.inven.co.kr/dataninfo/champion/detail.php?code={}"
    counter_base_url = "https://lol.inven.co.kr/dataninfo/counter/list.php?code={}"
    
    ddragon = json.load(open("ddragon.json", "r", encoding="utf-8"))
    all_champions = {}
    for i in range(1, 168 + 1):
        print(f"Crawling {i} / 168")
        url = base_url.format(i)
        counter_url = counter_base_url.format(i)
        champion_data, error = crawl(url, counter_url, ddragon)
        if champion_data:
            all_champions[champion_data["name"]] = champion_data
        else:
            print(f"오류 발생: {error}")

    with open("champions_counter.json", "w", encoding="utf-8") as f:
        json.dump(all_champions, f, ensure_ascii=False, indent=4)
