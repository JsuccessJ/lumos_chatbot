import requests
from bs4 import BeautifulSoup
import json

def crawl(url_detail, url_counter, ddragon):
    """
    Fetches champion data from both the detail page and the counter page.
    """
    try:
        # Fetch data from the detail page
        response_detail = requests.get(url_detail)
        response_detail.raise_for_status()
        soup_detail = BeautifulSoup(response_detail.text, "html.parser")

        # Fetch data from the counter page
        response_counter = requests.get(url_counter)
        response_counter.raise_for_status()
        soup_counter = BeautifulSoup(response_counter.text, "html.parser")

        # Process the detail page (existing logic)
        name = soup_detail.find(class_="engName").get_text().split(",")[0]
        name = name.replace("Kai’Sa", "Kai'Sa").replace("Khazix", "Kha'Zix")  # handle exceptions
        for short_name in ddragon["data"]:
            if ddragon["data"][short_name]["name"] == name:
                name = short_name
                break
        else:
            return None, f"Name {name} not found in ddragon data. URL: {url_detail}"

        key = ddragon["data"][name]["key"]
        desc = soup_detail.find(class_="descText").get_text()

        # Parse stats and skills from detail page (existing logic)
        # ... existing logic to extract stats and skills
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






        # New part: Extract counter data from the counter page
        counter_table = soup_counter.find(class_="counterTable")  # Example: locate the counter table
        counters = []
        if counter_table:
            rows = counter_table.find_all("tr")
            for row in rows[1:]:  # Skip header row
                cols = row.find_all("td")
                if len(cols) >= 2:
                    counter_champion = cols[0].text.strip()  # Champion name
                    win_rate = cols[1].text.strip()  # Win rate against this champion
                    counters.append({
                        "champion": counter_champion,
                        "win_rate": win_rate
                    })
        else:
            return None, f"Counter data not found for {name}. URL: {url_counter}"

        # Combine all data into a single dictionary
        champion_info = {
            "name": name,
            "key": key,
            "description": desc,
            "stats": stats,           # Existing stats parsed earlier
            "skills": combined_skills,  # Existing skills parsed earlier
            "counters": counters       # New: Counter data from the second URL
        }

        return champion_info, None

    except requests.exceptions.RequestException as e:
        return None, str(e)



if __name__ == "__main__":
    base_url_detail = "https://lol.inven.co.kr/dataninfo/champion/detail.php?code={}"
    base_url_counter = "https://lol.inven.co.kr/dataninfo/counter/list.php?code={}"
    
    ddragon = json.load(open("ddragon.json", "r", encoding="utf-8"))
    all_champions = {}

    for i in range(1, 168 + 1):
        print(f"Crawling {i} / 168")
        url_detail = base_url_detail.format(i)
        url_counter = base_url_counter.format(i)

        # Call updated crawl function with both URLs
        champion_data, error = crawl(url_detail, url_counter, ddragon)
        
        if champion_data:
            all_champions[champion_data["name"]] = champion_data
        else:
            print(f"Error: {error}")

    # Save the final data to a JSON file
    with open("champions_data_rere.json", "w", encoding="utf-8") as f:
        json.dump(all_champions, f, ensure_ascii=False, indent=4)

