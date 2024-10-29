import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import re  # For regular expressions

# Load Data Dragon data
with open("ddragon.json", "r", encoding="utf-8") as f:
    ddragon = json.load(f)

# Function to get champion key from name using Data Dragon
def get_champion_key(name, ddragon):
    name = name.replace("Kai’Sa", "Kai'Sa").replace("Khazix", "Kha'Zix")  # Handle exceptions
    for short_name in ddragon["data"]:
        if ddragon["data"][short_name]["name"] == name:
            return ddragon["data"][short_name]["key"]
    return None

# Function to crawl champion details (stats, skills)
def crawl_champion_details(url, ddragon):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Get champion name
        name = soup.find(class_="engName").get_text().split(",")[0]
        key = get_champion_key(name, ddragon)
        if not key:
            return None, f"Name {name} not in ddragon. url : {url}"

        desc = soup.find(class_="descText").get_text()

        # Parse stats table
        stat_table = soup.find(class_="statTable")
        stats = {}
        if stat_table:
            rows = stat_table.find_all("tr")
            for row in rows[1:]:  # Skip header
                cols = row.find_all("td")
                if len(cols) == 3:
                    stat_name = cols[0].text.strip()
                    base_value = cols[1].text.split()[0].strip()
                    growth = cols[1].find("span")
                    growth_value = growth.text.strip("()+-") if growth else None
                    final_value = cols[2].text.strip()
                    stats[stat_name] = {
                        "base": base_value,
                        "growth": growth_value,
                        "final": final_value
                    }
        else:
            return None, f"stat_table not found for {name}"

        # Parse skills
        skill_names = [div.get_text().strip() for div in soup.find_all("div", class_="skillName")]
        skill_descs = [div.get_text().strip() for div in soup.find_all("div", class_="skillDesc")]

        combined_skills = {}
        if name == "Aphelios":
            # Handle Aphelios separately due to unique skill structure
            if len(skill_names) == 13 and len(skill_descs) == 13:
                combined_skills["Passive"] = f"{skill_names[0]}: {skill_descs[0]}"
                combined_skills["Weapons"] = [f"{skill_names[i]}: {skill_descs[i]}" for i in range(1, 6)]
                combined_skills["Q"] = [f"{skill_names[i]}: {skill_descs[i]}" for i in range(6, 11)]
                combined_skills["W"] = f"{skill_names[11]}: {skill_descs[11]}"
                combined_skills["R"] = f"{skill_names[12]}: {skill_descs[12]}"
            else:
                return None, f"Aphelios skill count mismatch for {name}. Expected 13 skills, found {len(skill_names)} names and {len(skill_descs)} descriptions."
        else:
            # General case for other champions
            skill_order = ["Passive", "Q", "W", "E", "R"]
            for i in range(min(len(skill_names), len(skill_descs))):
                combined_skills[skill_order[i]] = {
                    "name": skill_names[i],
                    "description": skill_descs[i]
                }

        champion_info = {
            "champion_name": name,
            "champion_id": key,  # Data Dragon ID
            "description": desc,
            "stats": stats,
            "skills": combined_skills
        }

        return champion_info, None

    except requests.exceptions.RequestException as e:
        return None, str(e)

# Function to get counter and easy champion IDs
def get_champion_ids(page_url, icon_type='counter'):
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Error fetching page: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    champ_ids = []
    prefix = f'{icon_type}icon_'
    for img_tag in soup.find_all('img', id=lambda x: x and x.startswith(prefix)):
        champ_id = img_tag['id'].split('_')[-1]  # Extract the numeric ID
        champ_ids.append(champ_id)

    return champ_ids

# Function to fetch matchup text
def select_matchup(champ_code, opponent_champ_code, is_easy=False):
    url = 'https://lol.inven.co.kr/dataninfo/counter/list_ajax.php'
    params = {
        'wchamp': champ_code if not is_easy else opponent_champ_code,
        'schamp': opponent_champ_code if not is_easy else champ_code
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator="\n", strip=True)
        return text
    else:
        return f"Error {response.status_code}: {response.reason}"

# Function to parse matchup text
def parse_text(text, our_champion_name, is_easy=False):
    entries = {}
    if is_easy:
        # Easy data pattern
        pattern = re.compile(
            re.escape(our_champion_name) +
            r'\s*>\s*([^\n]+?)\n(.*?)\s*-.*?x\s*\d+',
            re.DOTALL
        )
    else:
        # Counter data pattern
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
        # Remove unnecessary content
        advice = re.sub(r'【.*?】', '', advice)
        advice = advice.strip()
        if opponent_name not in entries:
            entries[opponent_name] = []
        entries[opponent_name].append(advice)
    return entries

# Main function to fetch all champion data
def fetch_all_champion_data():
    base_url = 'https://lol.inven.co.kr/dataninfo/champion/detail.php?code={}'
    counter_base_url = 'https://lol.inven.co.kr/dataninfo/counter/list.php?code={}'

    all_champion_data = []
    total_champions = 168  # Total number of champions

    for i in tqdm(range(1, total_champions + 1), desc="Processing Champions"):
        champion_detail_url = base_url.format(i)
        champion_data, error = crawl_champion_details(champion_detail_url, ddragon)

        if champion_data:
            champion_data["code"] = str(i)  # Store the Inven champion code as a string
            champ_code = str(i)  # Use this code for Inven URLs and API parameters
            champ_name = champion_data["champion_name"]

            # Get counter and easy champion IDs
            counter_page_url = counter_base_url.format(champ_code)
            counter_champ_ids = get_champion_ids(counter_page_url, icon_type='counter')
            easy_champ_ids = get_champion_ids(counter_page_url, icon_type='easy')

            # Fetch and parse counter data
            counter_info_raw = [select_matchup(champ_code, counter_id, is_easy=False) for counter_id in counter_champ_ids]
            counter_entries = {}
            for text in counter_info_raw:
                entries = parse_text(text, champ_name, is_easy=False)
                for opponent_name, advices in entries.items():
                    if opponent_name not in counter_entries:
                        counter_entries[opponent_name] = []
                    counter_entries[opponent_name].extend(advices)

            # Fetch and parse easy data
            easy_info_raw = [select_matchup(champ_code, easy_id, is_easy=True) for easy_id in easy_champ_ids]
            easy_entries = {}
            for text in easy_info_raw:
                entries = parse_text(text, champ_name, is_easy=True)
                for opponent_name, advices in entries.items():
                    if opponent_name not in easy_entries:
                        easy_entries[opponent_name] = []
                    easy_entries[opponent_name].extend(advices)

            # Add matchup data to champion_data
            champion_data["counter"] = counter_entries
            champion_data["easy"] = easy_entries

            # Append to the list
            all_champion_data.append(champion_data)
        else:
            print(f"Error fetching champion data: {error}")

    return all_champion_data

if __name__ == "__main__":
    all_data = fetch_all_champion_data()

    # Save data to JSON file
    with open("combined_champion_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print("Champion data has been saved to 'combined_champion_data.json'.")
