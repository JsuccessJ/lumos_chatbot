import json

# Load the existing champion data
with open("../data/champions_data2.json", "r", encoding="utf-8") as f:
    champions_data = json.load(f)

# Load the additional counter/easy data
with open("../data/champion_data_counter_favorable.json", "r", encoding="utf-8") as f:
    counter_easy_data = json.load(f)

# Iterate over the counter/easy data to match and merge with the main champion data
for counter_easy_entry in counter_easy_data:
    champion_id = counter_easy_entry["champion_id"]
    
    # Find the champion by champion_id in the main data
    for champion_name, champion_info in champions_data.items():
        if champion_info["champion_id"] == champion_id:
            # Add 'counter' and 'easy' data to the matched champion
            champion_info["counter"] = counter_easy_entry.get("counter", {})
            champion_info["easy"] = counter_easy_entry.get("easy", {})
            break

# Save the merged data to a new JSON file
with open("merged_champions_data.json", "w", encoding="utf-8") as f:
    json.dump(champions_data, f, ensure_ascii=False, indent=4)

print("Data successfully merged into 'merged_champions_data.json'.")
