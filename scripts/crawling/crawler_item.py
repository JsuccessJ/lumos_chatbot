import requests
from bs4 import BeautifulSoup
import json

# URL of the page to scrape
url = "https://lol.inven.co.kr/dataninfo/item/list.php"

# Send an HTTP request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    html_content = response.text
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Locate the table containing the items
    item_table = soup.find('table', {'id': 'itemListTable'})
    
    # Extract item rows
    items = item_table.find_all('tr')

    # Extract item name, price, and description
    item_data = []
    for item in items[1:]:  # Skip header row
        item_name = item.find('td', {'class': 'itemname'}).get_text(strip=True)
        item_price = item.find('td', {'class': 'itemprice'}).get_text(strip=True)
        item_description = item.find('td', {'class': 'itemoption'}).get_text(strip=True)

        item_data.append({
            'name': item_name,
            'price': item_price,
            'description': item_description
        })

    # Save the data into a JSON file
    with open('14.22_items.json', 'w', encoding='utf-8') as f:
        json.dump(item_data, f, ensure_ascii=False, indent=4)

    print("Item data saved to items.json")
else:
    print("Failed to retrieve the page")
