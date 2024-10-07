from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저를 백그라운드에서 실행
chrome_options.add_argument("--no-sandbox")  # 샌드박스 모드 비활성화 (권한 문제 해결)
chrome_options.add_argument("--disable-dev-shm-usage")  # /dev/shm 사용 비활성화 (메모리 문제 해결)

# ChromeDriver를 자동으로 관리하여 실행 (Selenium 4.6.0 이상)
driver = webdriver.Chrome(options=chrome_options)

# 동적 콘텐츠가 있는 페이지 열기
url = "https://lol.inven.co.kr/dataninfo/counter/list.php?code=19"
driver.get(url)

# 페이지의 HTML 소스를 가져옴 (JavaScript 로딩 이후)
html_content = driver.page_source

# BeautifulSoup으로 파싱
soup = BeautifulSoup(html_content, "html.parser")

# 'counterContentWrap' 내에서 'class=content' 아래의 텍스트 노드를 추출하는 함수
def extract_text_nodes(soup):
    container = soup.find("div", id="counterContentWrap")
    
    if container:
        content_divs = container.find_all("div", class_="content")
        for content_div in content_divs:
            print("Full content div:")
            print(content_div.prettify())  # HTML 구조를 잘 보기 위해 출력
            
            for child in content_div.children:
                if child.name is None:
                    text = child.strip()
                    if text:
                        print("Extracted text:", text)

# 텍스트 추출 함수 호출
extract_text_nodes(soup)

# 브라우저 닫기
driver.quit()
