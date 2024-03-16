from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pandas as pd
import requests
from tqdm import tqdm
import time


data = pd.read_csv('./genre_11_tempo_4.csv')
data3 = data[['singer','song_name']]
# data3= data2.iloc[688:690]



options = webdriver.ChromeOptions()
options.add_argument("headless") 
options.add_argument('--window-size=1920,1080')
service = Service(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# driver.maximize_window()
youtube = "https://www.youtube.com/results?search_query="
url_link = []
text1_list = []
text2_list = []
text3_list = []
text4_list = []
text5_list = []
for idx, value in tqdm(data3.iterrows()):

    keyword = value['song_name'] + " " + value['singer']
    search_keyword_encode = requests.utils.quote(keyword)
    url = youtube + search_keyword_encode
    try:
        driver.get(url)
        
        element1 = WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#video-title')))
        next = element1.get_attribute("href")
        url_link.append(next)
    
        driver.get(next)
    except:
        url_link.append(None)
        text1_list.append(None)
        text2_list.append(None)
        text3_list.append(None)
        text4_list.append(None)
        text5_list.append(None)
        continue

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")

    try:
        WebDriverWait(driver, 2).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#content-text')))
        
        element2 = driver.find_elements(By.CSS_SELECTOR,'#content-text')
        text1 = element2[0].text
        text2 = element2[1].text
        text3 = element2[2].text
        text4 = element2[3].text
        text5 = element2[4].text

    except:
        text1 = None
        text2 = None
        text3 = None
        text4 = None
        text5 = None
    text1_list.append(text1)
    text2_list.append(text2)
    text3_list.append(text3)
    text4_list.append(text4)
    text5_list.append(text5)
driver.quit()

data3['url'] = url_link
data3['comments1']=text1_list
data3['comments2']=text2_list
data3['comments3']=text3_list
data3['comments4']=text4_list
data3['comments5']=text5_list
data3.to_csv('./add_url.csv',index=False)


