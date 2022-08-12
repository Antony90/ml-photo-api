import bs4
import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import os
import time

from tqdm import tqdm
import json

# for each category in categories.json, query google images with 
# the category, downloading the first ~300 image results.
# Image previews are saved instead of source images since 
# they aren't needed.
# 
# Guidance for web scraper: https://github.com/ivangrov/Downloading_Google_Images
 
# Load the categories
with open("./dataset/categories.json", "r") as f:
    categories = json.loads(f.read())
         
driver = webdriver.Chrome("./dataset/chromedriver.exe")

# 5 seconds to accept Google's user policy agreement
driver.get("https://google.com/")
time.sleep(5)

print("STARTING: Scraping")
            
print(f"Loaded categories: {categories}")
for category in categories:
    print(f"Scraping: {category=}")

    img_dir = f"./dataset/training/{category}/"
    os.makedirs(img_dir)


    search_URL = f"https://www.google.com/search?q={category}&source=lnms&tbm=isch"
    driver.get(search_URL)

    # Scroll to the bottom 4 times
    for _ in range(4):
        driver.execute_script("window.scrollBy(0,document.body.scrollHeight);")
        time.sleep(1)
    

    pageSoup = bs4.BeautifulSoup(driver.page_source, 'html.parser')
    containers = pageSoup.findAll('div', {'class':"isv-r PNCib MSM1fd BUooTd"} )
    num_containers = len(containers)

    img_ctr = 0
    for i in tqdm(range(1, num_containers)):

        xPath = """//*[@id="islrg"]/div[1]/div[%s]"""%(i)

        preview_img_xpath = """//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[1]/img"""%(i)
        try:   
            preview_img_elem = driver.find_element_by_xpath(preview_img_xpath)
        except NoSuchElementException:
            continue
        
        url = preview_img_elem.get_attribute("src")
        
        try:
            # download and save the image under it's training class directory
            reponse = requests.get(url)
            if reponse.status_code==200:
                with open(os.path.join(img_dir, str(img_ctr)+".jpg"), 'wb') as f:
                    f.write(reponse.content)
            img_ctr += 1
        except:
            pass