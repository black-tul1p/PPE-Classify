from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver import Chrome
import os, shutil, time
from tqdm import tqdm
import urllib.request

# Store queries and keywords
searches = ["one person wearing lab coat", "lab tech wearing gloves", "lab tech wearing scrub cap"]
desc = ["lab_coat", "gloves", "scrub_cap"]
keywords = ['lab coat', 'scrubs', 'scrub', 'scrub cap', 'gloves']

# Make urls for queries
search_url = [query.replace(" ", "+") for query in searches]
urls = [f"https://www.google.com/search?q={url}&tbm=isch" for url in search_url]

# Set up the web driver
driver_path = '/Users/divay/Downloads/ClassML/Driver/chromedriver' # replace with your driver path
root_path = os.path.join('.', 'Images')
service = Service(driver_path)
driver = Chrome(service=service)
driver_wait = 1

# Scrape images
counter = 1
for url in urls:
    # Display current query
    print(f"\n>>> Scraping images for {desc[counter-1]}")
    counter += 1

    # Create a folder to store the scraped images
    folder_path = os.path.join(root_path, desc[counter-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    # Navigate to the Google Images search page
    driver.get(url)

    # Scroll down to load more images 
    last_height = driver.execute_script('return document.body.scrollHeight')
    while True:
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(2)
        new_height = driver.execute_script('return document.body.scrollHeight')
        if new_height == last_height:
            # Find the "See more anyway" span element and click it
            try:
                WebDriverWait(driver, driver_wait).until(EC.element_to_be_clickable((By.XPATH, "//div[contains(., 'The rest of the results')]//span[contains(., 'See more anyway')]"))).click()
            except:
                print("Trying to load more results...")
                try:                    
                    WebDriverWait(driver, driver_wait).until(EC.element_to_be_clickable((By.XPATH, "//input[@value='Show more results']"))).click()
                except:
                    print("Reached end of results.")
                    break
        last_height = new_height

    # Find all image elements on the page
    images = driver.find_elements(By.TAG_NAME,'img')

    # Loop through the images and download the ones that match the keywords
    pbar = tqdm(range(len(images)), unit="images")
    for i in pbar:
        image = images[i]
        src = image.get_attribute('src')
        alt = image.get_attribute('alt')
        
        if alt and any(keyword in alt.lower() for keyword in keywords):
            # Download the image and save it to the folder
            if src is not None and isinstance(src, str):
                urllib.request.urlretrieve(src, os.path.join(folder_path, f"{i}.jpg"))

# Close the web driver
driver.quit()
