from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import os
import time
import urllib.request

# Set up the web driver
driver_path = '/Users/divay/Downloads/ClassML/Driver/chromedriver' # replace with your driver path
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# URL of the Google Images search page
searches = ["single person in lab coat", "lab tech wearing gloves", "lab tech wearing scrub cap"]
search_url = [query.replace(" ", "+") for query in searches]
urls = [f"https://www.google.com/search?q={url}&tbm=isch" for url in search_url]

counter = 1
for url in urls:
    # Create a folder to store the scraped images
    folder_name = f'search_{counter}_i'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    counter += 1

    # Navigate to the Google Images search page
    driver.get(url)

    # Scroll down to load more images 
    last_height = driver.execute_script('return document.body.scrollHeight')
    while True:
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(2)
        new_height = driver.execute_script('return document.body.scrollHeight')
        if new_height == last_height:
            break
        last_height = new_height

    # Find all image elements on the page
    images = driver.find_elements(By.TAG_NAME,'img')

    # Loop through the images and download the ones that match the keywords
    for i, image in enumerate(images):
        src = image.get_attribute('src')
        alt = image.get_attribute('alt')
        
        if alt and any(keyword in alt.lower() for keyword in ['lab coat', 'scrubs', 'scrub', 'scrub cap', 'gloves']):
            # Download the image and save it to the folder
            if src is not None and isinstance(src, str):
                urllib.request.urlretrieve(src, f"{folder_name}/{i}.jpg")

# Close the web driver
driver.quit()
