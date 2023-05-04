from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

##########
PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)
##########

# Gets Prizepicks Website
driver.get("https://app.prizepicks.com/")

# Waits and clicks on popup
wait = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CLASS_NAME, "close")))
driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div/div/div[3]/button").click()
time.sleep(2)

ppPlayers = []

# Clicking on NBA Tab
driver.find_element(By.XPATH, "//div[@class='name'][normalize-space()='NBA']").click()
time.sleep(2)

# Waiting until stat_container is visible
stat_container = WebDriverWait(driver, 1).until(EC.visibility_of_element_located((By.CLASS_NAME, "stat-container")))

# Finds all the stat elements in the stat-container such as pts, rebs, pra...
categories = driver.find_element(By.CSS_SELECTOR, ".stat-container").text.split('\n')

for category in categories:
    driver.find_element(By.XPATH, f"//div[text()='{category}']").click()

    projectionsPP = WebDriverWait(driver, 5).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".projection")))

    for projections in projectionsPP:
        name = projections.find_element(By.CLASS_NAME, "name").text
        pts = projections.find_element(By.CLASS_NAME, "presale-score").get_attribute('innerHTML')
        proptype = projections.find_element(By.CLASS_NAME, "text").get_attribute('innerHTML')

        players = {
            'Name': name,
            'Value': pts,
            'Prop': proptype.replace("<wbr>", "")
        }

        ppPlayers.append(players)

dfProps = pd.DataFrame(ppPlayers)

filepath = "/data/PPApril27.csv"
dfProps.to_csv(filepath, index=False)

from nba_api.stats.static.players import *
import pandas as pd

df = pd.read_csv(filepath)

player_ids = []

for index, row in df.iterrows():
    player_name = row["Name"]
    result = find_players_by_full_name(player_name)
    if result:
        player_id = result[0]['id']
    else:
        player_id = None

    player_ids.append(player_id)

df['ID'] = player_ids
filepath2 = "C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\PPApril27wID.csv"
df.sort_values("Name").to_csv(filepath2)



