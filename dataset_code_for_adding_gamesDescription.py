import pandas as pd
import requests
from bs4 import BeautifulSoup

df = pd.read_csv('/Users/constantinossapkas/Downloads/Video Games Dataset.csv')


def get_description(name):
    url_rawg = "https://gamesdb.launchbox-app.com/games/details/133984-300"
    url_giant_bomb = 'https://www.giantbomb.com/api/search/?api_key=f02bf4431ec531af7b085876f3876e848d7c34b0&format=json&query=' + name + '&resources=game'
    headers = {'user-agent': 'your user agent'}
    r = requests.get(url_rawg, headers=headers)
    if r.status_code!= 200:
        return ''
    data = r.json()
    try:
        desc = data['results'][0]['deck']
    except:
        desc = 'N/A'
    return desc


df['description'] = df['Name'].apply(get_description)

df.to_csv('Video Games Dataset.csv', index=True)
