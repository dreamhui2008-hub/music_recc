import requests
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('LASTFM_API_KEY')

def get_lastfm_tags(track_name, artist, api_key):
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getTopTags",
        "track": track_name,
        "artist": artist,
        "api_key": api_key,
        "format": "json"
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        tags = data.get('toptags', {}).get('tag', [])
        return [t['name'].lower() for t in tags[:10]]
    except Exception:
        return []
    
df = pd.read_csv('data/processed/tracks_clean.csv')

all_tags = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    tags = get_lastfm_tags(row['track_name'], row['artists'], API_KEY)
    all_tags.append(''.join(tags))
    time.sleep(0.2) # 5 req/sec

df['lastfm_tags'] = all_tags
df.to_csv('data/processed/tracks_enriched.csv', index=False)