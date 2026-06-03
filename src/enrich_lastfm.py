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

if __name__ == "__main__":
    df = pd.read_csv('data/processed/tracks_clean.csv')

    # Optional: start with top 100k by popularity to get results faster
    # df = df.nlargest(100000, 'popularity').reset_index(drop=True)

    all_tags = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        tags = get_lastfm_tags(row['track_name'], row['artists'], API_KEY)
        all_tags.append(''.join(tags))
        time.sleep(0.2) # 5 req/sec to stay within Last.fm rate limit

    # Checkpoint every 10k rows so a crash doesn't lose everything
    if i % 10000 == 0 and i > 0:
        df_checkpoint = df.copy()
        df_checkpoint['lastfm_tags'] = all_tags + [''] * (len(df) - len(all_tags))
        df_checkpoint.to_csv('data/processed/tracks_enriched_checkpoint.csv', index=False)
        print(f"Checkpoint saved at row {i}")

    df['lastfm_tags'] = all_tags
    df.to_csv('data/processed/tracks_enriched.csv', index=False)
    print(f"Done. {len(df)} tracks enriched with Last.fm tags.")

    '''
        Time estimate: 600k tracks × 0.2s = ~33 hours.
        Run it before you sleep.
        If it crashes,
            1) rename tracks_enriched_checkpoint.csv to tracks_enriched.csv and, 
            2) rewrite the script to skip rows where lastfm_tags is already populated, 
            3) add if pd.notna(row.get('lastfm_tags')) and row['lastfm_tags']: continue inside the loop.
    '''