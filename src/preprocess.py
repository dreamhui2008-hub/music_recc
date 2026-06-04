import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

AUDIO_FEATURES = [
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness_log', 'speechiness_log', 'loudness',
    'tempo', 'liveness', 'mode'
]

def build_track_description(row):
    parts = [
        row['track_name'],
        f"by {row['artists']}",
        f"genre: {row['track_genre']}",
    ]

    # Add Last.fm tags if present
    if pd.notna(row.get('lastfm_tags')) and row['lastfm_tags']:
        parts.append(f"tags: {row['lastfm_tags']}")

    parts += [
        f"valence {row['valence']:.2f}",
        f"energy {row['energy']:.2f}",
        f"danceability {row['danceability']:.2f}",
        f"tempo {row['tempo']:.0f} BPM",
        "major key" if row['mode'] == 1 else "minor key"
    ]
    return ' '.join(parts)

if __name__ =="__main__":
    df = pd.read_csv('data/processed/tracks_enriched.csv', encoding='cp1252') # This CSV was already opened so using encoding to align with Windows standard

    # Re-apply log transforms
    df['instrumentalness_log'] = np.log1p(df['instrumentalness'])
    df['speechiness_log'] = np.log1p(df['speechiness'])

    # Re-fit and save scaler
    scaler = StandardScaler()
    audio_matrix = scaler.fit_transform(df[AUDIO_FEATURES])
    np.save('embeddings/audio_matrix.npy', audio_matrix)

    with open('embeddings/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Rebuild text descriptions
    df['text_description'] = df.apply(build_track_description, axis=1)
    df = df.reset_index(drop=True)
    df.to_csv('data/processed/tracks_enriched_described.csv', index=False)

    print(f"Done. {len(df)} tracks. Sample description:")
    print(df['text_description'].iloc[0])