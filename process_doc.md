# Vibe-Based Music Recommender — Complete Process Document

> **Purpose:** A self-contained, step-by-step guide for building a vibe-based music recommendation engine from scratch. Designed to be followed independently, without AI assistance, as a learning project covering data engineering, ML, NLP, vector search, and UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1 — Data Acquisition & Exploration](#3-phase-1--data-acquisition--exploration)
4. [Phase 2 — Feature Engineering](#4-phase-2--feature-engineering)
5. [Phase 3 — Text Embeddings & NLP Layer](#5-phase-3--text-embeddings--nlp-layer)
6. [Phase 4 — Building the FAISS Index](#6-phase-4--building-the-faiss-index)
7. [Phase 5 — The Recommendation Engine](#7-phase-5--the-recommendation-engine)
8. [Phase 6 — Streamlit UI](#8-phase-6--streamlit-ui)
9. [Phase 7 — Spotify API Integration (Stretch)](#9-phase-7--spotify-api-integration-stretch)
10. [Phase 8 — Evaluation & Iteration](#10-phase-8--evaluation--iteration)
11. [Reference: Key Concepts Glossary](#11-reference-key-concepts-glossary)
12. [Reference: Common Errors & Fixes](#12-reference-common-errors--fixes)

---

## 1. Project Overview

### What You Are Building

A music recommendation engine where users can:
- Type a **natural language mood** ("something chill and melancholic for a rainy night") and get song recommendations.
- Enter a **song name** they like and find similar tracks.
- Tune results by energy, valence, tempo, and genre.
- Give thumbs up/down feedback that influences future results.

### How It Works (Bird's Eye View)

```
User Input (text or song name)
        ↓
  Sentence Transformer embeds the text into a vector
        ↓
  Audio features (valence, energy, tempo, etc.) are normalized
        ↓
  Hybrid vector = NLP embedding + audio features concatenated
        ↓
  FAISS index does approximate nearest-neighbor search
        ↓
  MMR reranker adds diversity to avoid repetitive results
        ↓
  Top N recommendations displayed in Streamlit UI
```

### Skills You Will Learn

- Exploratory data analysis with pandas and Plotly
- Feature normalization and engineering
- Sentence Transformers and text embeddings
- FAISS — approximate nearest neighbor (ANN) search
- Maximal Marginal Relevance (MMR) for diverse ranking
- Building interactive UIs with Streamlit
- (Stretch) REST API calls with the Spotipy library

### Final File Structure

```
music-recommender/
├── data/
│   ├── raw/                    # Original Kaggle download
│   └── processed/              # Cleaned, normalized CSVs
├── embeddings/
│   ├── audio_matrix.npy        # Normalized audio features
│   ├── text_matrix.npy         # Sentence-transformer embeddings
│   └── hybrid_matrix.npy       # Concatenated final vectors
├── index/
│   └── faiss.index             # Saved FAISS index
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_embeddings.ipynb
│   └── 04_index_build.ipynb
├── src/
│   ├── preprocess.py
│   ├── embed.py
│   ├── index_builder.py
│   ├── recommender.py
│   └── utils.py
├── app.py                      # Streamlit entry point
├── requirements.txt
└── README.md
```

---

## 2. Environment Setup

### 2.1 Prerequisites

- Python 3.10 or higher
- `pip` and `venv`
- A Kaggle account (free) for dataset download
- Git (optional but recommended)

### 2.2 Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows
```

Always activate this environment before working on the project.

### 2.3 Install Dependencies

Create a `requirements.txt` with the following content:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
plotly==5.21.0
sentence-transformers==3.0.1
faiss-cpu==1.8.0
streamlit==1.35.0
spotipy==2.23.0
kaggle==1.6.12
jupyter==1.0.0
tqdm==4.66.4
seaborn>=0.13.2
```

Then install:

```bash
pip install -r requirements.txt
```

> **Note on faiss-cpu vs faiss-gpu:** Use `faiss-cpu` unless you have a CUDA-capable GPU and the corresponding CUDA toolkit installed. For this project, CPU is perfectly fine.

### 2.4 Set Up Kaggle API

1. Go to https://www.kaggle.com → Account → API → Create New Token
2. Save the API via CLI with
        mkdir -p ~/.kaggle
        echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
3. Run `type $env:USERPROFILE\.kaggle\kaggle.json` to confirm it has been created under your local Kaggle folder
3. Make sure that the encoding is UTF-8, if it is not, do the following:
        `notepad $env:USERPROFILE\.kaggle\kaggle.json` on CLI
        In Notepad, click File → Save As
        At the bottom, change Encoding to UTF-8
        Save it as `kaggle.json`
        Replace the existing file

---

## 3. Phase 1 — Data Acquisition & Exploration

**Goal:** Download the dataset, understand its structure, clean it, and produce a solid mental model of what you're working with.

### 3.1 Download the Dataset

The primary dataset is the Spotify Tracks Dataset on Kaggle (~600k tracks):

```bash
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset
python -m zipfile -e .\-spotify-tracks-dataset.zip .\data\raw
```

Dataset page: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

You can supplement later with:
- `rodolfofigueroa/spotify-12m-songs` (~1.2M tracks, larger)
- `yamaerenay/spotify-dataset-19212020-600k-tracks` (alternative)

### 3.2 Open `notebooks/01_eda.ipynb`

Work through these steps in order:

#### Step 1 — Load and Inspect

```python
import pandas as pd

df = pd.read_csv('../data/raw/dataset.csv')
print(df.shape)
print(df.dtypes)
print(df.head())
```

**What to look for:**
- How many rows and columns?
- What are the column names?
- What dtype is each column?

#### Step 2 — Understand the Audio Features

These are the core Spotify audio features you will use. Read each description carefully:

| Feature | Range | What It Means |
|---|---|---|
| `valence` | 0.0–1.0 | Musical positiveness. High = happy/euphoric, Low = sad/angry |
| `energy` | 0.0–1.0 | Intensity and activity. High = fast/loud, Low = calm/quiet |
| `danceability` | 0.0–1.0 | How suitable for dancing based on tempo, rhythm stability, beat strength |
| `acousticness` | 0.0–1.0 | Confidence the track is acoustic (not electronic) |
| `instrumentalness` | 0.0–1.0 | Predicts whether a track has no vocals. >0.5 = likely instrumental |
| `liveness` | 0.0–1.0 | Probability of being a live recording |
| `speechiness` | 0.0–1.0 | Presence of spoken words. >0.66 = likely podcast/speech |
| `loudness` | dB (typically –60 to 0) | Overall loudness in decibels |
| `tempo` | BPM | Estimated tempo in beats per minute |
| `duration_ms` | milliseconds | Track length |
| `key` | 0–11 | Musical key (0 = C, 1 = C♯/D♭, etc.) |
| `mode` | 0 or 1 | Major (1) or minor (0) |
| `time_signature` | 3–7 | Estimated time signature (beats per measure) |

#### Step 3 — Check for Missing Values

```python
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # As percentages
```

**What to do:**
- Columns with <1% missing: drop those rows
- Columns with >10% missing: decide if the column is worth keeping
- Never impute audio features — a song with missing valence should be dropped, not guessed

#### Step 4 — Check Duplicates

```python
# Duplicate track IDs
print(df['track_id'].duplicated().sum())

# Duplicate name + artist combos
print(df.duplicated(subset=['track_name', 'artists']).sum())
```

Drop duplicates by `track_id` — that is the canonical identifier.

#### Step 5 — Distribution Analysis

Plot histograms for every audio feature:

```python
import matplotlib.pyplot as plt

audio_features = ['valence', 'energy', 'danceability', 'acousticness',
                  'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness']

df[audio_features].hist(bins=50, figsize=(15, 10))
plt.tight_layout()
plt.savefig('../data/processed/feature_distributions.png')
plt.show()
```

**What to look for:**
- Is `instrumentalness` heavily skewed toward 0? (It will be — most songs have vocals)
- Is `speechiness` also skewed? (Yes — very few tracks are pure speech)
- Is `loudness` normally distributed or skewed?
- Does `tempo` have a bimodal distribution? (Often yes — slow vs. fast music)

These insights will affect your normalization strategy in Phase 2.

#### Step 6 — Correlation Matrix

```python
import seaborn as sns

corr = df[audio_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Audio Feature Correlations')
plt.savefig('../data/processed/correlation_matrix.png')
plt.show()
```

**What to look for:**
- Are `energy` and `loudness` highly correlated? (They usually are, ~0.7+)
- Is `acousticness` negatively correlated with `energy`? (Yes, typically)
- Highly correlated features carry redundant information — note which ones for Phase 2

#### Step 7 — Genre Exploration

```python
print(df['track_genre'].value_counts().head(30))
print(f"Total unique genres: {df['track_genre'].nunique()}")
```

Map genres mentally to audio feature profiles. Example: pop tracks likely cluster high valence + high energy; ambient likely clusters low energy + high acousticness + high instrumentalness.

#### Step 8 — Save the Cleaned Dataset

```python
# Drop rows with any null in key feature columns
key_cols = audio_features + ['track_name', 'artists', 'track_genre', 'track_id']
df_clean = df.dropna(subset=key_cols)
df_clean = df_clean.drop_duplicates(subset=['track_id'])
df_clean = df_clean.reset_index(drop=True)

df_clean.to_csv('../data/processed/tracks_clean.csv', index=False)
print(f"Clean dataset: {len(df_clean)} tracks")
```

### 3.3 EDA Checklist Before Moving On

- [x] You know the shape of the dataset (rows, columns)
- [x] You understand what each audio feature means
- [x] Missing values handled
- [x] Duplicates removed
- [x] Distribution plots generated and studied
- [x] Correlation matrix analyzed
- [x] Clean CSV saved to `data/processed/`

---

## 4. Phase 2 — Feature Engineering

**Goal:** Transform raw audio features into a clean, normalized numeric matrix suitable for vector similarity search.

### 4.1 Why Normalization Matters

Imagine `tempo` ranges from 60 to 200 BPM and `valence` ranges from 0 to 1. Without normalization, `tempo` would dominate any distance calculation simply because its numbers are larger. You need all features on the same scale.

### 4.2 Open `notebooks/02_feature_engineering.ipynb`

#### Step 1 — Decide Which Features to Keep

Start with this core set:

```python
AUDIO_FEATURES = [
    'valence',
    'energy',
    'danceability',
    'acousticness',
    'instrumentalness',
    'speechiness',
    'loudness',
    'tempo',
    'liveness',
    'mode'          # binary, but still informative
]
```

**Features to exclude (for now):**
- `key` — ordinal encoding of a circular quantity (0–11) is misleading for distance; skip unless you later use circular encoding
- `time_signature` — rarely meaningful for mood
- `duration_ms` — not a mood characteristic
- `popularity` — could bias results, better used as a re-ranking signal later

#### Step 2 — Handle Skewed Features with Log Transform

For `instrumentalness` and `speechiness` (heavily right-skewed):

```python
import numpy as np
import pandas as pd

df = pd.read_csv('../data/processed/tracks_clean.csv')

# Log1p transform (log(1+x)) — safe for values that include 0
df['instrumentalness_log'] = np.log1p(df['instrumentalness'])
df['speechiness_log'] = np.log1p(df['speechiness'])
```

**Why log1p?** A value of 0 would cause `log(0) = -inf`. Using `log(1+x)` maps 0 → 0 safely.

Plot the before/after:
```python
fig, axes = plt.subplots(1,2, figsize=(12, 4))
df['instrumentalness'].hist(bins=50, ax=axes[0])
axes[0].set_title('Before')

df['instrumentalness_log'].hist(bins=50, ax=axes[1])
axes[1].set_title('After log1p')
plt.show()
```

#### Step 3 — Normalize Features

Use `sklearn`'s `StandardScaler` (z-score normalization) for continuous features:

```python
from sklearn.preprocessing import StandardScaler

FEATURES_TO_SCALE = [
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness_log', 'speechiness_log', 'loudness',
    'tempo', 'liveness', 'mode'
]

scaler = StandardScaler()
audio_matrix = scaler.fit_transform(df[FEATURES_TO_SCALE])

print(f"Audio matrix shape: {audio_matrix.shape}")
print(f"Mean (should be ~0): {audio_matrix.mean(axis=0).round(3)}")
print(f"Std (should be ~1): {audio_matrix.std(axis=0).round(3)}")
```

**Save the scaler** — you will need it later to normalize user queries:

```python
import pickle
with open('../embeddings/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

#### Step 4 — Encode Genre

You will use genre information as an additional signal. One-hot encode it:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['track_genre'])

# Save label encoder
with open('../embeddings/genre_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"Number of genres: {len(le.classes_)}")
print(le.classes_[:20])
```

> **Note:** You will decide in Phase 4 whether to include genre as part of the FAISS vector or use it as a filter. For now, just encode it and save.

#### Step 5 — Save the Audio Matrix

```python
import numpy as np
np.save('../embeddings/audio_matrix.npy', audio_matrix)
df.to_csv('../data/processed/tracks_engineered.csv', index=False)

print(f"Saved audio_matrix.npy: shape {audio_matrix.shape}")
```

#### Step 6 — Visualize Feature Space with PCA

This is optional but illuminating:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords_2d = pca.fit_transform(audio_matrix)

import plotly.express as px
sample = df.sample(5000, random_state=42).copy()
sample_coords = pca.transform(audio_matrix[sample.index])

fig = px.scatter(
    x=sample_coords[:, 0],
    y=sample_coords[:, 1],
    color=sample['track_genre'],
    hover_name=sample['track_name'],
    title='Track Feature Space (PCA 2D)',
    opacity=0.5
)
fig.show()
```

**What to look for:** Do genres cluster? Does pop occupy a different region than classical? Visible clustering suggests that the engineered audio features capture meaningful musical structure and may be useful for downstream tasks such as genre classification or recommendation.

### 4.3 Phase 2 Checklist

- [x] Skewed features log-transformed
- [x] All features z-score normalized
- [x] Scaler saved as `scaler.pkl`
- [x] Genre label-encoded and encoder saved
- [x] `audio_matrix.npy` saved with shape `(N, 10)`
- [x] PCA plot confirms genre clustering

---

## 5. Phase 3 — Text Embeddings & NLP Layer

**Goal:** Convert text (mood descriptions or song titles) into dense vectors so we can compare them against track metadata.

### 5.1 What Are Sentence Embeddings?

A sentence transformer model converts any text string into a fixed-size vector (e.g., 384 dimensions) such that semantically similar texts are close together in vector space. "Chill rainy day vibes" and "melancholic and calm" will produce vectors that are close together, even though they share no words.

This is the core of the "vibe-based" input mode.

### 5.2 Open `notebooks/03_embeddings.ipynb`

#### Step 1 — Load the Model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# This downloads ~80MB on first run and caches locally
```

`all-MiniLM-L6-v2` is a good default: fast, small, and produces 384-dimensional embeddings. You can experiment with larger models later (e.g., `all-mpnet-base-v2` for 768 dimensions).

#### Step 2 — Create Text Descriptions Per Track

For each track, build a text string combining metadata:

```python
import pandas as pd

df = pd.read_csv('../data/processed/tracks_engineered.csv')

def build_track_description(row):
    parts = [
        row['track_name'],
        f"by {row['artists']}",
        f"genre: {row['track_genre']}",
        f"valence {row['valence']:.2f}",
        f"energy {row['energy']:.2f}",
        f"danceability {row['danceability']:.2f}",
        f"tempo {row['tempo']:.0f} BPM",
        "major key" if row['mode'] == 1 else "minor key"
    ]
    return ' '.join(parts)

df['text_description'] = df.apply(build_track_description, axis=1)
print(df['text_description'].iloc[0])
```

**Why include audio feature values in the text?** Because when the user types "high energy" you want the model to associate that with tracks that have high energy values, even if those tracks have no word "energy" in their title. The combined description creates a bridge.

#### Step 3 — Encode All Tracks (Batch Processing)

Encoding ~600k tracks takes several minutes. Use batching and `tqdm` for progress:

```python
from tqdm import tqdm
import numpy as np

descriptions = df['text_description'].tolist()
batch_size = 512
all_embeddings = []

for i in tqdm(range(0, len(descriptions), batch_size)):
    batch = descriptions[i:i + batch_size]
    embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(embeddings)

text_matrix = np.vstack(all_embeddings)
print(f"Text matrix shape: {text_matrix.shape}")  # Should be (N, 384)
```

> **Warning:** This step may take 20–60 minutes depending on your hardware. Start it, grab a coffee, and let it run. If you have a GPU, `sentence-transformers` will use it automatically.

#### Step 4 — L2-Normalize the Text Embeddings

For cosine similarity search, normalize vectors to unit length:

```python
from sklearn.preprocessing import normalize

text_matrix_normalized = normalize(text_matrix, norm='l2')
np.save('../embeddings/text_matrix.npy', text_matrix_normalized)
print("Text matrix saved.")
```

#### Step 5 — Build the Hybrid Matrix

Combine audio features and text embeddings into a single vector per track. You need to weight them appropriately:

```python
audio_matrix = np.load('../embeddings/audio_matrix.npy')

# Normalize audio matrix to unit length as well (so scales match)
audio_matrix_normalized = normalize(audio_matrix, norm='l2')

# Weight: 60% text embedding, 40% audio features
# Adjust these weights later during evaluation
ALPHA = 0.6  # text weight
BETA = 0.4   # audio weight

hybrid_matrix = np.hstack([
    ALPHA * text_matrix_normalized,
    BETA * audio_matrix_normalized
])

hybrid_matrix = normalize(hybrid_matrix, norm='l2')  # Normalize the full hybrid too

np.save('../embeddings/hybrid_matrix.npy', hybrid_matrix)
print(f"Hybrid matrix shape: {hybrid_matrix.shape}")  # Should be (N, 394)
```

**Understanding the alpha/beta weights:**
- Higher `ALPHA` → text similarity matters more (mood descriptions will dominate)
- Higher `BETA` → audio features matter more (energy/valence similarity will dominate)
- You will tune these in Phase 8

### 5.3 Phase 3 Checklist

- [x] `all-MiniLM-L6-v2` loaded and tested
- [x] Text descriptions built per track
- [x] All tracks encoded (this takes time — be patient)
- [x] `text_matrix.npy` saved with shape `(N, 384)`
- [x] `hybrid_matrix.npy` saved with shape `(N, 394)`
- [x] Both matrices are L2-normalized

---

## 6. Phase 4 — Building the FAISS Index

**Goal:** Build a fast approximate nearest neighbor (ANN) index over the hybrid matrix, enabling sub-second similarity search across 600k tracks.

### 6.1 What Is FAISS?

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vectors. Instead of comparing your query against every single track (brute-force, O(N)), FAISS uses indexing structures that make this much faster, typically returning results in milliseconds even for millions of vectors.

### 6.2 Open `notebooks/04_index_build.ipynb`

#### Step 1 — Load the Hybrid Matrix

```python
import numpy as np
import faiss

hybrid_matrix = np.load('../embeddings/hybrid_matrix.npy').astype('float32')
# FAISS requires float32; embeddings are often float64 by default
print(f"Matrix shape: {hybrid_matrix.shape}")
print(f"dtype: {hybrid_matrix.dtype}")
```

#### Step 2 — Choose the Right Index Type

For this project, you have three reasonable choices:

| Index | How It Works | When to Use |
|---|---|---|
| `IndexFlatL2` | Brute force, exact | Small datasets (<100k), debugging |
| `IndexFlatIP` | Brute force inner product (cosine if normalized) | Same as above, L2-normalized vectors |
| `IndexIVFFlat` | Clusters vectors into buckets, searches only nearest buckets | Large datasets, good speed/accuracy tradeoff |
| `IndexHNSWFlat` | Graph-based, very fast query, no training needed | Production-grade, best for your use case |

**Use `IndexHNSWFlat` for this project.** It requires no training and has excellent query speed with high recall:

```python
d = hybrid_matrix.shape[1]   # Dimension of vectors (394)
M = 32                         # Number of neighbors to connect in the graph
# Higher M = better recall, more memory. 32 is a good default.

index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200  # Quality of index construction. Higher = better but slower build.
index.hnsw.efSearch = 100        # Quality of search. Higher = better recall but slower query.

# Add all vectors
index.add(hybrid_matrix)
print(f"Index trained: {index.is_trained}")
print(f"Total vectors in index: {index.ntotal}")
```

#### Step 3 — Test the Index

Before saving, verify it works:

```python
import pandas as pd

df = pd.read_csv('../data/processed/tracks_engineered.csv')

# Pick a track index to test
test_idx = 0
test_track = df.iloc[test_idx]
print(f"Query track: {test_track['track_name']} by {test_track['artists']}")

# Query the index
query_vector = hybrid_matrix[test_idx:test_idx+1]  # Shape (1, 394)
k = 10  # Number of results

distances, indices = index.search(query_vector, k + 1)  # +1 because query itself will appear

# Skip the query track itself (index 0)
results = indices[0][1:]
for rank, idx in enumerate(results):
    track = df.iloc[idx]
    print(f"  {rank+1}. {track['track_name']} by {track['artists']} [{track['track_genre']}]")
```

Are the results reasonable? Similar genre? Similar energy/valence?

#### Step 4 — Save the Index

```python
faiss.write_index(index, '../index/faiss.index')
print("FAISS index saved.")
```

#### Step 5 — Benchmark Query Speed

```python
import time

N_QUERIES = 100
random_indices = np.random.choice(len(hybrid_matrix), N_QUERIES)
query_vectors = hybrid_matrix[random_indices].astype('float32')

start = time.time()
distances, indices = index.search(query_vectors, 10)
end = time.time()

print(f"Average query time: {(end - start) / N_QUERIES * 1000:.2f} ms")
```

Target: <10ms per query. If it's slower, reduce `efSearch`.

### 6.3 Phase 4 Checklist

- [x] `hybrid_matrix.npy` loaded as float32
- [x] `IndexHNSWFlat` built and populated
- [x] Spot-check query returns sensible results
- [x] Query speed benchmarked and within target
- [x] Index saved to `index/faiss.index`

---

## 7. Phase 5 — The Recommendation Engine

**Goal:** Wrap the FAISS index in a Python class that handles both text queries and song-name queries, plus diversity reranking.

### 7.1 Create `src/recommender.py`

This is the core module. Build it in layers:

#### Layer 1 — Initialization

```python
import numpy as np
import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

class MusicRecommender:
    def __init__(self):
        print("Loading data...")
        self.df = pd.read_csv('data/processed/tracks_engineered.csv')
        
        print("Loading index...")
        self.index = faiss.read_index('index/faiss.index')
        
        print("Loading matrices...")
        self.hybrid_matrix = np.load('embeddings/hybrid_matrix.npy').astype('float32')
        self.audio_matrix = np.load('embeddings/audio_matrix.npy')
        
        print("Loading model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        with open('embeddings/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.ALPHA = 0.6
        self.BETA = 0.4
        
        print("Ready.")
```

#### Layer 2 — Query Vector Builder

```python
    def _build_query_vector_from_text(self, text: str) -> np.ndarray:
        """Embed a text mood description into a query vector."""
        text_embedding = self.model.encode([text], convert_to_numpy=True)
        text_embedding = normalize(text_embedding, norm='l2')
        
        # For audio features, use the mean of the entire dataset as the neutral baseline
        audio_neutral = np.zeros((1, self.audio_matrix.shape[1]))
        audio_neutral = normalize(audio_neutral + 1e-8, norm='l2')  # Avoid zero vector
        
        hybrid = np.hstack([
            self.ALPHA * text_embedding,
            self.BETA * audio_neutral
        ])
        return normalize(hybrid.astype('float32'), norm='l2')
    
    def _build_query_vector_from_track(self, track_idx: int) -> np.ndarray:
        """Build a query vector from an existing track in the dataset."""
        return self.hybrid_matrix[track_idx:track_idx+1]
```

#### Layer 3 — Track Lookup

```python
    def find_track(self, name: str, artist: str = None) -> int:
        """Find a track index by name (and optionally artist). Returns -1 if not found."""
        mask = self.df['track_name'].str.lower() == name.lower()
        if artist:
            mask &= self.df['artists'].str.lower().str.contains(artist.lower())
        
        matches = self.df[mask]
        if len(matches) == 0:
            return -1
        return matches.index[0]
```

#### Layer 4 — Maximal Marginal Relevance (MMR)

MMR is a reranking algorithm that balances relevance and diversity. Without it, results can be repetitive (e.g., 10 near-identical remixes of the same song).

```python
    def _mmr_rerank(self, query_vec: np.ndarray, candidate_indices: np.ndarray,
                    lambda_param: float = 0.7, k: int = 20) -> list:
        """
        Maximal Marginal Relevance reranking.
        
        lambda_param: 1.0 = pure relevance, 0.0 = pure diversity. 0.7 is a good default.
        k: number of final results to return.
        """
        candidates = self.hybrid_matrix[candidate_indices]
        
        # Cosine similarities between query and candidates
        query_sims = (candidates @ query_vec.T).flatten()
        
        selected = []
        remaining = list(range(len(candidates)))
        
        while len(selected) < k and remaining:
            if not selected:
                # First pick: highest relevance
                best = max(remaining, key=lambda i: query_sims[i])
            else:
                # Subsequent picks: balance relevance and diversity
                selected_vecs = candidates[selected]
                
                def mmr_score(i):
                    relevance = query_sims[i]
                    # Max similarity to any already-selected item
                    redundancy = max((candidates[i] @ selected_vecs.T).flatten())
                    return lambda_param * relevance - (1 - lambda_param) * redundancy
                
                best = max(remaining, key=mmr_score)
            
            selected.append(best)
            remaining.remove(best)
        
        return [candidate_indices[i] for i in selected]
```

#### Layer 5 — Main Recommend Function

```python
    def recommend(self,
                  query: str = None,
                  track_name: str = None,
                  artist: str = None,
                  n: int = 20,
                  diversity: float = 0.7,
                  genre_filter: str = None) -> pd.DataFrame:
        """
        Main recommendation function.
        
        Args:
            query: Natural language mood description
            track_name: Name of a song to base recommendations on
            artist: Optional artist name to help find the right track
            n: Number of results to return
            diversity: Lambda for MMR (0=pure diversity, 1=pure relevance)
            genre_filter: Optional genre to restrict results to
        
        Returns:
            DataFrame with recommended tracks
        """
        # Build the query vector
        if track_name:
            idx = self.find_track(track_name, artist)
            if idx == -1:
                raise ValueError(f"Track '{track_name}' not found in dataset.")
            query_vec = self._build_query_vector_from_track(idx)
        elif query:
            query_vec = self._build_query_vector_from_text(query)
        else:
            raise ValueError("Provide either 'query' or 'track_name'.")
        
        # Fetch more candidates than needed for MMR to work with
        k_candidates = n * 5
        distances, indices = self.index.search(query_vec, k_candidates)
        candidates = indices[0]
        
        # Apply genre filter if specified
        if genre_filter:
            genre_mask = self.df.iloc[candidates]['track_genre'].str.lower() == genre_filter.lower()
            candidates = candidates[genre_mask.values]
        
        # MMR reranking
        final_indices = self._mmr_rerank(query_vec, candidates, lambda_param=diversity, k=n)
        
        # Build result dataframe
        results = self.df.iloc[final_indices][
            ['track_name', 'artists', 'track_genre', 'valence', 'energy',
             'danceability', 'tempo', 'acousticness', 'instrumentalness', 'popularity']
        ].copy()
        results['rank'] = range(1, len(results) + 1)
        
        return results
```

#### Layer 6 — Test It

Create `src/test_recommender.py`:

```python
from recommender import MusicRecommender
import os
os.chdir('..')

rec = MusicRecommender()

# Test 1: Text mood query
print("=== Mood Query: 'sad and rainy' ===")
results = rec.recommend(query="sad and rainy", n=10)
print(results[['track_name', 'artists', 'track_genre', 'valence', 'energy']].to_string())

# Test 2: Song-based query
print("\n=== Song-Based: Bohemian Rhapsody ===")
results = rec.recommend(track_name="Bohemian Rhapsody", n=10)
print(results[['track_name', 'artists', 'track_genre']].to_string())

# Test 3: Filtered by genre
print("\n=== Mood + Genre Filter: upbeat, pop only ===")
results = rec.recommend(query="upbeat and happy", genre_filter="pop", n=10)
print(results[['track_name', 'artists', 'valence', 'energy']].to_string())
```

### 7.2 Phase 5 Checklist

- [ ] `MusicRecommender` class built with all five layers
- [ ] Text query returns sensible results
- [ ] Track-name query returns similar songs
- [ ] Genre filter works
- [ ] MMR reduces obvious duplicates in results
- [ ] All three test cases pass manually

---

## 8. Phase 6 — Streamlit UI

**Goal:** Build an interactive web UI that lets users query the recommender, adjust parameters, and give feedback.

### 8.1 What Is Streamlit?

Streamlit is a Python library that turns scripts into web apps. You write Python; it renders HTML/CSS/JS automatically. No frontend experience required.

Run any Streamlit app with:
```bash
streamlit run app.py
```

### 8.2 Build `app.py`

#### Section 1 — Setup and Loading

```python
import streamlit as st
import pandas as pd
import sys
sys.path.append('src')
from recommender import MusicRecommender

st.set_page_config(
    page_title="Vibe Recommender",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource  # Cache so the model only loads once
def load_recommender():
    return MusicRecommender()

rec = load_recommender()
```

> **`@st.cache_resource` is critical.** Without it, the model reloads on every user interaction, making the app unusably slow.

#### Section 2 — Sidebar Controls

```python
with st.sidebar:
    st.title("🎛️ Controls")
    
    mode = st.radio("Query Mode", ["Mood Description", "Song Name"])
    
    n_results = st.slider("Number of Results", 5, 50, 20)
    diversity = st.slider(
        "Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Higher = more varied results. Lower = more similar to query."
    )
    
    genre_options = ["All Genres"] + sorted(rec.df['track_genre'].unique().tolist())
    selected_genre = st.selectbox("Filter by Genre", genre_options)
    genre_filter = None if selected_genre == "All Genres" else selected_genre
```

#### Section 3 — Main Query Area

```python
st.title("🎵 Vibe-Based Music Recommender")
st.markdown("Describe a mood or enter a song you love, and discover new music.")

if mode == "Mood Description":
    query = st.text_input(
        "Describe your mood or vibe:",
        placeholder="e.g. melancholic and rainy, perfect for 3am introspection"
    )
    track_name = None
    artist = None
else:
    col1, col2 = st.columns(2)
    with col1:
        track_name = st.text_input("Song Name:", placeholder="e.g. Bohemian Rhapsody")
    with col2:
        artist = st.text_input("Artist (optional):", placeholder="e.g. Queen")
    query = None

search_clicked = st.button("Find Music 🔍", type="primary")
```

#### Section 4 — Results Display

```python
if search_clicked:
    try:
        with st.spinner("Finding your vibe..."):
            results = rec.recommend(
                query=query,
                track_name=track_name,
                artist=artist,
                n=n_results,
                diversity=diversity,
                genre_filter=genre_filter
            )
        
        st.success(f"Found {len(results)} tracks!")
        
        for _, row in results.iterrows():
            with st.expander(f"**{row['track_name']}** — {row['artists']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Genre", row['track_genre'])
                    st.metric("Valence", f"{row['valence']:.2f}")
                with col2:
                    st.metric("Energy", f"{row['energy']:.2f}")
                    st.metric("Danceability", f"{row['danceability']:.2f}")
                with col3:
                    st.metric("Tempo", f"{row['tempo']:.0f} BPM")
                    st.metric("Popularity", f"{row['popularity']:.0f}/100")
                
                # Feedback buttons
                col_up, col_down, _ = st.columns([1, 1, 8])
                with col_up:
                    if st.button("👍", key=f"up_{row['track_name']}_{_}"):
                        st.session_state[f"feedback_{row['track_name']}"] = 1
                with col_down:
                    if st.button("👎", key=f"down_{row['track_name']}_{_}"):
                        st.session_state[f"feedback_{row['track_name']}"] = -1
    
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")
```

#### Section 5 — Feedback Summary (Optional)

```python
if 'feedback' in str(st.session_state):
    with st.sidebar:
        st.markdown("---")
        st.subheader("Your Feedback")
        for k, v in st.session_state.items():
            if k.startswith("feedback_"):
                track = k.replace("feedback_", "")
                emoji = "👍" if v == 1 else "👎"
                st.write(f"{emoji} {track}")
```

### 8.3 Running the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 8.4 Phase 6 Checklist

- [ ] App loads without errors
- [ ] Model loads once and is cached
- [ ] Mood mode returns results
- [ ] Song-name mode returns results
- [ ] Genre filter works from UI
- [ ] Diversity slider visibly changes result variety
- [ ] Feedback buttons render (logic can be extended later)

---

## 9. Phase 7 — Spotify API Integration (Stretch)

**Goal:** Extend the app to optionally query Spotify for tracks not in the dataset, and add Spotify preview links to results.

### 9.1 Spotify Developer Setup

1. Go to https://developer.spotify.com/dashboard
2. Create a new app — name it anything
3. Set redirect URI to `http://localhost:8080`
4. Copy your `Client ID` and `Client Secret`

Store them in a `.env` file (never commit this to Git):

```bash
SPOTIFY_CLIENT_ID=your_id_here
SPOTIFY_CLIENT_SECRET=your_secret_here
```

Load them in Python:

```python
import os
from dotenv import load_dotenv
load_dotenv()

client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
```

Install `python-dotenv`: `pip install python-dotenv`

### 9.2 Basic Spotipy Setup

```python
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))
```

### 9.3 Fetch Audio Features for a Track Not in Your Dataset

```python
def get_spotify_track_features(track_name: str, artist: str = None) -> dict:
    """Search Spotify for a track and return its audio features."""
    query = f"track:{track_name}"
    if artist:
        query += f" artist:{artist}"
    
    results = sp.search(q=query, type='track', limit=1)
    tracks = results['tracks']['items']
    
    if not tracks:
        return None
    
    track = tracks[0]
    track_id = track['id']
    features = sp.audio_features([track_id])[0]
    
    return {
        'track_name': track['name'],
        'artists': ', '.join([a['name'] for a in track['artists']]),
        'preview_url': track['preview_url'],
        'spotify_url': track['external_urls']['spotify'],
        'features': features
    }
```

### 9.4 Add Preview Links to UI

In the results section of `app.py`, add:

```python
# After displaying track info
if 'spotify_url' in row and pd.notna(row.get('spotify_url')):
    st.markdown(f"[Open in Spotify]({row['spotify_url']})")
```

To populate `spotify_url` in your dataset, you can run a batch enrichment script overnight using the Spotify API (be mindful of rate limits: ~180 requests/minute).

---

## 10. Phase 8 — Evaluation & Iteration

**Goal:** Measure how good your recommender actually is and improve it systematically.

### 10.1 Qualitative Evaluation First

Before any metrics, do a structured manual review:

Create a set of 20 test queries covering different modes:

| Query | Type | What Good Results Look Like |
|---|---|---|
| "happy summer vibes" | mood | High valence (>0.6), high energy (>0.5), pop/dance genres |
| "sad breakup song at 3am" | mood | Low valence (<0.4), low-medium energy, minor key |
| "Bohemian Rhapsody" | song | Rock, varied structure, high popularity |
| "Clair de Lune" | song | Classical, very low energy, high acousticness |
| "hype gym music" | mood | Very high energy (>0.8), high danceability |

Run each query, inspect results, and note problems.

### 10.2 Quantitative Metrics

#### Metric 1: Average Feature Coherence

For a mood query, compute how consistent the returned tracks are in the target feature dimensions:

```python
def evaluate_coherence(results_df: pd.DataFrame, feature: str) -> float:
    """Standard deviation of a feature across results — lower = more coherent."""
    return results_df[feature].std()

# For "happy summer" query, valence std should be low
results = rec.recommend("happy summer vibes", n=20)
print(f"Valence coherence: {evaluate_coherence(results, 'valence'):.3f}")
```

#### Metric 2: Genre Diversity

```python
def genre_diversity(results_df: pd.DataFrame) -> float:
    """Fraction of unique genres — higher = more diverse."""
    return results_df['track_genre'].nunique() / len(results_df)

print(f"Genre diversity: {genre_diversity(results):.2f}")
```

#### Metric 3: Intra-List Diversity (ILD)

Average pairwise distance between recommended tracks (higher = more diverse):

```python
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

def intra_list_diversity(result_indices: list, matrix: np.ndarray) -> float:
    vecs = matrix[result_indices]
    distances = cosine_distances(vecs)
    # Upper triangle only (no self-distances)
    n = len(vecs)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += distances[i][j]
            count += 1
    return total / count if count > 0 else 0
```

### 10.3 Tuning Parameters

Once you have metrics, tune systematically:

**Alpha/Beta weights (text vs audio balance):**
- Try: (0.8, 0.2), (0.6, 0.4), (0.4, 0.6)
- Measure: do mood queries improve or degrade?

**MMR diversity (lambda):**
- Try: 0.5, 0.7, 0.9
- Measure: ILD score and user preference

**FAISS efSearch:**
- Try: 50, 100, 200
- Measure: recall (how often the true nearest neighbor is returned) vs. query speed

**Log-transform choices:**
- Try applying log1p to `loudness` as well (it has a long left tail)
- Measure: does feature coherence improve for "loud/quiet" queries?

### 10.4 Feedback Loop (Advanced)

Store user feedback and use it to adjust query vectors:

```python
def adjust_query_with_feedback(query_vec: np.ndarray,
                                liked_indices: list,
                                disliked_indices: list,
                                matrix: np.ndarray,
                                alpha: float = 0.1) -> np.ndarray:
    """
    Rocchio algorithm: move query vector toward liked items, away from disliked.
    alpha: learning rate (how much to adjust)
    """
    if liked_indices:
        liked_center = matrix[liked_indices].mean(axis=0)
        query_vec = query_vec + alpha * liked_center
    if disliked_indices:
        disliked_center = matrix[disliked_indices].mean(axis=0)
        query_vec = query_vec - alpha * disliked_center
    
    return normalize(query_vec.reshape(1, -1), norm='l2')[0]
```

This is a classic technique from information retrieval called **Rocchio relevance feedback**. It's simple but effective.

---

## 11. Reference: Key Concepts Glossary

**Approximate Nearest Neighbor (ANN):** Finding the closest vectors to a query without checking every single vector. Trades tiny accuracy loss for massive speed gains.

**Cosine Similarity:** A measure of similarity between two vectors based on the angle between them. Two vectors pointing in the same direction = similarity 1.0; perpendicular = 0.0. Insensitive to magnitude, only direction matters.

**Embeddings:** Dense, fixed-size numerical representations of data (text, audio, images). Similar items have similar embeddings.

**FAISS:** A library by Meta for fast similarity search over billions of vectors. Your core retrieval engine.

**HNSW (Hierarchical Navigable Small World):** A graph-based ANN algorithm. Builds a layered graph where higher layers have long-range connections and lower layers have local connections. Fast queries, no training required.

**L2 Normalization:** Scaling a vector so its length (L2 norm) equals 1. After L2 normalization, inner product equals cosine similarity.

**Maximal Marginal Relevance (MMR):** A reranking technique that iteratively picks the next item to maximize relevance while penalizing similarity to already-selected items. Produces diverse ranked lists.

**Rocchio Algorithm:** A relevance feedback mechanism that adjusts a query vector based on which results the user liked (move toward) and disliked (move away from).

**Sentence Transformer:** A neural model that encodes sentences/paragraphs into fixed-size vectors, optimized so similar sentences have similar vectors. Based on BERT architecture.

**StandardScaler (Z-Score Normalization):** Transforms features so they have mean=0 and std=1. Formula: `z = (x - mean) / std`.

**TF-IDF:** Term Frequency–Inverse Document Frequency. A classic text representation technique. Not used in this project but foundational to understand before sentence transformers.

**Valence:** Spotify's measure of musical positiveness. High valence = happy/euphoric. Low valence = sad/tense/angry.

---

## 12. Reference: Common Errors & Fixes

### `ValueError: Input contains NaN`

**Cause:** Your audio matrix has missing values that survived cleaning.
**Fix:**
```python
print(df[FEATURES_TO_SCALE].isnull().sum())  # Find which columns
df = df.dropna(subset=FEATURES_TO_SCALE)
```

### `RuntimeError: FAISS error: d != self.d`

**Cause:** Your query vector has a different dimension than the index.
**Fix:** Make sure your query vector is built from the same concatenation as your hybrid matrix. Print and compare `query_vec.shape` and `index.d`.

### Sentence Transformer is Slow

**Cause:** You're encoding one sentence at a time in a loop.
**Fix:** Always encode in batches:
```python
embeddings = model.encode(list_of_texts, batch_size=512)
```

### Streamlit Reloads the Model on Every Click

**Cause:** Missing `@st.cache_resource` decorator.
**Fix:** Add it to your loading function.

### FAISS Returns the Query Track as the Top Result

**Cause:** You're not excluding the query track from results.
**Fix:** Fetch `k+1` results and skip the first one, or filter out `track_id == query_track_id`.

### All Results Are the Same Genre

**Cause:** Genre is too heavily weighted in your text descriptions.
**Fix:** Remove genre from `build_track_description`, or reduce its weight by only including it once.

### `IndexError: index out of bounds`

**Cause:** Your DataFrames and numpy matrices may have misaligned indices after cleaning and resetting.
**Fix:** Always `reset_index(drop=True)` after filtering, and verify `len(df) == len(audio_matrix)`.

---

*End of Process Document. Good luck — this project will teach you more ML than most courses.*