import numpy as np
import faiss
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

KEYWORD_BOOSTS = {
    # Language / culture
    "chinese":  ["mandopop", "c-pop", "chinese", "mandarin", "cantopop"],
    "japanese": ["j-pop", "j-rock", "anime", "japanese", "jpop"],
    "korean":   ["k-pop", "korean", "kpop"],
    "thai":     ["thai pop", "thai", "t-pop"],
    "spanish":  ["latin", "reggaeton", "spanish", "flamenco"],

    # Subculture
    "vtuber":   ["vtuber", "hololive", "virtual youtuber", "anime", "j-pop"],
    "anime":    ["anime", "j-pop", "japanese", "vtuber"],
    "kpop":     ["k-pop", "korean", "kpop"],

    # Era
    "80s":      ["80s", "synthpop", "new wave", "classic rock"],
    "90s":      ["90s", "grunge", "britpop", "rnb"],
    "2000s":    ["2000s", "pop punk", "indie rock"],

    # Mood shortcuts
    "lofi":     ["lo-fi", "lofi", "chillhop", "study"],
    "sleep":    ["ambient", "sleep", "relaxing", "meditation"],
    "workout":  ["workout", "gym", "high energy", "hype"],
}

def get_tag_boost(query: str) -> list:
    """Return a list of tags to boost based on keywords found in the query."""
    query_lower = query.lower()
    boosted_tags = []
    for keyword, tags in KEYWORD_BOOSTS.items():
        if keyword in query_lower:
            boosted_tags.extend(tags)
    return list(set(boosted_tags))

class MusicRecommender:

    # Layer 1 — Initialization
    def __init__(self):
        print("Loading data...")
        self.df = pd.read_csv('data/processed/tracks_enriched_described.csv')

        print("Loading index...")
        self.index = faiss.read_index('index/faiss.index')

        print("Loading metrics...")
        self.hybrid_matrix = np.load('embeddings/hybrid_matrix.npy').astype('float32')
        self.audio_matrix = np.load('embeddings/audio_matrix.npy')

        print("Loading model...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') # 50+ languages, same 384 dimensions

        with open('embeddings/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.ALPHA = 0.6
        self.BETA = 0.4

        print("Ready.")
    
    # Layer 2 — Query Vector Builder
    def _build_query_vector_from_text(self, text: str) -> np.ndarray:
        """Embed a text mood description into a query vector."""
        text_embedding = self.model.encode([text], convert_to_numpy=True)
        text_embedding = normalize(text_embedding, norm='l2')

        # For audio features, use the mean of the entire dataset as the neutral baseline
        audio_neutral = np.zeros((1, self.audio_matrix.shape[1]))
        audio_neutral = normalize(audio_neutral + 1e-8, norm='l2') # Avoid zero vector

        hybrid = np.hstack([
            self.ALPHA * text_embedding,
            self.BETA * audio_neutral
        ])
        return normalize(hybrid.astype('float32'), norm='l2')
    
    def _build_query_vector_from_track(self, track_idx: int) -> np.ndarray:
         """Build a query vector from an existing track in the dataset."""
         return self.hybrid_matrix[track_idx:track_idx+1]
    
    # Layer 3 — Track Lookup
    def find_track(self, name: str, artist: str = None) -> int:
        """Find a track index by name (and optionally artist). Returns -1 if not found."""
        mask = self.df['track_name'].str.lower() == name.lower()
        if artist:
            mask &= self.df['artists'].str.lower().str.contains(artist.lower())

        matches = self.df[mask]
        if len(matches) == 0:
            return -1
        return matches.index[0]
    
    def _mmr_rerank(self, query_vec: np.ndarray, candidate_indices: np.ndarray, lambda_param: float = 0.7, k: int = 20) -> list:
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
                best = max(remaining, key=lambda i: query_sims[i]) # For each candidate, compute its similarity to the query, then pick the highest
            else:
                # Subsequent picks: balance relevance and diversity
                selected_vecs = candidates[selected]

                def mmr_score(i):
                    relevance = query_sims[i]
                    # Max similarity to any already-selected item
                    redundancy = max((candidates[i] @ selected_vecs.T).flatten()) # .T transpose to align matrix dimensions for batch similarity computation
                    return lambda_param * relevance - (1 - lambda_param) * redundancy
                
                best = max(remaining, key=mmr_score) # Don’t compare the numbers directly, compute a score first, then compare that

            selected.append(best)
            remaining.remove(best)

        return[candidate_indices[i] for i in selected]
    
    # Layer 5 — Main Recommend Function
    def recommend(self,
                  query: str = None,
                  track_name: str = None,
                  artist: str = None,
                  n: int =28,
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
        boosted_tags = get_tag_boost(query or "")

        # Apply boosted_tags:
        if boosted_tags:
            tag_mask = self.df['lastfm_tags'].fillna('').apply(
                lambda t: any(tag in t for tag in boosted_tags)
            )
            boosted_candidates = self.df.index[tag_mask].tolist()

            # Boosted candidates go first, then FAISS candidates, deduplicated
            merged = list(dict.fromkeys(boosted_candidates[:200] + list(candidates)))
            candidates = np.array(merged[:k_candidates])

        # Apply genre filter if specified
        if genre_filter:
            genre_mask = self.df.iloc[candidates]['track_genre'].str.lower().str.contains(genre_filter.lower(), na=False)
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
