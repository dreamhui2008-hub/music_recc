from recommender import MusicRecommender
import os

#os.chdir('..') Removed from pdxn since want to run directly inside /src

rec = MusicRecommender()

# Test 1: Text mood query
print("=== Mood Query: 'sad and rainy' ===")
results = rec.recommend(query="sad and rainy", n=10)
print(results[['track_name', 'artists', 'track_genre', 'valence', 'energy']].to_string()) # Converts a pandas DataFrame into a nicely formatted text table

# Test 2: Song-based query
print("\n=== Song-Based: Bohemian Rhapsody ===")
results = rec.recommend(track_name="Bohemian Rhapsody", n=18)
print(results[['track_name', 'artists', 'track_genre']].to_string())

# Test 3: Filtered by genre
print("\n=== Mood + Genre Filter: upbeat, pop only ===")
results = rec.recommend(query="upbeat and happy", genre_filter="pop", n=10)
print(results[['track_name', 'artists', 'valence', 'energy']].to_string())
'''
    Notice how we don't have that many pop-genres?
    Because FAISS may retrieve >50 songs that are neighbouring to the query, but not all of them may contain 'pop' in the 'genre' field.
    You can fix this by:
        1) Increasing the pool of candidates with
            k_candidates = n * 20
        2) Soften filter with a preferential fix so that specific genres like pop rank higher to not get passsed by FAISS
            score += genre_bonus
        3) Do a combination of both per above (real-life pdxn fix)
'''
