import streamlit as st
import pandas as pd
import sys
from src.recommender import MusicRecommender

# Section 1 — Setup and Loading
st.set_page_config(
    page_title="Vibe Recommender",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource  # Cache so the model only loads once
def load_recommender():
    return MusicRecommender()

rec = load_recommender()

# Section 2 — Sidebar Controls
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

# Section 3 — Main Query Area
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

# Section 4 — Results Display
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
            with st.expander(f"**{row['track_name']}** - {row['artists']}"):
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
                            st.session_state[f"feedback_{row['track_name']}"] = 1
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Section 5 — Feedback Summary (Optional)
if 'feedback' in str(st.session_state):
    with st.sidebar:
        st.markdown("---")
        st.subheader("Your Feedback")
        for k, v in st.session_state.items():
            if k.startswith("feedback_"):
                track = k.replace("feedback_", "")
                emoji = "👍" if v == 1 else "👎"
                st.write(f"{emoji} {track}")