import os
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import re
from io import BytesIO
from PIL import Image
import time
from hybrid_recommender_validation import (
    load_dataset,
    create_text_features,
    load_or_build_tfidf,
    load_or_build_similarity,
    synthesize_user_data,
    load_or_train_svd,
    HybridRecommender,
)

DATA_PATH = Path("Preprocessed_dataset1.xls")
MODEL_DIR = Path("models")

st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data():
    df = load_dataset(DATA_PATH)
    df = create_text_features(df)
    df['song_key'] = df['song_name'].astype(str) + ' | ' + df['artist'].astype(str)
    return df

@st.cache_data(show_spinner=False)
def build_song_options(df):
    songs = (
        df[['song_name', 'artist', 'popularity']]
        .dropna(subset=['song_name'])
        .drop_duplicates('song_name')
        .sort_values(['popularity', 'song_name'], ascending=[False, True])
    )
    return songs['song_name'].tolist()

@st.cache_resource(show_spinner=False)
def load_models(df):
    vectorizer = load_or_build_tfidf(df, MODEL_DIR)
    similarity_matrix = load_or_build_similarity(df, vectorizer, MODEL_DIR, max_songs=5000)
    df_content = df.head(similarity_matrix.shape[0]).reset_index(drop=True)
    df_cf = synthesize_user_data(df)
    svd_model, training_metrics = load_or_train_svd(df_cf, MODEL_DIR)
    recommender = HybridRecommender(df_content, df_cf, vectorizer, similarity_matrix, svd_model)
    return df_content, df_cf, recommender, training_metrics

@st.cache_data(show_spinner=False)
def build_available_songs(df_content):
    """Build song list from only the songs in df_content (first 5000)"""
    songs = (
        df_content[['song_name', 'artist', 'popularity']]
        .dropna(subset=['song_name'])
        .drop_duplicates('song_name')
        .sort_values(['popularity', 'song_name'], ascending=[False, True])
    )
    return songs['song_name'].tolist()

@st.cache_data(show_spinner=False)
def build_user_options(df_cf):
    user_ids = sorted(df_cf['user_id'].unique())
    return ["New user (cold start)"] + [str(uid) for uid in user_ids]


def parse_user_selection(selection):
    if selection.startswith("New user"):
        return None
    return int(selection)


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def get_spotify_access_token():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    token_url = "https://accounts.spotify.com/api/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'client_credentials'}

    try:
        response = requests.post(token_url, auth=(client_id, client_secret), data=data, headers=headers, timeout=10)
        if response.ok:
            return response.json().get('access_token')
    except Exception:
        return None

    return None


def fetch_spotify_artwork(song_name, artist):
    """Fetch album artwork from Spotify API"""
    token = get_spotify_access_token()
    if not token:
        return None

    query = f"{song_name} {artist}".replace(" ", "%20")
    search_url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"

    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.ok:
            data = response.json()
            tracks = data.get('tracks', {}).get('items', [])
            if tracks:
                album_images = tracks[0].get('album', {}).get('images', [])
                if album_images:
                    return album_images[0]['url']  # Highest resolution
    except Exception:
        pass
    return None


def fetch_itunes_artwork(song_name, artist):
    """Fetch album artwork from iTunes API"""
    query = f"{song_name} {artist}".replace(" ", "+")
    url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"

    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            results = data.get('results', [])
            if results:
                artwork_url = results[0].get('artworkUrl100', '').replace('100x100', '600x600')
                return artwork_url
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def get_spotify_preview_url(song_name, artist):
    """Fetch preview URL from Spotify API"""
    token = get_spotify_access_token()
    if not token:
        return None

    query = f"{song_name} {artist}".replace(" ", "%20")
    search_url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"

    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.ok:
            data = response.json()
            tracks = data.get('tracks', {}).get('items', [])
            if tracks:
                preview_url = tracks[0].get('preview_url')
                return preview_url
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def get_itunes_preview_url(song_name, artist):
    """Fetch preview URL from iTunes API"""
    query = f"{song_name} {artist}".replace(" ", "+")
    url = f"https://itunes.apple.com/search?term={query}&entity=song&limit=1"

    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            results = data.get('results', [])
            if results:
                preview_url = results[0].get('previewUrl')
                return preview_url
    except Exception:
        pass
    return None


def image_from_url(url):
    """Download and return PIL Image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            return Image.open(BytesIO(response.content))
    except Exception:
        pass
    return None


def create_placeholder_artwork():
    """Create a placeholder album artwork"""
    img = Image.new('RGB', (300, 300), color=(64, 64, 64))
    return img


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def fetch_album_image(song_name, artist):
    """Fetch album artwork with fallback options"""
    # Try Spotify first
    image_url = fetch_spotify_artwork(song_name, artist)
    if image_url:
        img = image_from_url(image_url)
        if img:
            return img

    # Fallback to iTunes
    image_url = fetch_itunes_artwork(song_name, artist)
    if image_url:
        img = image_from_url(image_url)
        if img:
            return img

    # Final fallback: placeholder
    return create_placeholder_artwork()


def display_enhanced_song_details(df, song_title):
    """
    Display enhanced song details with album artwork and better formatting.
    """
    song = df[df['song_name'].str.lower() == song_title.lower()]
    if song.empty:
        return None

    song = song.iloc[0]

    st.markdown("### Song Details")

    # Fetch and display album artwork
    album_image = fetch_album_image(song.get('song_name', ''), song.get('artist', ''))
    if album_image:
        st.image(album_image, caption=f"{song.get('album_name', 'Unknown')} by {song.get('artist', 'Unknown')}", width=300)

    # Audio preview if available
    preview_url = get_spotify_preview_url(song.get('song_name', ''), song.get('artist', ''))
    if not preview_url:
        preview_url = get_itunes_preview_url(song.get('song_name', ''), song.get('artist', ''))
    if preview_url:
        st.audio(preview_url, format='audio/mp3')

    st.markdown("---")

    st.markdown(f"**Song:** {song.get('song_name', 'Unknown')}")
    st.markdown(f"**Artist:** {song.get('artist', 'Unknown')}")
    st.markdown(f"**Album:** {song.get('album_name', 'Unknown')}")
    st.markdown(f"**Genre:** {song.get('genre', 'Unknown').title()}")
    st.markdown(f"**Popularity:** {song.get('popularity', 0):.1f}/100")

    st.markdown("---")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Audio Attributes")
        duration = f"{int(song.get('duration_ms', 0) / 1000 // 60)}:{int(song.get('duration_ms', 0) / 1000 % 60):02d}"
        st.markdown(f"**Duration:** {duration}")
        st.markdown(f"**Tempo:** {song.get('tempo', 0):.0f} BPM")

    with right_col:
        st.markdown("#### Performance Metrics")
        st.markdown(f"**Energy:** {song.get('energy', 0):.2f}")
        st.markdown(f"**Danceability:** {song.get('danceability', 0):.2f}")

    st.markdown("---")

    additional_info = []
    if pd.notna(song.get('valence')):
        valence = song.get('valence', 0)
        if valence > 0.6:
            additional_info.append("Mood: Positive 😊")
        elif valence < 0.4:
            additional_info.append("Mood: Moody 😢")
        else:
            additional_info.append("Mood: Neutral 😐")

    if pd.notna(song.get('acousticness')):
        acoustic = song.get('acousticness', 0)
        acoustic_type = "Acoustic-forward" if acoustic > 0.5 else "Electronic-forward"
        additional_info.append(f"Style: {acoustic_type}")

    if additional_info:
        st.markdown("#### Additional Notes")
        for note in additional_info:
            st.markdown(f"- {note}")

    return song


def display_song_details(df, song_title):
    song = df[df['song_name'].str.lower() == song_title.lower()]
    if song.empty:
        return None
    song = song.iloc[0]
    return {
        'Song': song['song_name'],
        'Artist': song['artist'],
        'Album': song.get('album_name', "Unknown"),
        'Genre': song.get('genre', "Unknown"),
        'Popularity': song.get('popularity', np.nan),
    }


def render_score_breakdown(results: pd.DataFrame):
    if {'content_score', 'collab_score', 'hybrid_score'}.issubset(results.columns):
        breakdown = results[['song_name', 'content_score', 'collab_score', 'hybrid_score']].head(10).set_index('song_name')
        st.write("#### Score Contributors")
        st.bar_chart(breakdown)


def show_dataset_insights(df):
    top_songs = (
        df[['song_name', 'artist', 'popularity']]
        .drop_duplicates('song_name')
        .sort_values('popularity', ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    st.write("### 🎵 Top popular songs")
    st.dataframe(top_songs)

    top_artists = (
        df.groupby('artist')['popularity']
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )
    st.write("### 🌟 Top artist popularity")
    st.bar_chart(top_artists.rename(columns={'artist': 'Artist', 'popularity': 'Average popularity'}).set_index('Artist'))


st.title("Hybrid Music Recommendation System")
st.markdown(
    "## The most dynamic hybrid recommender experience for your music library"
)
st.markdown(
    "Choose a song seed, select or simulate a user profile, and explore how content-based and collaborative signals combine to surface surprising music discoveries."
)

with st.spinner("Loading dataset and recommender..."):
    df = load_data()
    df_content, df_cf, recommender, training_metrics = load_models(df)
    song_options = build_available_songs(df_content)
    user_options = build_user_options(df_cf)

with st.sidebar:
    st.header("Recommendation controls")
    mode = st.radio("Mode", ["Content-based", "Collaborative", "Hybrid"], index=2)

    if "song_title" not in st.session_state:
        st.session_state.song_title = song_options[0]
    if "user_selection" not in st.session_state:
        st.session_state.user_selection = user_options[0]
    if "generate_recommendations" not in st.session_state:
        st.session_state.generate_recommendations = False

    song_title = st.selectbox(
        "Song title",
        options=song_options,
        index=song_options.index(st.session_state.song_title) if st.session_state.song_title in song_options else 0,
        key="song_title",
    )
    user_selection = st.selectbox(
        "User profile",
        options=user_options,
        index=user_options.index(st.session_state.user_selection) if st.session_state.user_selection in user_options else 0,
        key="user_selection",
    )
    top_n = st.slider("Top N recommendations", min_value=5, max_value=20, value=10, step=1, key="top_n")
    alpha = st.slider(
        "Content weight (alpha)", 
        0.0, 1.0, 0.5, 0.05, 
        key="alpha",
        help="🎵 Controls how much the system focuses on songs similar to your seed song (artist, genre, style). Higher values = more similar music."
    )
    beta = st.slider(
        "Collaborative weight (beta)", 
        0.0, 1.0, 0.5, 0.05, 
        key="beta",
        help="👥 Controls how much the system considers what people with similar tastes enjoy. Higher values = more crowd-approved recommendations."
    )
    if mode != "Hybrid":
        st.info("Hybrid mode gives you the best of both worlds. Switch to Hybrid to blend signals.")
    else:
        if abs(alpha + beta - 1.0) > 1e-6:
            st.warning("💡 Tip: Alpha + Beta = 1.0 gives balanced results. The system auto-normalizes your values.")
        
        # Add quick explanation
        st.markdown("**Quick Guide:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"🎵 **Content (α={alpha:.1f})**: Song similarity focus")
        with col2:
            st.markdown(f"👥 **Collaborative (β={beta:.1f})**: User preference focus")

    if st.button("Randomize inputs"):
        st.session_state.song_title = np.random.choice(song_options)
        st.session_state.user_selection = np.random.choice(user_options)
        st.experimental_rerun()

    if st.button("Refresh model artifacts"):
        load_data.clear()
        load_models.clear()
        build_available_songs.clear()
        build_user_options.clear()
        st.experimental_rerun()

    st.markdown("---")
    if "generate_recommendations" not in st.session_state:
        st.session_state.generate_recommendations = False

    if mode != "Hybrid":
        generate_button = st.button("Generate recommendations")
        if generate_button:
            st.session_state.generate_recommendations = True
    else:
        # In hybrid mode, always enable recommendations and detect weight changes.
        st.session_state.generate_recommendations = True
        prev_alpha = st.session_state.get("prev_alpha", 0.5)
        prev_beta = st.session_state.get("prev_beta", 0.5)
        
        # If weights changed, update session state and trigger rerun.
        if abs(alpha - prev_alpha) > 0.001 or abs(beta - prev_beta) > 0.001:
            st.session_state.prev_alpha = alpha
            st.session_state.prev_beta = beta


    st.write("### Model insights")
    st.metric("Songs indexed", df_content.shape[0])
    st.metric("Synthetic users", len(df_cf['user_id'].unique()))
    st.metric("Unique song titles", df['song_name'].nunique())
    if training_metrics:
        st.write(training_metrics)
    else:
        st.write("Loaded saved model artifacts.")

selected_user_id = parse_user_selection(user_selection)

with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Your recommendation session")
        st.write(f"**Mode:** {mode}")
        st.write(f"**Song seed:** {song_title}")
        st.write(f"**User profile:** {user_selection}")
        st.write(f"**Recommendation count:** {top_n}")
        if mode == "Hybrid":
            st.write(f"**Alpha:** {alpha}, **Beta:** {beta}")

        if st.session_state.generate_recommendations:
            if mode == "Content-based":
                results = recommender.content_recommend(song_title, top_n=top_n)
            elif mode == "Collaborative":
                results = recommender.collaborative_recommend(selected_user_id if selected_user_id is not None else -1, song_title, top_n=top_n)
            else:
                recommender.alpha = float(alpha)
                recommender.beta = float(beta)
                results = recommender.hybrid_recommend(selected_user_id if selected_user_id is not None else -1, song_title, top_n=top_n)

            if results.empty:
                st.warning("No recommendations were found. Try another song or user profile.")
            else:
                st.success("Recommendations generated successfully.")
                if mode == "Hybrid":
                    st.write(f"### Results (Alpha={alpha:.2f} → Content | Beta={beta:.2f} → Collaborative)")
                    st.dataframe(results.reset_index(drop=True))
                    render_score_breakdown(results)
                else:
                    st.dataframe(results.reset_index(drop=True))
        else:
            if mode == "Hybrid":
                st.info("Adjust alpha/beta sliders to see recommendations update live.")
            else:
                st.info("Use the Generate recommendations button in the sidebar to see recommendations.")

    with right:
        details = display_enhanced_song_details(df, song_title)
        if details is None:
            st.write("Song metadata not available for this title.")
        st.markdown("---")
        show_dataset_insights(df)

st.markdown("---")
st.write(
    "### Notes\n"
    "- This interface uses the same recommender engine from `hybrid_recommender_validation.py`.\n"
    "- Pick a favorite song to seed the content model, then tune hybrid weights to explore discovery.\n"
    "- Refresh artifacts to rebuild your model if the dataset changes."
)
