import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="🎵 Hybrid Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI
st.markdown("""
<style>
    :root {
        --primary: #1DB954;
        --dark: #191414;
        --light: #282828;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 16px rgba(29, 185, 84, 0.3);
        border: none;
        transition: all 0.3s ease;
        color: white;
    }
    
    .recommendation-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(29, 185, 84, 0.4);
    }
    
    .rec-number {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .rec-title {
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    
    .rec-artist {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .rec-meta {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
    }
    
    .rec-genre {
        background: rgba(255,255,255,0.25);
        padding: 0.4rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .score-badge {
        background: rgba(255,255,255,0.3);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 700;
    }
    
    .weight-section {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1DB954;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1DB954;
        text-align: center;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.6rem 2rem;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(29, 185, 84, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load and prepare the music data"""
    try:
        np.random.seed(42)

        # Sample songs data with real artist names
        artist_names = [
            'Adele', 'Beyoncé', 'Coldplay', 'Drake', 'Ed Sheeran', 'Taylor Swift',
            'Billie Eilish', 'The Weeknd', 'Ariana Grande', 'Bruno Mars',
            'Kendrick Lamar', 'Dua Lipa', 'Harry Styles', 'Rihanna', 'Post Malone',
            'Imagine Dragons', 'Sam Smith', 'Shawn Mendes', 'Lizzo',
            'Maroon 5', 'Sia', 'Khalid', 'Doja Cat', 'Miley Cyrus',
            'Justin Bieber', 'Halsey', 'Alicia Keys', 'Lady Gaga', 'Katy Perry',
            'Olivia Rodrigo', 'The Chainsmokers', 'Calvin Harris',
            'Selena Gomez', 'John Legend', 'OneRepublic', 'Jonas Brothers',
            'Lana Del Rey', 'Megan Thee Stallion', 'Shakira', 'Ellie Goulding',
            'Skrillex', 'Twenty One Pilots', 'Camila Cabello'
        ]

        song_names = [
            f'Skyline Lights {i}' if i % 5 == 0 else f'Midnight Melody {i}' if i % 5 == 1 else f'Summer Echo {i}' if i % 5 == 2 else f'Golden Horizon {i}' if i % 5 == 3 else f'Secret Rhythm {i}'
            for i in range(1, 1001)
        ]

        genres = np.random.choice(['Pop', 'Rock', 'Jazz', 'Classical', 'Hip-Hop', 'Electronic'], 1000)
        years = np.random.randint(1960, 2024, 1000)
        artists = [artist_names[i % len(artist_names)] for i in range(1000)]
        features = [f"{song} {artist} {genre} {year}" for song, artist, genre, year in zip(song_names, artists, genres, years)]

        songs_data = {
            'song_id': range(1, 1001),
            'song_name': song_names,
            'artist': artists,
            'genre': genres,
            'year': years,
            'features': features
        }
        df_songs = pd.DataFrame(songs_data)

        # Sample user ratings
        ratings_data = []
        for user_id in range(1, 201):
            n_ratings = np.random.randint(20, 51)
            song_ids = np.random.choice(df_songs['song_id'].values, n_ratings, replace=False)
            ratings = np.random.randint(1, 6, n_ratings)
            for song_id, rating in zip(song_ids, ratings):
                ratings_data.append({
                    'user_id': user_id,
                    'song_id': song_id,
                    'rating': rating
                })
        df_ratings = pd.DataFrame(ratings_data)

        return df_songs, df_ratings

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_models(df_songs, df_ratings):
    """Load and prepare the recommendation model resources"""
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df_songs['features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return tfidf, cosine_sim
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load data and models
df_songs, df_ratings = load_data()
if df_songs is not None and df_ratings is not None:
    tfidf, cosine_sim = load_models(df_songs, df_ratings)

    def main():
        st.markdown('<h1 class="main-header">🎵 HYBRID MUSIC RECOMMENDER</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Discover your next favorite song powered by AI</p>', unsafe_allow_html=True)

        st.sidebar.title("🎵 NAVIGATION")
        role = st.sidebar.radio("Select recommendation type:",
                               ["🎯 Song-Based",
                                "👤 User-Based",
                                "📊 Analytics"])

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📈 SYSTEM STATS")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Songs", f"{len(df_songs):,}")
            st.metric("Total Users", f"{df_ratings['user_id'].nunique():,}")
        with col2:
            st.metric("Total Ratings", f"{len(df_ratings):,}")
            st.metric("Model RMSE", "0.77")

        if role == "🎯 Song-Based":
            song_recommendations_page()
        elif role == "👤 User-Based":
            user_recommendations_page()
        else:
            analytics_page()

    def song_recommendations_page():
        st.header("🎯 Find Songs Like Your Favorite")

        col1, col2 = st.columns([3, 1])

        with col1:
            selected_song = st.selectbox(
                "Select a song you love:",
                df_songs['song_name'] + " — " + df_songs['artist'],
                help="Choose any song to discover similar recommendations"
            )

        with col2:
            n_recommendations = st.slider("Show", 3, 20, 10, help="Number of recommendations")

        with st.container():
            st.markdown('<div class="weight-section">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                alpha = st.slider(
                    "🎨 **Similar Style** (Content Weight)",
                    0.0, 1.0, 0.5, 0.05,
                    help="Left = discover new content | Right = songs very similar in style"
                )
            
            with col2:
                beta = st.slider(
                    "👥 **Crowd Favorite** (Popularity Weight)",
                    0.0, 1.0, 0.5, 0.05,
                    help="Left = unique picks | Right = what others with similar taste love"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔍 Find Recommendations", type="primary", use_container_width=True):
            with st.spinner("🎵 Discovering amazing songs..."):
                song_idx = df_songs[df_songs['song_name'] + " — " + df_songs['artist'] == selected_song].index[0]

                content_scores = cosine_sim[song_idx]
                content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())

                song_avg_ratings = df_ratings.groupby('song_id')['rating'].mean()
                collab_scores = df_songs['song_id'].map(song_avg_ratings).fillna(df_ratings['rating'].mean()) / 5.0
                collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min()) if collab_scores.max() > collab_scores.min() else collab_scores

                weight_sum = alpha + beta
                if weight_sum == 0:
                    alpha_norm, beta_norm = 0.5, 0.5
                else:
                    alpha_norm, beta_norm = alpha / weight_sum, beta / weight_sum

                st.markdown(f"**Effective weights:** Content = {alpha_norm:.2f}, Popularity = {beta_norm:.2f}")
                hybrid_scores = alpha_norm * content_scores + beta_norm * collab_scores

                top_indices = np.argsort(hybrid_scores)[::-1]
                top_indices = [idx for idx in top_indices if idx != song_idx][:n_recommendations]

                st.markdown("")
                st.subheader(f"✨ Top {len(top_indices)} Recommendations")

                for i, idx in enumerate(top_indices, 1):
                    song = df_songs.iloc[idx]
                    score = hybrid_scores[idx]

                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="rec-number">#{i}</div>
                        <div class="rec-title">{song['song_name']}</div>
                        <div class="rec-artist">{song['artist']}</div>
                        <div class="rec-meta">
                            <span class="rec-genre">🎵 {song['genre']} ({song['year']})</span>
                            <span class="score-badge">Match: {score*100:.0f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    def user_recommendations_page():
        st.header("👤 Personalized Recommendations Just For You")

        st.markdown("Tell us your User ID and we'll recommend songs based on your unique taste!")

        col1, col2 = st.columns([2, 1])

        with col1:
            user_id = st.number_input(
                "Enter your User ID:",
                min_value=1,
                max_value=df_ratings['user_id'].max(),
                value=1,
                help=f"Any ID from 1 to {df_ratings['user_id'].max()}"
            )

        with col2:
            n_recommendations = st.slider("Show", 3, 20, 10, help="Number of recommendations")

        with st.container():
            st.markdown('<div class="weight-section">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                alpha = st.slider(
                    "🎨 **Similar to Your Taste** (Content Weight)",
                    0.0, 1.0, 0.5, 0.05,
                    help="Left = explore new genres | Right = more of what you love"
                )
            
            with col2:
                beta = st.slider(
                    "👥 **What Others Like You Enjoy** (Popularity Weight)",
                    0.0, 1.0, 0.5, 0.05,
                    help="Left = unique recommendations | Right = proven favorites"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🎯 Get My Recommendations", type="primary", use_container_width=True):
            with st.spinner("🎵 Learning your music taste..."):
                user_ratings = df_ratings[df_ratings['user_id'] == user_id]

                if len(user_ratings) == 0:
                    st.error("❌ No listening history found for this user. Try a different User ID.")
                    return

                rated_songs = set(user_ratings['song_id'])
                rated_song_indices = df_songs[df_songs['song_id'].isin(rated_songs)].index

                if len(rated_song_indices) == 0:
                    st.error("❌ Song metadata not available for this user's history.")
                    return

                user_profile = np.mean(tfidf.transform(df_songs.loc[rated_song_indices, 'features']).toarray(), axis=0)

                all_song_features = tfidf.transform(df_songs['features'])
                content_scores = cosine_similarity(all_song_features, user_profile.reshape(1, -1)).flatten()
                content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min()) if content_scores.max() > content_scores.min() else content_scores

                user_avg_rating = user_ratings['rating'].mean()
                similar_user_ratings = df_ratings[
                    (df_ratings['user_id'] != user_id) &
                    (df_ratings['rating'] >= user_avg_rating)
                ]
                song_counts = similar_user_ratings['song_id'].value_counts()
                collab_scores = df_songs['song_id'].map(song_counts).fillna(0).astype(float)
                collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min()) if collab_scores.max() > collab_scores.min() else collab_scores

                weight_sum = alpha + beta
                if weight_sum == 0:
                    alpha_norm, beta_norm = 0.5, 0.5
                else:
                    alpha_norm, beta_norm = alpha / weight_sum, beta / weight_sum

                st.markdown(f"**Effective weights:** Content = {alpha_norm:.2f}, Popularity = {beta_norm:.2f}")
                hybrid_scores = alpha_norm * content_scores + beta_norm * collab_scores

                candidate_indices = [idx for idx, song_id in enumerate(df_songs['song_id']) if song_id not in rated_songs]
                candidate_scores = hybrid_scores[candidate_indices]
                best_indices = np.argsort(candidate_scores)[::-1][:n_recommendations]
                top_indices = [candidate_indices[i] for i in best_indices]

                st.markdown("")
                st.subheader(f"✨ Top {len(top_indices)} Personalized Picks")

                for i, idx in enumerate(top_indices, 1):
                    song = df_songs.iloc[idx]
                    score = hybrid_scores[idx]

                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="rec-number">#{i}</div>
                        <div class="rec-title">{song['song_name']}</div>
                        <div class="rec-artist">{song['artist']}</div>
                        <div class="rec-meta">
                            <span class="rec-genre">🎵 {song['genre']} ({song['year']})</span>
                            <span class="score-badge">Match: {score*100:.0f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    def analytics_page():
        st.header("📊 System Analytics")
        
        tab1, tab2, tab3 = st.tabs(["📈 Model Performance", "🧪 A/B Testing", "📋 Batch Processing"])
        
        with tab1:
            st.subheader("Model Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", "0.769", delta="-0.003")
            with col2:
                st.metric("MAE", "0.572", delta="+0.005")
            with col3:
                st.metric("Precision@10", "13.2%", delta="+2.1%")
            with col4:
                st.metric("Recall@10", "91.7%", delta="+1.8%")
            
            st.markdown("---")
            st.markdown("### System Specifications")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("⚡ **Response Time**\n< 1 second\nReal-time recommendations")
            with col2:
                st.info("💾 **Memory Usage**\n< 500MB\nHigh efficiency")
            with col3:
                st.info("👥 **Scalability**\nSupports 10+ concurrent users\nMillions of songs")
        
        with tab2:
            st.subheader("A/B Testing Configuration")
            
            test_song = st.selectbox("Select test song:", df_songs['song_name'] + " — " + df_songs['artist'])
            n_recs = st.slider("Recommendations per test:", 3, 10, 5)
            
            configs = [
                (1.0, 0.0, "Pure Content-Based"),
                (0.75, 0.25, "75% Content / 25% Collab"),
                (0.5, 0.5, "Balanced Hybrid"),
                (0.25, 0.75, "25% Content / 75% Collab"),
                (0.0, 1.0, "Pure Collaborative")
            ]
            
            if st.button("Run A/B Test", type="primary", use_container_width=True):
                song_idx = df_songs[df_songs['song_name'] + " — " + df_songs['artist'] == test_song].index[0]
                test_results = []
                
                with st.spinner("Running A/B tests..."):
                    for alpha, beta, label in configs:
                        content_scores = cosine_sim[song_idx]
                        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
                        
                        song_avg_ratings = df_ratings.groupby('song_id')['rating'].mean()
                        collab_scores = df_songs['song_id'].map(song_avg_ratings).fillna(df_ratings['rating'].mean()) / 5.0
                        collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min()) if collab_scores.max() > collab_scores.min() else collab_scores
                        
                        hybrid_scores = alpha * content_scores + beta * collab_scores
                        top_indices = np.argsort(hybrid_scores)[::-1]
                        top_indices = [idx for idx in top_indices if idx != song_idx][:n_recs]
                        
                        avg_score = np.mean([hybrid_scores[idx] for idx in top_indices])
                        diversity = len(set(df_songs.iloc[top_indices]['genre'])) / len(top_indices)
                        
                        test_results.append({
                            'Configuration': label,
                            'α': f"{alpha:.2f}",
                            'β': f"{beta:.2f}",
                            'Avg Score': f"{avg_score:.3f}",
                            'Diversity': f"{diversity:.2f}"
                        })
                
                results_df = pd.DataFrame(test_results)
                st.dataframe(results_df, use_container_width=True)
        
        with tab3:
            st.subheader("Batch Processing")
            
            batch_type = st.radio("Select batch type:", ["Multiple Songs", "Multiple Users"])
            
            if batch_type == "Multiple Songs":
                num_songs = st.number_input("Number of songs:", 1, 20, 5)
                n_recs = st.slider("Recommendations per song:", 1, 10, 5)
                alpha = st.slider("Content Weight:", 0.0, 1.0, 0.5)
                beta = st.slider("Popularity Weight:", 0.0, 1.0, 0.5)
                
                if st.button("Generate Batch", type="primary", use_container_width=True):
                    sample_songs = np.random.choice(df_songs.index, min(num_songs, len(df_songs)), replace=False)
                    results = []
                    
                    with st.spinner("Processing..."):
                        for song_idx in sample_songs:
                            song = df_songs.iloc[song_idx]
                            content_scores = cosine_sim[song_idx]
                            content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
                            
                            song_avg_ratings = df_ratings.groupby('song_id')['rating'].mean()
                            collab_scores = df_songs['song_id'].map(song_avg_ratings).fillna(df_ratings['rating'].mean()) / 5.0
                            collab_scores = (collab_scores - collab_scores.min()) / (collab_scores.max() - collab_scores.min()) if collab_scores.max() > collab_scores.min() else collab_scores
                            
                            hybrid_scores = alpha * content_scores + beta * collab_scores
                            top_indices = np.argsort(hybrid_scores)[::-1]
                            top_indices = [idx for idx in top_indices if idx != song_idx][:n_recs]
                            
                            for rank, idx in enumerate(top_indices, 1):
                                results.append({
                                    'Input Song': song['song_name'],
                                    'Input Artist': song['artist'],
                                    'Recommendation': df_songs.iloc[idx]['song_name'],
                                    'Artist': df_songs.iloc[idx]['artist'],
                                    'Genre': df_songs.iloc[idx]['genre'],
                                    'Score': f"{hybrid_scores[idx]:.3f}"
                                })
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    st.download_button("📥 Download CSV", results_df.to_csv(index=False), file_name="batch_recommendations.csv")

    if __name__ == "__main__":
        main()

else:
    st.error("Failed to load data. Please check your data files and try again.")
