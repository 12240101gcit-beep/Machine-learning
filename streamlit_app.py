import os
import json
import hashlib
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import re
from io import BytesIO
from PIL import Image
import time
import base64
from hybrid_recommender_validation import (
    load_dataset,
    create_text_features,
    load_or_build_tfidf,
    load_or_build_similarity,
    synthesize_user_data,
    load_or_train_svd,
    HybridRecommender,
    DATA_PATH,
    MODEL_DIR,
)

import base64
from pathlib import Path

# --- Embedded auth helpers (restored from auth.py) ---
USERS_FILE = Path("users_db.json")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def init_user_store() -> None:
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({"users": {}}, indent=2))


def load_users() -> dict:
    try:
        init_user_store()
        with USERS_FILE.open('r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('users', {})
    except Exception:
        return {}


def save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open('w', encoding='utf-8') as f:
        json.dump({"users": users}, f, indent=2)


def authenticate_user(email: str, password: str) -> tuple:
    users = load_users()
    stored = users.get(email.lower())
    if not stored:
        return False, "No account found for that email."
    if stored.get('password') != hash_password(password):
        return False, "Incorrect password."
    return True, stored.get('name', email)


def register_user(name: str, email: str, password: str) -> tuple:
    users = load_users()
    email_key = email.lower()
    if email_key in users:
        return False, "An account using that email already exists."
    users[email_key] = {'name': name.strip() or email_key, 'password': hash_password(password)}
    save_users(users)
    return True, "Account created successfully."


def validate_email(email: str) -> bool:
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email.strip()) is not None


def show_auth_page() -> None:
    # (auth UI content restored)
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, rgba(30,215,96,0.18), transparent 18%),
                            radial-gradient(circle at bottom right, rgba(29,185,84,0.16), transparent 22%),
                            linear-gradient(135deg, #050505 0%, #0f0f0f 38%, #121212 100%);
                color: #ffffff;
            }
            .main .block-container,
            .block-container {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                margin: 0 auto !important;
                max-width: 1180px;
                min-height: 100vh;
            }
            .spotify-auth-shell {
                display: flex;
                align-items: flex-start;
                justify-content: center;
                padding: 3.5rem 1rem 4rem !important;
                margin: 0 !important;
                overflow: visible;
                position: relative;
                background: radial-gradient(circle at 20% 15%, rgba(29,185,84,0.2), transparent 20%),
                            radial-gradient(circle at 80% 20%, rgba(59,130,246,0.18), transparent 22%),
                            radial-gradient(circle at 50% 80%, rgba(148,163,184,0.1), transparent 25%);
                animation: pagePulse 14s ease infinite;
            }
            .spotify-auth-shell::before {
                content: "";
                position: absolute;
                inset: 0;
                background: radial-gradient(circle at 10% 10%, rgba(255,255,255,0.08), transparent 16%),
                            radial-gradient(circle at 85% 15%, rgba(29,185,84,0.08), transparent 16%),
                            radial-gradient(circle at 70% 80%, rgba(14,165,233,0.06), transparent 20%);
                pointer-events: none;
            }
            .spotify-auth-card {
                position: relative;
                width: 100%;
                max-width: 1040px;
                background: linear-gradient(180deg, rgba(12,15,22,0.98), rgba(7,8,12,0.96));
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 36px;
                box-shadow: 0 42px 100px rgba(0,0,0,0.55);
                overflow: visible !important;
                max-height: none;
                animation: floatUp 0.9s ease-out;
                backdrop-filter: blur(10px);
            }
            @keyframes pagePulse {
                0%,100% { background-position: 20% 15%, 80% 20%, 50% 80%; }
                50% { background-position: 25% 10%, 75% 25%, 55% 78%; }
            }
            .spotify-auth-card:hover {
                transform: translateY(-4px);
                transition: transform 0.25s ease;
            }
            .spotify-auth-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
                gap: 1px;
            }
            @keyframes floatUp {
                from { transform: translateY(16px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .spotify-auth-panel,
            .spotify-auth-form {
                padding: 2.5rem;
            }
            /* Make the entire auth area pure black and remove panel borders */
            html, body, .stApp, .streamlit-expanderHeader, .main {
                background: #000000 !important;
            }
            .spotify-auth-shell,
            .spotify-auth-card,
            .spotify-auth-panel {
                background: transparent !important;
                background-color: #000000 !important;
                border: none !important;
                box-shadow: none !important;
            }
            /* center contents and give the logo its own space */
            .spotify-auth-panel {
                display: flex !important;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
                padding: 2.5rem 2rem !important;
                gap: 1rem;
            }
            .spotify-auth-panel .stImage {
                width: 100%;
                max-width: 190px;
                margin: 0 auto 1.5rem;
                padding: 0 !important;
                border-radius: 0 !important;
                background: transparent !important;
                box-shadow: none !important;
            }
            .spotify-auth-panel .stImage>div,
            .spotify-auth-panel .stImage>div>img {
                background: transparent !important;
                width: 100% !important;
                border-radius: 50% !important;
                border: none !important;
                display: block !important;
            }
            /* Add a subtle green ring / glow behind the circular logo */
            .spotify-auth-panel .stImage>div {
                position: relative !important;
                width: 170px !important;
                height: 170px !important;
                display: block !important;
                margin: 0 auto 0.9rem !important;
                border-radius: 50% !important;
                overflow: visible !important;
            }
            .spotify-auth-panel .stImage>div::after {
                content: "";
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 320px;
                height: 320px;
                border-radius: 50%;
                background: radial-gradient(circle at center, rgba(29,185,84,0.22), rgba(29,185,84,0.08) 35%, transparent 50%);
                filter: blur(16px);
                z-index: 0;
                pointer-events: none;
            }
            .spotify-auth-panel .stImage>div>img {
                position: relative !important;
                z-index: 1 !important;
            }
            /* Also support the explicit auth-logo wrapper we render via markdown */
            .spotify-auth-panel .auth-logo,
            .auth-logo {
                position: relative !important;
                width: 170px !important;
                height: 170px !important;
                margin: 0 auto 0.9rem !important;
                border-radius: 50% !important;
                overflow: visible !important;
                display: block !important;
            }
            .spotify-auth-panel .auth-logo::after,
            .auth-logo::after {
                content: "";
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: 320px;
                height: 320px;
                border-radius: 50%;
                background: radial-gradient(circle at center, rgba(29,185,84,0.24), rgba(29,185,84,0.08) 35%, transparent 50%);
                filter: blur(16px);
                z-index: 0;
                pointer-events: none;
            }
            .spotify-auth-panel .auth-logo img,
            .auth-logo img {
                position: relative !important;
                z-index: 1 !important;
                border-radius: 50% !important;
            }
            .spotify-auth-panel h1 {
                font-size: clamp(2.4rem, 3vw, 3.2rem);
                margin: 0 0 1rem;
                line-height: 1.02;
            }
            .spotify-auth-panel p {
                color: #d6d6d8;
                font-size: 1rem;
                line-height: 1.75;
                margin-bottom: 1.75rem;
                max-width: 30rem;
            }
            .spotify-auth-feature {
                display: flex;
                gap: 0.85rem;
                margin-bottom: 1rem;
                align-items: flex-start;
            }
            .spotify-auth-feature span {
                width: 2rem;
                min-width: 2rem;
                height: 2rem;
                border-radius: 50%;
                background: rgba(29,185,84,0.18);
                color: #22c55e;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
            }
            .spotify-auth-feature div {
                color: #cbd5e1;
                font-size: 0.98rem;
                line-height: 1.6;
            }
            .spotify-auth-form {
                background: rgba(7, 7, 10, 0.98);
                position: relative;
                overflow: hidden;
            }
            .spotify-auth-form::before {
                content: "";
                position: absolute;
                top: 0;
                right: -40px;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(29,185,84,0.16), transparent 55%);
                filter: blur(40px);
                pointer-events: none;
            }
            .spotify-auth-form h2 {
                margin-bottom: 0.35rem;
                font-size: 2rem;
                color: #ffffff;
            }
            .spotify-auth-card::after {
                content: "";
                position: absolute;
                left: -40px;
                top: -40px;
                width: 280px;
                height: 280px;
                background: radial-gradient(circle at center, rgba(29,185,84,0.14), transparent 45%);
                filter: blur(30px);
                z-index: 1;
                pointer-events: none;
            }
            .spotify-auth-panel > * { position: relative; z-index: 2; }
            .spotify-auth-panel h1 { text-shadow: 0 10px 30px rgba(29,185,84,0.06), 0 2px 8px rgba(0,0,0,0.6); }
            .spotify-auth-feature span { box-shadow: 0 10px 30px rgba(29,185,84,0.10), inset 0 1px 0 rgba(255,255,255,0.02); }
            .spotify-auth-form .stButton>button { box-shadow: 0 22px 50px rgba(29,185,84,0.16) !important; }
            .oauth-button:hover { box-shadow: 0 14px 40px rgba(29,185,84,0.18), 0 6px 18px rgba(0,0,0,0.45); transform: translateY(-2px); }
            .spotify-auth-form::after { content: ""; position: absolute; right: -80px; bottom: -80px; width: 260px; height: 260px; background: radial-gradient(circle at center, rgba(29,185,84,0.08), transparent 40%); filter: blur(28px); pointer-events: none; z-index: 1; }
            .spotify-auth-form .form-subtitle { color: #94a3b8; margin-bottom: 1.5rem; }
            .spotify-auth-form .stTextInput>div>div>input { border-radius: 0.95rem !important; border: 1px solid rgba(255,255,255,0.14) !important; background: rgba(255,255,255,0.05) !important; color: #f8fafc !important; min-height: 3rem; }
            .spotify-auth-form .stTextInput>div>label { color: #cbd5e1 !important; }
            .spotify-auth-form .stCaption { color: #8b98a8 !important; margin-top: -0.3rem; margin-bottom: 1rem; }
            .spotify-auth-form .stButton>button { background: #1db954 !important; color: white !important; border-radius: 999px !important; padding: 0.95rem 1rem !important; font-weight: 700 !important; box-shadow: 0 20px 35px rgba(29, 185, 84, 0.22) !important; }
            .oauth-button { width: 100%; display: inline-flex; align-items: center; gap: 0.85rem; text-align: left; margin-bottom: 0.9rem; padding: 0.95rem 1rem; border-radius: 999px; color: #ffffff !important; border: 1px solid rgba(255,255,255,0.12); background: rgba(255,255,255,0.05); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); transition: background 0.2s ease, transform 0.2s ease; }
            .oauth-button a { width:100%; display:inline-flex; align-items:center; }
            .oauth-button:hover { background: rgba(255,255,255,0.14); transform: translateY(-1px); }
            .oauth-icon { width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; border-radius: 999px; background: rgba(255,255,255,0.1); }
            .oauth-icon svg { width: 18px; height: 18px; }
            .spotify-auth-footer { margin-top: 1.5rem; color: #94a3b8; font-size: 0.95rem; }
            .spotify-auth-footer a { color: #22c55e; text-decoration: none; }
            @media (max-width: 900px) { .spotify-auth-grid { grid-template-columns: 1fr; } .spotify-auth-panel, .spotify-auth-form { padding: 1.75rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="spotify-auth-shell">', unsafe_allow_html=True)
    st.markdown('<div class="spotify-auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="spotify-auth-grid">', unsafe_allow_html=True)

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.markdown('<div class="spotify-auth-panel">', unsafe_allow_html=True)
        # Inline the logo as base64 so we can reliably style and center it
        logo_path = Path("wangdalockedin.jpeg")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<div class="auth-logo"><img src="data:image/jpeg;base64,{logo_b64}" alt="logo" style="width:170px;max-width:100%;border-radius:50%;display:block;margin:0 auto;"></div>'
        else:
            logo_html = '<div class="auth-logo"><img alt="logo" style="width:170px;max-width:100%;border-radius:50%;display:block;margin:0 auto;opacity:0.2"></div>'
        st.markdown(logo_html, unsafe_allow_html=True)
        st.markdown(
            """
                <h1>Welcome to your premium music lab.</h1>
                <p>Sign in, feel the vibe, and step directly into your hybrid music discovery experience.</p>
                <div class="spotify-auth-feature"><span>✓</span><div>Immersive dark aesthetic that feels like a modern music app.</div></div>
                <div class="spotify-auth-feature"><span>✓</span><div>Fast account access with polished spacing and glow accents.</div></div>
                <div class="spotify-auth-feature"><span>✓</span><div>Seamless login and signup flow for the best first impression.</div></div>
                <div class="spotify-auth-feature"><span>✓</span><div>Immediate access to personalized recommendations.</div></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            """
            <div class="spotify-auth-form">
                <h2>Welcome back</h2>
                <p class="form-subtitle">Log in or create an account to continue to your personalized song recommendations.</p>
            """,
            unsafe_allow_html=True,
        )

        tabs = st.tabs(["Login", "Sign up"])

        with tabs[0]:
            st.session_state.auth_mode = "Login"
            with st.form("auth_login_form"):
                email = st.text_input("Email address", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                st.markdown(
                    '''
                    <div class="oauth-button">
                        <a href="https://accounts.google.com/signin/v2/identifier" target="_blank" rel="noopener noreferrer" style="text-decoration:none;color:inherit;display:flex;align-items:center;gap:0.85rem;">
                            <span class="oauth-icon">
                                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M23.4 12.2c0-.8-.1-1.6-.3-2.4H12v4.6h6.4c-.3 1.6-1.3 3-2.8 3.9v3.2h4.6c2.7-2.5 4.2-6.2 4.2-10.3Z" fill="#4285F4"/>
                                    <path d="M12 24c3.3 0 6.1-1.1 8.1-3l-4.6-3.2c-1.3.9-3 1.5-5.2 1.5-4 0-7.3-2.7-8.5-6.3H.7v3.9C2.7 21.9 7 24 12 24Z" fill="#34A853"/>
                                    <path d="M3.5 14.9c-.3-1-.5-2-.5-3s.2-2 .5-3V5.9H.7C-.3 8.1-.9 10.5-.9 12.9c0 2.4.6 4.8 1.9 7l2.5-2.9Z" fill="#FBBC05"/>
                                    <path d="M12 4.8c1.8 0 3.4.6 4.7 1.7l3.5-3.5C18.1 1.1 15.3 0 12 0 7 0 2.7 2.1.7 5.9l2.8 2.6C4.7 7.5 8 4.8 12 4.8Z" fill="#EA4335"/>
                                </svg>
                            </span>
                            Continue with Google
                        </a>
                    </div>
                    <div class="oauth-button">
                        <a href="https://appleid.apple.com/" target="_blank" rel="noopener noreferrer" style="text-decoration:none;color:inherit;display:flex;align-items:center;gap:0.85rem;">
                            <span class="oauth-icon">
                                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M16.365 1.43c-.98.06-2.18.63-2.86 1.4-.62.72-1.16 1.76-1.02 2.87 1.12.08 2.22-.55 2.88-1.43.62-.79 1.15-1.79 1-2.84z" fill="#FFFFFF"/>
                                    <path d="M20.17 10.15c-.04-2.02.85-3.57 2.49-4.72-1.03-1.49-2.56-2.32-4.07-2.35-1.72-.03-3.36 1.02-4.19 1.02-.93 0-2.35-.99-3.86-.96-1.99.03-3.83 1.16-4.86 2.92-2.05 3.5-.53 8.96 1.45 11.9.97 1.44 2.13 3.06 3.68 2.99 1.5-.06 2.06-.98 3.88-.98 1.82 0 2.33.98 3.88.95 1.58-.03 2.59-1.47 3.6-2.9 1.13-1.61 1.6-3.17 1.64-3.24-.02-.02-2.15-.82-2.19-3.25zm-1.62-7.17c.63-.85 1.02-2.02.86-3.18-.88.04-1.94.64-2.51 1.5-.56.82-1.05 1.96-.88 3.12 1.05.08 2.06-.56 2.53-1.44z" fill="#FFFFFF"/>
                                </svg>
                            </span>
                            Continue with Apple
                        </a>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
                submit = st.form_submit_button("Log in")
                if submit:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                    elif not validate_email(email):
                        st.error("Enter a valid email address.")
                    else:
                        success, message = authenticate_user(email, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.current_user = message
                            st.session_state.auth_loading = True
                            st.success(f"Welcome back, {message}! Preparing your music feed...")
                            st.info("Loading your dashboard now... just a moment.")
                            time.sleep(0.8)
                            st.rerun()
                        else:
                            st.error(message)

        with tabs[1]:
            st.session_state.auth_mode = "Sign up"
            with st.form("auth_signup_form"):
                name = st.text_input("Full name", key="signup_name")
                email = st.text_input("Email address", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                st.caption("Use at least 8 characters with letters, numbers, and a symbol for the best security.")
                submit = st.form_submit_button("Create account")
                if submit:
                    if not name or not email or not password or not confirm:
                        st.error("All fields are required to create an account.")
                    elif not validate_email(email):
                        st.error("Enter a valid email address.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 8:
                        st.error("Password should be at least 8 characters.")
                    else:
                        success, message = register_user(name, email, password)
                        if success:
                            st.success("Account created successfully. Please log in now.")
                            st.session_state.auth_mode = "Login"
                            st.session_state.auth_message = message
                            st.session_state.auth_loading = True
                            st.info("Taking you to the login screen...")
                            time.sleep(0.8)
                            st.rerun()
                        else:
                            st.error(message)

        if st.session_state.get('auth_message'):
            st.info(st.session_state['auth_message'])

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Ensure user store exists
init_user_store()


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


@st.cache_data(show_spinner=False, ttl=3600)
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
                    return album_images[0]['url']
    except Exception:
        pass
    return None


def fetch_itunes_artwork(song_name, artist):
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


@st.cache_data(show_spinner=False, ttl=3600)
def get_spotify_preview_url(song_name, artist):
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


@st.cache_data(show_spinner=False, ttl=3600)
def get_itunes_preview_url(song_name, artist):
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
    try:
        response = requests.get(url, timeout=10)
        if response.ok:
            return Image.open(BytesIO(response.content))
    except Exception:
        pass
    return None


def create_placeholder_artwork(size=(600, 600), color=(30, 30, 30)):
    img = Image.new('RGB', size, color)
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


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = "Guest"
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"
if "auth_message" not in st.session_state:
    st.session_state.auth_message = ""
if "auth_loading" not in st.session_state:
    st.session_state.auth_loading = False

if not st.session_state.authenticated:
    show_auth_page()
    st.stop()

st.title(f"Welcome back, {st.session_state.current_user} 🎶")
st.markdown(
    "## The most dynamic hybrid recommender experience for your music library"
)
st.markdown(
    "Choose a song seed, select or simulate a user profile, and explore how content-based and collaborative signals combine to surface surprising music discoveries."
)

spinner_text = "Loading your music feed..." if st.session_state.auth_loading else "Loading dataset and recommender..."
with st.spinner(spinner_text):
    df = load_data()
    df_content, df_cf, recommender, training_metrics = load_models(df)
    song_options = build_available_songs(df_content)
    user_options = build_user_options(df_cf)

st.session_state.auth_loading = False

with st.sidebar:
    st.markdown(f"### Signed in as **{st.session_state.current_user}**")
    if st.button("Sign out"):
        st.session_state.authenticated = False
        st.session_state.current_user = "Guest"
        st.session_state.auth_message = "You have signed out successfully."
        st.rerun()

    st.markdown("---")
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
        st.rerun()

    if st.button("Refresh model artifacts"):
        load_data.clear()
        load_models.clear()
        build_available_songs.clear()
        build_user_options.clear()
        st.rerun()

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
