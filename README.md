# Hybrid Music Recommendation System

A Python-based hybrid music recommendation system with Streamlit UI and supporting model code.

---
title: "Wangda Recommendation System"
emoji: "🎵"
colorFrom: "blue"
colorTo: "teal"
sdk: "docker"
sdk_version: "0.50.2"
python_version: "3.10"
app_file: src/streamlit_app.py
pinned: false
---

## Project structure

- `music_recommender_app.py` - main recommendation logic
- `streamlit_app.py` - Streamlit interface
- `hybrid_recommender_validation.py` - validation scripts
- `requirements.txt` - project dependencies
- `models/` - saved model files
- `plots/` - visualizations

## Run locally

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

> The Streamlit UI now includes a polished login and signup page to access the music recommender experience.

## GitLab setup

1. Create the GitLab project.
2. Connect the local repo:
   ```bash
   git remote add origin https://gitlab.com/enigma-group/music-recommendation-system.git
   git branch -M main
   ```
3. Push your code:
   ```bash
   git push -u origin main
   ```

## Contributing

Invite your teammates on GitLab under **Project > Members** and use branches + merge requests to collaborate.
