# Hybrid Music Recommendation System

A Python-based hybrid music recommendation system with Streamlit UI and supporting model code.

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
   source venv/Scripts/activate  # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

## GitLab setup

1. Create a GitLab project.
2. Add the GitLab remote and push:
   ```bash
   git remote add origin <your-gitlab-repo-url>
   git branch -M main
   git push -u origin main
   ```
