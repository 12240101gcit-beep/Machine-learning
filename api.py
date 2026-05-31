from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import traceback

# Import recommender utilities
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

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Hybrid Music Recommender API")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for request bodies
class ContentRequest(BaseModel):
    song_title: str
    top_n: Optional[int] = 10

class CollabRequest(BaseModel):
    user_id: int
    song_title: Optional[str] = None
    top_n: Optional[int] = 10

class HybridRequest(BaseModel):
    user_id: int
    song_title: str
    alpha: Optional[float] = 0.5
    beta: Optional[float] = 0.5
    top_n: Optional[int] = 10

# Global objects
recommender = None
_df_content = None
_df_cf = None


@app.on_event("startup")
def startup_load_models():
    global recommender, _df_content, _df_cf
    try:
        print("[api] Loading dataset and models...")
        df = load_dataset(DATA_PATH)
        df = create_text_features(df)
        df['song_key'] = df['song_name'].astype(str) + ' | ' + df['artist'].astype(str)

        vectorizer = load_or_build_tfidf(df, MODEL_DIR)
        similarity_matrix = load_or_build_similarity(df, vectorizer, MODEL_DIR, max_songs=5000)

        _df_content = df.head(similarity_matrix.shape[0]).reset_index(drop=True)
        _df_cf = synthesize_user_data(df)

        svd_model, training_metrics = load_or_train_svd(_df_cf, MODEL_DIR)
        recommender = HybridRecommender(_df_content, _df_cf, vectorizer, similarity_matrix, svd_model)
        print("[api] Models loaded successfully")
    except Exception as exc:
        print("[api] Failed to load models:", exc)
        traceback.print_exc()


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": recommender is not None}


def df_to_records(df):
    records = []
    for _, row in df.reset_index(drop=True).iterrows():
        rec = {}
        for col, val in row.items():
            # make sure values are json serializable
            if hasattr(val, 'tolist') and not isinstance(val, str):
                try:
                    rec[col] = val.tolist()
                except Exception:
                    rec[col] = str(val)
            else:
                try:
                    # cast numpy types
                    if hasattr(val, 'item'):
                        rec[col] = val.item()
                    else:
                        rec[col] = val
                except Exception:
                    rec[col] = str(val)
        records.append(rec)
    return records


@app.post("/recommend/content")
def recommend_content(req: ContentRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        df_res = recommender.content_recommend(req.song_title, top_n=req.top_n)
        return {"results": df_to_records(df_res)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/recommend/collab")
def recommend_collab(req: CollabRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        df_res = recommender.collaborative_recommend(req.user_id, req.song_title, top_n=req.top_n)
        return {"results": df_to_records(df_res)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/recommend/hybrid")
def recommend_hybrid(req: HybridRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        # normalize alpha/beta inside recommender call
        recommender.alpha = float(req.alpha)
        recommender.beta = float(req.beta)
        df_res = recommender.hybrid_recommend(req.user_id, req.song_title, top_n=req.top_n)
        return {"results": df_to_records(df_res)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")
