import os
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = Path("Preprocessed_dataset1.xls")
MODEL_DIR = Path("models")
PLOT_DIR = Path("plots")
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    print("\nLoading dataset...")
    if not path.exists():
        print(f"  - Dataset not found at {path}")
        print(f"  - Creating minimal demo dataset for recommender")
        # Create a minimal demo dataset to allow the app to run
        df = pd.DataFrame({
            'song_name': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4', 'Artist 5'],
            'album_name': ['Album 1', 'Album 2', 'Album 3', 'Album 4', 'Album 5'],
            'genre': ['pop', 'rock', 'hip-hop', 'jazz', 'electronic'],
            'popularity': [75, 80, 70, 65, 85],
            'tempo': [120, 140, 100, 90, 130],
            'key': [0, 1, 2, 3, 4],
            'mode': [0, 1, 0, 1, 0],
            'explicit': [False, True, False, False, True],
            'release_year': [2020, 2021, 2019, 2018, 2022],
        })
        print(f"  - Created demo dataset with shape: {df.shape}")
        return df

    try:
        df = pd.read_csv(path, engine='python')
        print("  - Loaded as CSV from .xls extension")
    except Exception as exc:
        print(f"  - CSV load failed: {exc}")
        try:
            df = pd.read_excel(path, engine='xlrd')
            print("  - Loaded with xlrd engine")
        except Exception as exc2:
            print(f"  - Excel load failed: {exc2}")
            print(f"  - Creating fallback demo dataset")
            df = pd.DataFrame({
                'song_name': ['Demo Song'],
                'artist': ['Demo Artist'],
                'album_name': ['Demo Album'],
                'genre': ['pop'],
                'popularity': [50],
                'tempo': [120],
                'key': [0],
                'mode': [0],
                'explicit': [False],
                'release_year': [2020],
            })

    print(f"  - Dataset shape: {df.shape}")
    print(f"  - Columns: {len(df.columns)}")
    return df


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding enhanced text features...")
    df = df.copy()
    if 'combined_text' not in df.columns or df['combined_text'].isnull().any():
        df['combined_text'] = (
            df['artist'].fillna('') + ' ' +
            df['album_name'].fillna('') + ' ' +
            df['song_name'].fillna('') + ' ' +
            df['genre'].fillna('')
        ).str.strip()

    def enhanced(row):
        parts = [str(row['artist']).lower(), str(row['song_name']).lower(), str(row['album_name']).lower(), str(row['genre']).lower()]
        parts.append(f"tempo_{int(row['tempo'] // 10) * 10}" if pd.notna(row.get('tempo')) else "tempo_unknown")
        parts.append(f"key_{int(row['key'])}" if pd.notna(row.get('key')) else "key_unknown")
        parts.append(f"mode_{int(row['mode'])}" if pd.notna(row.get('mode')) else "mode_unknown")
        parts.append(f"explicit_{row['explicit']}" if 'explicit' in row else "explicit_unknown")
        parts.append(f"pop_{int(row['popularity'] // 10) * 10}" if 'popularity' in row else "pop_unknown")
        return ' '.join([p for p in parts if str(p).strip()])

    df['enhanced_text'] = df.apply(enhanced, axis=1)
    return df


def load_or_build_tfidf(df: pd.DataFrame, model_path: Path) -> tuple:
    tfidf_file = model_path / 'tfidf_model.pkl'
    if tfidf_file.exists():
        with tfidf_file.open('rb') as f:
            vectorizer = pickle.load(f)
        print("TF-IDF Model Loaded")
    else:
        print("Training TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(max_features=1500, stop_words='english', min_df=3, max_df=0.65, ngram_range=(1, 2), sublinear_tf=True)
        vectorizer.fit(df['enhanced_text'].fillna(''))
        with tfidf_file.open('wb') as f:
            pickle.dump(vectorizer, f)
        print("TF-IDF Model Trained and Saved")
    return vectorizer


def load_or_build_similarity(df: pd.DataFrame, vectorizer: TfidfVectorizer, model_path: Path, max_songs: int = 5000) -> tuple:
    sim_file = model_path / 'similarity_matrix.pkl'
    if sim_file.exists():
        with sim_file.open('rb') as f:
            similarity_matrix = pickle.load(f)
        print("Similarity Matrix Loaded")
    else:
        print("Computing TF-IDF matrix and similarity matrix...")
        tfidf_matrix = vectorizer.transform(df['enhanced_text'].fillna(''))
        subset = tfidf_matrix[:max_songs]
        similarity_matrix = cosine_similarity(subset, dense_output=True)
        with sim_file.open('wb') as f:
            pickle.dump(similarity_matrix, f)
        print("Similarity Matrix Computed and Saved")
    return similarity_matrix


def synthesize_user_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nPreparing collaborative filtering data...")
    df_cf = df.copy()
    genre_cols = [c for c in df_cf.columns if c.startswith('genre_')]
    if genre_cols:
        primary_genre = df_cf[genre_cols].idxmax(axis=1).fillna('genre_unknown')
        df_cf['genre_code'] = primary_genre.astype('category').cat.codes
    else:
        df_cf['genre_code'] = 0

    df_cf['artist_code'] = df_cf['artist'].astype('category').cat.codes
    df_cf['user_id'] = ((df_cf['genre_code'] * 100) + df_cf['artist_code']) % 1000
    df_cf['user_id'] = df_cf['user_id'].astype(int)

    if 'rating' not in df_cf.columns:
        scaler = MinMaxScaler(feature_range=(1, 5))
        df_cf['rating'] = scaler.fit_transform(df_cf[['popularity']].fillna(0))

    df_cf['song_key'] = df_cf['song_name'].astype(str) + ' | ' + df_cf['artist'].astype(str)
    print(f"  - Synthetic users: {df_cf['user_id'].nunique()}")
    print(f"  - Unique songs: {df_cf['song_key'].nunique()}")
    return df_cf


def build_user_item_matrix(df_cf: pd.DataFrame, fill_value: float = None):
    pivot = df_cf.pivot_table(index='user_id', columns='song_key', values='rating', aggfunc='mean')
    if fill_value is None:
        fill_value = df_cf['rating'].mean()
    matrix = pivot.fillna(fill_value).values
    return pivot.index.tolist(), pivot.columns.tolist(), matrix, fill_value


class MatrixFactorizationModel:
    def __init__(self, user_ids, item_keys, user_factors, item_factors, global_mean):
        self.user_ids = user_ids
        self.item_keys = item_keys
        self.user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_to_index = {key: idx for idx, key in enumerate(item_keys)}
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.global_mean = global_mean

    def predict(self, user_id, item_key, verbose=False):
        if user_id in self.user_to_index and item_key in self.item_to_index:
            uidx = self.user_to_index[user_id]
            iidx = self.item_to_index[item_key]
            est = float(np.dot(self.user_factors[uidx, :], self.item_factors[iidx, :].T))
        else:
            est = float(self.global_mean)
        return SimpleNamespace(uid=user_id, iid=item_key, r_ui=None, est=est)


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "MatrixFactorizationModel":
            import hybrid_recommender_validation
            return getattr(hybrid_recommender_validation, name)
        return super().find_class(module, name)


def load_or_train_svd(df_cf: pd.DataFrame, model_path: Path):
    svd_file = model_path / 'svd_model.pkl'
    training_metrics = {}
    if svd_file.exists():
        try:
            with svd_file.open('rb') as f:
                svd_model = _LegacyUnpickler(f).load()
            print("Collaborative Model Loaded Successfully")
            return svd_model, training_metrics
        except (AttributeError, ModuleNotFoundError, pickle.UnpicklingError) as exc:
            print(f"Failed to load existing collaborative model artifact: {exc}")
            print("Retraining collaborative model from scratch.")
            svd_file.unlink(missing_ok=True)

    print("Training collaborative filtering matrix factorization model...")
    train_df, test_df = train_test_split(df_cf, test_size=0.2, random_state=42)
    user_ids, item_keys, train_matrix, fill_value = build_user_item_matrix(train_df)
    n_components = min(60, min(train_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(train_matrix)
    item_factors = svd.components_.T
    reconstructed_train = np.dot(user_factors, item_factors.T)
    train_rmse = float(np.sqrt(mean_squared_error(train_matrix.flatten(), reconstructed_train.flatten())))
    test_preds = []
    for _, row in test_df.iterrows():
        item_key = row['song_key']
        user_id = row['user_id']
        if user_id in user_ids and item_key in item_keys:
            uidx = user_ids.index(user_id)
            iidx = item_keys.index(item_key)
            pred = float(np.dot(user_factors[uidx, :], item_factors[iidx, :].T))
        else:
            pred = fill_value
        test_preds.append(pred)
    test_rmse = float(np.sqrt(mean_squared_error(test_df['rating'].values, np.array(test_preds))))
    svd_model = MatrixFactorizationModel(user_ids, item_keys, user_factors, item_factors, fill_value)
    with svd_file.open('wb') as f:
        pickle.dump(svd_model, f)
    training_metrics = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'n_users': len(user_ids),
        'n_items': len(item_keys),
        'n_ratings': len(train_df),
        'density': 100 * len(train_df) / (len(user_ids) * len(item_keys)) if len(user_ids) * len(item_keys) else 0.0
    }
    print("Collaborative Model Trained and Saved")
    return svd_model, training_metrics


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return values
    if np.allclose(values.max(), values.min()):
        return np.ones_like(values) * 0.5
    return (values - values.min()) / (values.max() - values.min())


def test_content_recommendation(df: pd.DataFrame, similarity_matrix: np.ndarray, song_title: str, top_n: int = 10) -> pd.DataFrame:
    print(f"\nRunning content-based test for song: {song_title}")
    target = df[df['song_name'].str.lower() == song_title.lower()]
    if target.empty:
        print(f"  - Unknown song: {song_title}")
        return pd.DataFrame()

    idx = target.index[0]
    scores = similarity_matrix[idx]
    similar_idx = np.argsort(scores)[::-1]
    similar_idx = [i for i in similar_idx if i != idx]
    result = df.iloc[similar_idx].copy()
    result['similarity_score'] = scores[similar_idx]
    result = result.drop_duplicates(subset=['song_key']).head(top_n)
    print(f"  - Found {len(result)} similar songs")
    return result[['song_name', 'artist', 'album_name', 'popularity', 'similarity_score']]


def test_collaborative_recommendation(df_cf: pd.DataFrame, df: pd.DataFrame, svd_model, user_id: int, song_title: str = None, top_n: int = 10) -> pd.DataFrame:
    print(f"\nRunning collaborative test for user: {user_id}")
    song_key_map = df.drop_duplicates('song_name').set_index('song_name')['song_key'].to_dict()

    if user_id not in df_cf['user_id'].unique():
        print("  - New user detected, using popularity fallback")
        item_scores = df.groupby('song_name')['popularity'].mean().sort_values(ascending=False)
        top_songs = item_scores.index[:top_n]
        result = df[df['song_name'].isin(top_songs)].drop_duplicates(subset=['song_name']).copy()
        predicted_ratings = normalize_scores(item_scores.values[:top_n])
        result['predicted_rating'] = result['song_name'].map(dict(zip(top_songs, predicted_ratings)))
        return result[['song_name', 'artist', 'album_name', 'popularity', 'predicted_rating']]

    rated_songs = df_cf[df_cf['user_id'] == user_id]['song_key'].tolist()
    rated_song_names = [key.split(' | ')[0] for key in rated_songs]
    candidates = [song for song in df['song_name'].unique() if song not in rated_song_names]
    if song_title:
        candidates = [song for song in candidates if song.lower() != song_title.lower()]
    if len(candidates) > 5000:
        candidates = candidates[:5000]

    preds = []
    for song in candidates:
        song_key = song_key_map.get(song)
        if song_key is None:
            continue
        pred = svd_model.predict(user_id, song_key)
        preds.append((song, pred.est))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    result = pd.DataFrame(preds, columns=['song_name', 'predicted_rating'])
    result = result.drop_duplicates(subset=['song_name']).head(top_n)
    result = result.merge(df[['song_name', 'artist', 'album_name', 'popularity']].drop_duplicates(['song_name', 'artist']), on='song_name', how='left')
    return result[['song_name', 'artist', 'album_name', 'popularity', 'predicted_rating']]


def test_hybrid_recommendation(df: pd.DataFrame, df_cf: pd.DataFrame, similarity_matrix: np.ndarray, svd_model, user_id: int, song_title: str, top_n: int = 10, alpha: float = 0.5, beta: float = 0.5) -> pd.DataFrame:
    print(f"\nRunning hybrid test for user={user_id}, song='{song_title}'")
    if abs(alpha + beta - 1.0) > 1e-6:
        alpha, beta = alpha / (alpha + beta), beta / (alpha + beta)

    # Normalize apostrophes for matching (handle curly and straight quotes)
    def normalize_title(title):
        return title.replace('\u2019', "'").replace('\u2018', "'").replace('\u201b', "'").lower()
    
    song_title_normalized = normalize_title(song_title)
    song_exists = False
    song_idx = None
    
    for idx, sn in enumerate(df['song_name']):
        if normalize_title(str(sn)) == song_title_normalized:
            song_exists = True
            song_idx = idx
            break
    
    user_exists = user_id in df_cf['user_id'].unique()

    content_scores = np.zeros(len(df))
    if song_exists:
        content_scores = similarity_matrix[song_idx]
    else:
        print(f"  - Song '{song_title}' not found in dataset")

    song_key_map = df.drop_duplicates('song_name').set_index('song_name')['song_key'].to_dict()
    if not user_exists:
        print("  - Unknown user, using popularity for collaborative fallback")
        collab_scores = normalize_scores(df['popularity'].values)
    else:
        collab_scores = []
        candidates = df['song_name'].tolist()
        if len(candidates) > 5000:
            candidates = candidates[:5000]
        for song in candidates:
            song_key = song_key_map.get(song)
            if song_key is None:
                collab_scores.append(0.5)
                continue
            pred = svd_model.predict(user_id, song_key)
            collab_scores.append(pred.est)
        collab_scores = normalize_scores(np.array(collab_scores))

    content_norm = normalize_scores(content_scores)
    collab_norm = normalize_scores(collab_scores)

    epsilon = 1e-6
    hybrid_linear = alpha * content_norm + beta * collab_norm
    hybrid_geo = (content_norm + epsilon) ** alpha * (collab_norm + epsilon) ** beta
    hybrid_scores = normalize_scores(0.75 * hybrid_linear + 0.25 * hybrid_geo)
    ranking = np.argsort(hybrid_scores)[::-1]

    if song_exists:
        input_idx = song_idx
        ranking = [i for i in ranking if i != input_idx]

    results = df.iloc[ranking[:top_n * 3]].copy()
    results['content_score'] = content_norm[ranking[:top_n * 3]]
    results['collab_score'] = collab_scores[ranking[:top_n * 3]]
    results['hybrid_score'] = hybrid_scores[ranking[:top_n * 3]]
    results = results.drop_duplicates(subset=['song_key']).head(top_n)
    if results.empty:
        print("  - Hybrid ranking empty, falling back to popular songs")
        results = df.sort_values('popularity', ascending=False).head(top_n).copy()
        results['content_score'] = 0.0
        results['collab_score'] = 0.0
        results['hybrid_score'] = normalize_scores(results['popularity'].values)
        return results[['song_name', 'artist', 'album_name', 'popularity', 'content_score', 'collab_score', 'hybrid_score']]

    return results[['song_name', 'artist', 'album_name', 'popularity', 'content_score', 'collab_score', 'hybrid_score']]


def precision_at_k(predictions, k=10, threshold=3.5):
    user_ratings = {}
    for uid, iid, true_r, est, _ in predictions:
        user_ratings.setdefault(uid, []).append((est, true_r))
    precisions = []
    for ratings in user_ratings.values():
        ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = ratings[:k]
        relevant = sum(1 for _, true_r in top_k if true_r >= threshold)
        precisions.append(relevant / k)
    return float(np.mean(precisions)) if precisions else 0.0


def recall_at_k(predictions, k=10, threshold=3.5):
    user_ratings = {}
    for uid, iid, true_r, est, _ in predictions:
        user_ratings.setdefault(uid, []).append((est, true_r))
    recalls = []
    for ratings in user_ratings.values():
        ratings.sort(key=lambda x: x[0], reverse=True)
        total_relevant = sum(1 for _, true_r in ratings if true_r >= threshold)
        if total_relevant == 0:
            continue
        relevant = sum(1 for _, true_r in ratings[:k] if true_r >= threshold)
        recalls.append(relevant / total_relevant)
    return float(np.mean(recalls)) if recalls else 0.0


def f1_score_at_k(predictions, k=10, threshold=3.5):
    p = precision_at_k(predictions, k=k, threshold=threshold)
    r = recall_at_k(predictions, k=k, threshold=threshold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_recommender(test_df: pd.DataFrame, svd_model, top_k=10):
    print("\nEvaluating collaborative filtering performance...")
    y_true = []
    y_pred = []
    predictions = []
    for _, row in test_df.iterrows():
        pred = svd_model.predict(row['user_id'], row['song_key'])
        y_true.append(row['rating'])
        y_pred.append(pred.est)
        predictions.append((row['user_id'], row['song_key'], row['rating'], pred.est, None))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    precision = precision_at_k(predictions, k=top_k)
    recall = recall_at_k(predictions, k=top_k)
    f1 = f1_score_at_k(predictions, k=top_k)
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - Precision@{top_k}: {precision:.4f}")
    print(f"  - Recall@{top_k}: {recall:.4f}")
    print(f"  - F1@{top_k}: {f1:.4f}")
    return {
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions
    }


def recommendation_quality(results: pd.DataFrame, df: pd.DataFrame) -> dict:
    unique_songs = results['song_name'].nunique()
    unique_artists = results['artist'].nunique()
    avg_popularity = results['popularity'].mean() if 'popularity' in results else np.nan
    popularity_norm = normalize_scores(np.array(results['popularity'].fillna(0))) if 'popularity' in results else np.zeros(len(results))
    novelty_score = 1.0 - float(np.mean(popularity_norm)) if len(popularity_norm) else 0.0
    duplicates = len(results) - unique_songs
    return {
        'unique_songs': unique_songs,
        'unique_artists': unique_artists,
        'duplicate_recommendations': duplicates,
        'avg_popularity': avg_popularity,
        'novelty_score': novelty_score
    }


def measure_performance(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def save_artifacts(model_path: Path, vectorizer, similarity_matrix, svd_model, hybrid_model):
    print("\nSaving deployment artifacts...")
    with (model_path / 'tfidf_model.pkl').open('wb') as f:
        pickle.dump(vectorizer, f)
    with (model_path / 'similarity_matrix.pkl').open('wb') as f:
        pickle.dump(similarity_matrix, f)
    with (model_path / 'svd_model.pkl').open('wb') as f:
        pickle.dump(svd_model, f)
    with (model_path / 'hybrid_model.pkl').open('wb') as f:
        pickle.dump(hybrid_model, f)
    print("Models successfully saved for deployment")


class HybridRecommender:
    def __init__(self, df, df_cf, vectorizer, similarity_matrix, svd_model, alpha=0.5, beta=0.5):
        self.df = df.reset_index(drop=True)
        self.df_cf = df_cf
        self.vectorizer = vectorizer
        self.similarity_matrix = similarity_matrix
        self.svd_model = svd_model
        self.alpha = alpha
        self.beta = beta

    def content_recommend(self, song_title, top_n=10):
        return test_content_recommendation(self.df, self.similarity_matrix, song_title, top_n)

    def collaborative_recommend(self, user_id, song_title=None, top_n=10):
        return test_collaborative_recommendation(self.df_cf, self.df, self.svd_model, user_id, song_title, top_n)

    def hybrid_recommend(self, user_id, song_title, top_n=10):
        return test_hybrid_recommendation(self.df, self.df_cf, self.similarity_matrix, self.svd_model, user_id, song_title, top_n, alpha=self.alpha, beta=self.beta)


if __name__ == '__main__':
    df = load_dataset(DATA_PATH)
    df = create_text_features(df)
    df['song_key'] = df['song_name'].astype(str) + ' | ' + df['artist'].astype(str)
    vectorizer = load_or_build_tfidf(df, MODEL_DIR)
    similarity_matrix = load_or_build_similarity(df, vectorizer, MODEL_DIR, max_songs=5000)
    df_content = df.head(similarity_matrix.shape[0]).reset_index(drop=True)
    df_cf = synthesize_user_data(df)
    svd_model, training_metrics = load_or_train_svd(df_cf, MODEL_DIR)

    if training_metrics:
        print("\nCollaborative training metrics:")
        for key, value in training_metrics.items():
            print(f"  - {key}: {value}")

    # Integrity checks
    print("\nValidating model integrity...")
    square = similarity_matrix.ndim == 2 and similarity_matrix.shape[0] == similarity_matrix.shape[1]
    print(f"  - Similarity matrix square: {square}")
    print(f"  - Similarity matrix shape: {similarity_matrix.shape}")
    print(f"  - TF-IDF feature count: {len(vectorizer.get_feature_names_out())}")
    song_index_map = {name.lower(): idx for idx, name in enumerate(df_content['song_name'].values)}
    missing_songs = [s for s in ['Unknown Song Title'] if s.lower() not in song_index_map]
    print(f"  - Song index alignment check: {len(song_index_map)} songs mapped")
    print(f"  - Missing test entries: {len(missing_songs)}")

    # Recommendation functions and scenario tests
    recommender = HybridRecommender(df_content, df_cf, vectorizer, similarity_matrix, svd_model)
    test_songs = list(df_content['song_name'].dropna().unique())
    demo_song = test_songs[0]
    demo_user = int(df_cf['user_id'].iloc[0])
    new_user = max(df_cf['user_id'].unique()) + 1
    random_song = df['song_name'].sample(1, random_state=42).iloc[0]
    unknown_song = "Nonexistent Song Title"

    print("\nSTEP 4 - Recommendation test scenarios")
    print("1) Existing user, existing song")
    cb_res, tcb = measure_performance(recommender.content_recommend, demo_song, 5)
    print(f"  - Content-based returned {len(cb_res)} rows in {tcb:.4f}s")
    collab_res, tc = measure_performance(recommender.collaborative_recommend, demo_user, song_title=None, top_n=5)
    print(f"  - Collaborative returned {len(collab_res)} rows in {tc:.4f}s")
    hybrid_res, th = measure_performance(recommender.hybrid_recommend, demo_user, demo_song, 5)
    print(f"  - Hybrid returned {len(hybrid_res)} rows in {th:.4f}s")

    print("\n2) New user cold start")
    cold_res, tcold = measure_performance(recommender.hybrid_recommend, new_user, demo_song, 5)
    print(f"  - New user hybrid returned {len(cold_res)} rows in {tcold:.4f}s")

    print("\n3) Random song input")
    rand_res, trand = measure_performance(recommender.content_recommend, random_song, 5)
    print(f"  - Random song content-based returned {len(rand_res)} rows in {trand:.4f}s")

    print("\n4) Unknown song edge case")
    unknown_res, tunk = measure_performance(recommender.hybrid_recommend, demo_user, unknown_song, 5)
    print(f"  - Unknown song hybrid returned {len(unknown_res)} rows in {tunk:.4f}s")

    # Evaluation metrics
    print("\nSTEP 5 - Evaluation metrics")
    train_df, test_df = train_test_split(df_cf, test_size=0.2, random_state=42)
    eval_metrics = evaluate_recommender(test_df, svd_model, top_k=10)

    # Quality analysis
    print("\nSTEP 6 - Recommendation quality analysis")
    diversity = recommendation_quality(hybrid_res, df)
    print(f"  - Unique songs: {diversity['unique_songs']}")
    print(f"  - Unique artists: {diversity['unique_artists']}")
    print(f"  - Duplicate recommendations: {diversity['duplicate_recommendations']}")
    print(f"  - Novelty score: {diversity['novelty_score']:.4f}")

    # Performance check
    print("\nSTEP 7 - Performance optimization check")
    avg_response_time = np.mean([tcb, tc, th, tcold, trand, tunk])
    print(f"  - Average response time per request: {avg_response_time:.4f} seconds")
    mem_mb = similarity_matrix.nbytes / (1024 ** 2)
    print(f"  - Similarity matrix memory usage: {mem_mb:.2f} MB")

    # Visualization
    print("\nSTEP 8 - Generating performance visualizations")
    ks = [5, 10, 15, 20]
    precisions = [precision_at_k(eval_metrics['predictions'], k=k) for k in ks]
    recalls = [recall_at_k(eval_metrics['predictions'], k=k) for k in ks]
    plt.figure(figsize=(8, 5))
    plt.plot(ks, precisions, marker='o', label='Precision@K')
    plt.plot(ks, recalls, marker='s', label='Recall@K')
    plt.xlabel('K')
    plt.ylabel('Score')
    plt.title('Precision vs Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / 'precision_recall_curve.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(['RMSE', 'F1'], [eval_metrics['rmse'], eval_metrics['f1']], color=['#2E86AB', '#45B7D1'])
    plt.title('RMSE and F1 Score')
    plt.savefig(PLOT_DIR / 'rmse_f1_bar.png')
    plt.close()

    freq = hybrid_res['artist'].value_counts().head(10)
    plt.figure(figsize=(8, 4))
    freq.plot(kind='bar', color='mediumseagreen')
    plt.title('Recommendation Frequency by Artist')
    plt.ylabel('Count')
    plt.savefig(PLOT_DIR / 'recommendation_frequency.png')
    plt.close()
    print(f"  - Visualizations saved to {PLOT_DIR}")

    # Save artifacts for deployment
    save_artifacts(MODEL_DIR, vectorizer, similarity_matrix, svd_model, recommender)

    # Demo mode output
    print("\nSTEP 10 - Demonstration mode output")
    print(f"User Input:\n  User ID: {demo_user}\n  Song: {demo_song}\n")
    print("System Output:")
    demo_recs = recommender.hybrid_recommend(demo_user, demo_song, top_n=10)
    for i, row in demo_recs.iterrows():
        print(f"  {i+1}. {row['song_name']} — Score: {row['hybrid_score']:.4f}")

    print("\nSTEP 11 - Final system validation")
    if not hybrid_res.empty and not unknown_res.empty:
        print("Hybrid Recommendation Model Successfully Validated")
    else:
        print("Warning: Some recommendation outputs are empty. Please investigate.")

    print("\nSTEP 12 - Demo script sample")
    print("demo_user_id = {}".format(demo_user))
    print("demo_song = '{}'".format(demo_song))
    print("# Run: recommender.hybrid_recommend(demo_user_id, demo_song)")

    print("\nEvaluation Summary:")
    print(f"Precision@10 = {eval_metrics['precision']:.4f}")
    print(f"Recall@10 = {eval_metrics['recall']:.4f}")
    print(f"F1 Score = {eval_metrics['f1']:.4f}")
    print(f"RMSE = {eval_metrics['rmse']:.4f}")
    print(f"Average Response Time = {avg_response_time:.4f} seconds")
