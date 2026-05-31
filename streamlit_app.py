import os
import json
import hashlib
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
from io import BytesIO
from email.message import EmailMessage
from email.utils import make_msgid
import smtplib
from PIL import Image
import time
import base64
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, Boolean, select
from sqlalchemy.exc import SQLAlchemyError
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
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    pg_user = os.getenv("POSTGRES_USER", "").strip()
    pg_pass = os.getenv("POSTGRES_PASSWORD", "").strip()
    pg_host = os.getenv("POSTGRES_HOST", "localhost").strip()
    pg_port = os.getenv("POSTGRES_PORT", "5432").strip()
    pg_db = os.getenv("POSTGRES_DB", "music_recommender").strip()
    if pg_user and pg_pass:
        DATABASE_URL = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"

DB_ENGINE = None
DB_METADATA = MetaData()
users_table = Table(
    "users",
    DB_METADATA,
    Column("email", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("password", String, nullable=False),
    Column("is_admin", Boolean, default=False, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow),
)

if DATABASE_URL:
    try:
        DB_ENGINE = create_engine(DATABASE_URL, echo=False, future=True)
    except SQLAlchemyError:
        DB_ENGINE = None

# --- Embedded auth helpers (restored from auth.py) ---
USERS_FILE = BASE_DIR / "users_db.json"
USER_PROFILE_DIR = BASE_DIR / "user_profiles"
DEFAULT_AVATAR_PATH = BASE_DIR / "Gemini_Generated_Image_i0gnofi0gnofi0gn.png"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def _profile_hash(email: str) -> str:
    return hashlib.sha256(email.lower().encode("utf-8")).hexdigest()


def _profile_meta_path(email: str) -> Path:
    return USER_PROFILE_DIR / f"{_profile_hash(email)}.json"


def _profile_avatar_path(email: str) -> Path:
    return USER_PROFILE_DIR / f"{_profile_hash(email)}.png"


def ensure_profile_dir() -> None:
    USER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def load_user_profile(email: str) -> dict:
    ensure_profile_dir()
    profile = {"memo": "", "avatar": ""}
    meta_path = _profile_meta_path(email)
    if meta_path.exists():
        try:
            profile.update(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return profile


def save_user_profile(email: str, memo: str | None = None, avatar_bytes: bytes | None = None) -> None:
    ensure_profile_dir()
    profile = load_user_profile(email)
    if memo is not None:
        profile["memo"] = memo
    if avatar_bytes is not None:
        avatar_path = _profile_avatar_path(email)
        try:
            with avatar_path.open("wb") as f:
                f.write(avatar_bytes)
            profile["avatar"] = str(avatar_path.name)
        except Exception:
            pass
    profile["updated_at"] = datetime.utcnow().isoformat()
    meta_path = _profile_meta_path(email)
    try:
        meta_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_avatar_data_uri(email: str | None = None) -> str:
    avatar_path = None
    if email:
        candidate = _profile_avatar_path(email)
        if candidate.exists():
            avatar_path = candidate
    if avatar_path is None and DEFAULT_AVATAR_PATH.exists():
        avatar_path = DEFAULT_AVATAR_PATH
    if not avatar_path or not avatar_path.exists():
        return ""
    try:
        with Image.open(avatar_path) as img:
            img = img.convert("RGBA")
            img = img.resize((120, 120), Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def get_query_params() -> dict:
    if hasattr(st, "query_params"):
        return dict(st.query_params)
    if hasattr(st, "experimental_get_query_params"):
        return st.experimental_get_query_params()
    return {}


def set_query_params(params: dict) -> None:
    if hasattr(st, "query_params"):
        st.query_params = params
    elif hasattr(st, "experimental_set_query_params"):
        st.experimental_set_query_params(**params)


def init_user_store() -> None:
    if DB_ENGINE is not None:
        try:
            DB_METADATA.create_all(DB_ENGINE)
        except SQLAlchemyError:
            pass
        return

    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({"users": {}}, indent=2))


def load_users() -> dict:
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.connect() as conn:
                result = conn.execute(select(users_table)).mappings().all()
                return {
                    row["email"]: {"name": row["name"], "password": row["password"]}
                    for row in result
                }
        except SQLAlchemyError:
            return {}

    try:
        init_user_store()
        with USERS_FILE.open('r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('users', {})
    except Exception:
        return {}


def save_users(users: dict) -> None:
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.begin() as conn:
                for email_key, user_data in users.items():
                    conn.execute(
                        users_table.insert().values(
                            email=email_key,
                            name=user_data.get('name', email_key),
                            password=user_data.get('password', ''),
                            created_at=datetime.utcnow(),
                        )
                    )
        except SQLAlchemyError:
            pass
        return

    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USERS_FILE.open('w', encoding='utf-8') as f:
        json.dump({"users": users}, f, indent=2)


def authenticate_user(email: str, password: str) -> tuple:
    email_key = email.lower()
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.connect() as conn:
                result = conn.execute(
                    select(users_table).where(users_table.c.email == email_key)
                ).mappings().first()
                if not result:
                    return False, "No account found for that email."
                if result["password"] != hash_password(password):
                    return False, "Incorrect password."
                return True, result["name"] or email
        except SQLAlchemyError:
            pass

    users = load_users()
    stored = users.get(email_key)
    if not stored:
        return False, "No account found for that email."
    if stored.get('password') != hash_password(password):
        return False, "Incorrect password."
    return True, stored.get('name', email)


def user_exists(email: str) -> bool:
    email_key = email.lower()
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.connect() as conn:
                return conn.execute(
                    select(users_table.c.email).where(users_table.c.email == email_key)
                ).first() is not None
        except SQLAlchemyError:
            pass

    users = load_users()
    return email_key in users


def register_user(name: str, email: str, password: str) -> tuple:
    email_key = email.lower()
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.connect() as conn:
                result = conn.execute(
                    select(users_table).where(users_table.c.email == email_key)
                ).first()
                if result:
                    return False, "An account using that email already exists."
                conn.execute(
                    users_table.insert().values(
                        email=email_key,
                        name=name.strip() or email_key,
                        password=hash_password(password),
                        created_at=datetime.utcnow(),
                    )
                )
                conn.commit()
                return True, "Account created successfully."
        except SQLAlchemyError as exc:
            return False, f"Failed to register account: {exc}"

    users = load_users()
    if email_key in users:
        return False, "An account using that email already exists."
    users[email_key] = {'name': name.strip() or email_key, 'password': hash_password(password)}
    save_users(users)
    return True, "Account created successfully."


def update_user_name(email: str, new_name: str) -> tuple[bool, str]:
    email_key = email.lower()
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.begin() as conn:
                result = conn.execute(
                    select(users_table).where(users_table.c.email == email_key)
                ).first()
                if not result:
                    return False, "No account found for that email."
                conn.execute(
                    users_table.update()
                    .where(users_table.c.email == email_key)
                    .values(name=new_name.strip() or email_key)
                )
            return True, "Name updated successfully."
        except SQLAlchemyError as exc:
            return False, f"Failed to update name: {exc}"

    users = load_users()
    if email_key not in users:
        return False, "No account found for that email."
    users[email_key]['name'] = new_name.strip() or email_key
    save_users(users)
    return True, "Name updated successfully."


def check_if_admin(email: str) -> bool:
    email_key = email.lower()
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.connect() as conn:
                result = conn.execute(
                    select(users_table.c.is_admin).where(users_table.c.email == email_key)
                ).scalar()
                return bool(result) if result is not None else False
        except SQLAlchemyError:
            pass
    users = load_users()
    user_rec = users.get(email_key, {})
    return bool(user_rec.get('is_admin', False))


def set_admin_status(email: str, is_admin: bool) -> tuple[bool, str]:
    email_key = email.lower()
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.begin() as conn:
                result = conn.execute(
                    select(users_table).where(users_table.c.email == email_key)
                ).first()
                if not result:
                    return False, "No account found for that email."
                conn.execute(
                    users_table.update()
                    .where(users_table.c.email == email_key)
                    .values(is_admin=is_admin)
                )
            return True, f"Admin status updated for {email_key}."
        except SQLAlchemyError as exc:
            return False, f"Failed to update admin status: {exc}"
    
    users = load_users()
    if email_key not in users:
        return False, "No account found for that email."
    users[email_key]['is_admin'] = is_admin
    save_users(users)
    return True, f"Admin status updated for {email_key}."


def delete_user(email: str) -> tuple[bool, str]:
    email_key = email.lower()
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.begin() as conn:
                result = conn.execute(
                    select(users_table).where(users_table.c.email == email_key)
                ).first()
                if not result:
                    return False, "User not found."
                conn.execute(
                    users_table.delete().where(users_table.c.email == email_key)
                )
            return True, f"User {email_key} deleted."
        except SQLAlchemyError as exc:
            return False, f"Failed to delete user: {exc}"
    
    users = load_users()
    if email_key not in users:
        return False, "User not found."
    users.pop(email_key)
    save_users(users)
    return True, f"User {email_key} deleted."


def get_all_users_list() -> list:
    if DB_ENGINE is not None:
        try:
            with DB_ENGINE.connect() as conn:
                result = conn.execute(select(users_table)).mappings().all()
                return [{"email": row["email"], "name": row["name"], "is_admin": bool(row["is_admin"])} for row in result]
        except SQLAlchemyError:
            pass
    users = load_users()
    return [{"email": k, "name": v.get("name", k), "is_admin": bool(v.get("is_admin", False))} for k, v in users.items()]


def validate_email(email: str) -> bool:
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email.strip()) is not None


def get_query_params() -> dict:
    if hasattr(st, "query_params"):
        return dict(st.query_params)
    if hasattr(st, "experimental_get_query_params"):
        return st.experimental_get_query_params()
    return {}


def set_query_params(params: dict) -> None:
    if hasattr(st, "query_params"):
        st.query_params = params
    elif hasattr(st, "experimental_set_query_params"):
        st.experimental_set_query_params(**params)


OTP_EXPIRY_SECONDS = 300


def generate_otp_code() -> str:
    return str(np.random.randint(0, 1000000)).zfill(6)


def clear_otp_state() -> None:
    for key in [
        'otp_pending',
        'otp_code',
        'otp_expires',
        'pending_signup_name',
        'pending_signup_email',
        'pending_signup_password',
        'pending_signup_confirm',
        'signup_otp_input',
        'otp_message',
    ]:
        if key in st.session_state:
            del st.session_state[key]


def is_otp_valid(code: str) -> bool:
    if not st.session_state.get('otp_code'):
        return False
    if time.time() > st.session_state.get('otp_expires', 0):
        return False
    return str(code).strip() == st.session_state.get('otp_code')


def send_otp_notification(email: str, otp: str) -> tuple[bool, str]:
    smtp_server = os.getenv("SMTP_SERVER", "")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USERNAME", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    from_address = os.getenv("EMAIL_FROM", smtp_user)

    if not smtp_server or not smtp_user or not smtp_pass or not from_address:
        return False, "SMTP settings are not configured. Set SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, and EMAIL_FROM."

    sender_name = "Music Recommendation System"
    message = EmailMessage()
    message["Subject"] = "Your OTP code for Music Recommendation System"
    message["From"] = f"{sender_name} <{from_address}>"
    message["To"] = email

    logo_path = BASE_DIR / "wangdalockedin.jpeg"
    logo_cid = None
    logo_bytes = None
    if logo_path.exists():
        logo_bytes = logo_path.read_bytes()
        logo_cid = make_msgid(domain="example.com")[1:-1]

    plain_text = (
        f"Your one-time verification code is {otp}.\n\n"
        "It will expire in 5 minutes. Do not share it with anyone.\n\n"
        "If you did not request this, please ignore this message.\n"
    )
    message.set_content(plain_text)

    img_tag = f'<img src="cid:{logo_cid}" alt="Music Recommendation System" width="120" style="display:block;margin:0 auto 16px;" />' if logo_cid else ''

    html_content = f"""
    <html>
      <body style="font-family:Arial,Helvetica,sans-serif;background:#f4f6fb;padding:0;margin:0;">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;margin:0 auto;background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 20px 40px rgba(0,0,0,0.08);">
          <tr style="background:#0d9488;color:#ffffff;">
            <td style="padding:24px;text-align:center;">
              {img_tag}
              <h1 style="margin:0;font-size:24px;letter-spacing:0.01em;">Music Recommendation System</h1>
              <p style="margin:8px 0 0;font-size:14px;color:#d1faf4;">Your secure one-time login code</p>
            </td>
          </tr>
          <tr>
            <td style="padding:32px;">
              <h2 style="margin:0 0 16px;color:#111827;font-size:20px;">Hello,</h2>
              <p style="margin:0 0 24px;color:#374151;font-size:15px;line-height:1.7;">Use the code below to verify your account and finish the signup process.</p>
              <div style="text-align:center;margin:0 auto 24px;padding:22px 0;background:#ecfdf5;border:1px solid #a7f3d0;border-radius:14px;max-width:240px;font-size:32px;letter-spacing:4px;font-weight:700;color:#065f46;">
                {otp}
              </div>
              <p style="margin:0 0 12px;color:#4b5563;font-size:14px;">This code expires in <strong>5 minutes</strong>.</p>
              <p style="margin:0 0 24px;color:#4b5563;font-size:14px;">If you did not request this code, you can safely ignore this message.</p>
              <p style="margin:0;color:#6b7280;font-size:12px;">Music Recommendation System • Sent from {from_address}</p>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    message.add_alternative(html_content, subtype="html")
    if logo_cid and logo_bytes is not None:
        message.get_payload()[1].add_related(
            logo_bytes,
            maintype="image",
            subtype="jpeg",
            cid=f"<{logo_cid}>",
            filename="wangdalockedin.jpeg",
        )

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15) as smtp:
                smtp.login(smtp_user, smtp_pass)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(smtp_user, smtp_pass)
                smtp.send_message(message)
        return True, "OTP sent successfully. Check your email inbox."
    except Exception as exc:
        return False, f"Failed to send OTP email: {exc}"


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
                            is_admin = check_if_admin(email.lower())
                            st.session_state.authenticated = True
                            st.session_state.current_user = message
                            st.session_state.current_email = email.lower()
                            st.session_state.auth_loading = True
                            # Admin users go to Admin console; ensure admins bypass onboarding
                            if is_admin:
                                st.session_state.current_page = "Admin"
                                st.session_state.onboarding_complete = True
                                st.success(f"Welcome back, admin {message}! Redirecting to Admin console...")
                                set_query_params({"page": ["Admin"]})
                                time.sleep(0.6)
                                st.rerun()
                            # Regular users land on Home and go through onboarding
                            st.session_state.current_page = "Home"
                            st.session_state.onboarding_complete = False
                            st.session_state.onboarding_data = {}
                            st.success(f"Welcome back, {message}! Preparing your music feed...")
                            st.info("Loading your dashboard now... just a moment.")
                            time.sleep(0.8)
                            set_query_params({})
                            st.rerun()
                        else:
                            st.error(message)

        with tabs[1]:
            st.session_state.auth_mode = "Sign up"
            if not st.session_state.otp_pending:
                with st.form("auth_signup_form"):
                    name = st.text_input("Full name", key="signup_name")
                    email = st.text_input("Email address", key="signup_email")
                    password = st.text_input("Password", type="password", key="signup_password")
                    confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                    st.caption("Use at least 8 characters with letters, numbers, and a symbol for the best security.")
                    send_otp = st.form_submit_button("Send OTP")
                    if send_otp:
                        if not name or not email or not password or not confirm:
                            st.error("All fields are required to create an account.")
                        elif not validate_email(email):
                            st.error("Enter a valid email address.")
                        elif password != confirm:
                            st.error("Passwords do not match.")
                        elif len(password) < 8:
                            st.error("Password should be at least 8 characters.")
                        elif user_exists(email):
                            st.error("An account using that email already exists. Please log in or use a different email.")
                        else:
                            otp = generate_otp_code()
                            success, message = send_otp_notification(email, otp)
                            if not success:
                                st.error(message)
                            else:
                                st.session_state.otp_pending = True
                                st.session_state.otp_code = otp
                                st.session_state.otp_expires = time.time() + OTP_EXPIRY_SECONDS
                                st.session_state.pending_signup_name = name
                                st.session_state.pending_signup_email = email
                                st.session_state.pending_signup_password = password
                                st.session_state.pending_signup_confirm = confirm
                                st.session_state.otp_message = f"OTP sent to {email}. It expires in 5 minutes."
                                st.success(st.session_state.otp_message)
                                st.rerun()
            else:
                with st.form("auth_signup_verify_form"):
                    st.markdown("#### Verify your account with OTP")
                    st.markdown(f"<p style='color:#cbd5e1;'>We sent a one-time password to <strong>{st.session_state.pending_signup_email}</strong>. Enter it below to complete registration.</p>", unsafe_allow_html=True)
                    otp_input = st.text_input("One-time password", key="signup_otp_input")
                    verify = st.form_submit_button("Verify OTP")
                    if verify:
                        if not otp_input:
                            st.error("Please enter the OTP to continue.")
                        elif time.time() > st.session_state.otp_expires:
                            st.error("OTP expired. Please restart the signup process.")
                            clear_otp_state()
                        elif not is_otp_valid(otp_input):
                            st.error("The OTP is incorrect. Please try again.")
                        else:
                            name = st.session_state.pending_signup_name
                            email = st.session_state.pending_signup_email
                            password = st.session_state.pending_signup_password
                            success, message = register_user(name, email, password)
                            clear_otp_state()
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


def show_onboarding_page():
    st.markdown(
        """
        <style>
            .onboard-shell {
                background: linear-gradient(135deg, rgba(29,185,84,0.18), rgba(58,123,213,0.18));
                border-radius: 2rem;
                padding: 2rem;
                box-shadow: 0 32px 90px rgba(10, 25, 47, 0.18);
                color: #e2e8f0;
            }
            .onboard-header h1 { margin:0; font-size:3rem; line-height:1.05; }
            .onboard-header p { color:#cbd5e1; font-size:1.1rem; margin-top:0.8rem; }
            .onboard-card { background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 1.5rem; padding: 1.75rem; margin-bottom: 1.5rem; }
            .onboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
            .onboard-note { background: rgba(30, 41, 59, 0.75); border-radius: 1.25rem; padding: 1.25rem; border: 1px solid rgba(255,255,255,0.08); }
            .onboard-note h3 { margin-top:0; color:#f8fafc; }
            .onboard-note p { margin-bottom:0.8rem; color:#cbd5e1; }
            .onboard-button { background: #10b981 !important; color: #fff !important; border-radius: 999px !important; padding: 1rem 1.4rem !important; font-weight: 700 !important; }
            .onboard-button:hover { background: #14b8a6 !important; }
            @media (max-width: 900px) { .onboard-grid { grid-template-columns: 1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="onboard-shell">', unsafe_allow_html=True)
    st.markdown('<div class="onboard-header">', unsafe_allow_html=True)
    st.markdown('<h1>Tell us your vibe.</h1>', unsafe_allow_html=True)
    st.markdown('<p>We create a richer recommendation experience when we know your mood, genre, and the music attributes you love.</p>', unsafe_allow_html=True)
    st.markdown('</div>')

    with st.form('onboarding_form'):
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin-bottom:1rem;">'
            '<span style="background:#0f172a;color:#a5f3fc;padding:0.65rem 1rem;border-radius:999px;box-shadow:0 10px 30px rgba(16,185,129,0.14);">🎧 Personalize my vibe</span>'
            '<span style="background:#0f172a;color:#fde68a;padding:0.65rem 1rem;border-radius:999px;box-shadow:0 10px 30px rgba(56,189,248,0.12);">✨ Quick setup</span>'
            '<span style="background:#0f172a;color:#fbcfe8;padding:0.65rem 1rem;border-radius:999px;box-shadow:0 10px 30px rgba(234,179,8,0.12);">💡 Spotify-style feel</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="onboard-grid">', unsafe_allow_html=True)
        with st.container():
            mood = st.selectbox(
                'How are you feeling right now?',
                ['🧘 Chill & mellow', '⚡ Energetic & upbeat', '❤️ Romantic & warm', '🌧️ Moody & reflective', '🎯 Focused & motivated'],
                key='onboard_mood',
            )
            
            df_temp = load_data()
            available_genres = sorted([g for g in df_temp['genre'].dropna().unique() if isinstance(g, str)])
            genre_display = [f"🎵 {g.title()}" for g in available_genres]
            
            genre = st.selectbox(
                'Choose your genre vibe',
                genre_display,
                key='onboard_genre',
            )
            tempo = st.slider('Preferred tempo (BPM)', 60, 180, 112, step=5, key='onboard_tempo')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="onboard-grid">', unsafe_allow_html=True)
        with st.container():
            era = st.radio(
                'Which era sounds best to you?',
                ['Today ✨', '90s/00s 🔥', 'Classics 🎼', 'Underground 🎙️'],
                index=0,
                horizontal=True,
                key='onboard_era',
            )
            focus = st.selectbox(
                'Your music mission',
                ['Discover fresh tracks ✨', 'Curate a mood playlist 🎶', 'Hear crowd favorites 🔥', 'Find hidden gems 💎'],
                key='onboard_focus',
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="onboard-note">', unsafe_allow_html=True)
        st.markdown('<h3>Why this matters</h3>', unsafe_allow_html=True)
        st.markdown('<p>These preferences let the recommender start your session with a stronger sense of vibe. You can still refine the experience inside the app at any time.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        submit = st.form_submit_button("Let\'s go listen")
        if submit:
            genre_clean = genre.replace("🎵 ", "").lower()
            st.session_state.onboarding_data = {
                'Mood': mood,
                'Genre': genre_clean,
                'Tempo': tempo,
                'Era': era,
                'Focus': focus,
            }
            st.session_state.onboarding_complete = True
            st.success('Your music profile is ready. Redirecting to your recommendation studio...')
            st.caption('You can still update your listening preferences inside the app when you want to explore a different vibe.')
            time.sleep(0.9)
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def show_profile_page():
    user_email = st.session_state.current_email or "Not available"
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:1rem;'>", unsafe_allow_html=True)
    if st.button("← Back to studio", key="profile_back_top"):
        st.session_state.current_page = "Home"
        set_query_params({})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='padding:1.5rem;border-radius:1.8rem;background:linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.96));box-shadow:0 28px 65px rgba(0,0,0,0.22);'>", unsafe_allow_html=True)
    # top avatar above greeting
    avatar_src = get_avatar_data_uri(user_email)
    st.markdown(f"<div style='display:flex;justify-content:center;margin-bottom:0.6rem;'><img src=\"{avatar_src}\" style='width:120px;height:120px;border-radius:50%;object-fit:cover;border:3px solid rgba(255,255,255,0.06);' alt='avatar'/></div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin:0;color:#ffffff;font-size:2.4rem;text-align:center;'>Hello, {st.session_state.current_user}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='margin:.75rem 0 0;color:#cbd5e1;font-size:1rem;'>Your music profile puts your listening preferences, avatar, and notes in one refined place.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    profile = load_user_profile(user_email) if user_email != "Not available" else {"memo": "", "avatar": ""}
    profile_memo = profile.get("memo", "")
    updated_at = profile.get("updated_at", "Never")
    # preferences may come from onboarding stored in session state
    user_prefs = st.session_state.get("onboarding_data", {})

    st.markdown("<div style='margin-top:1rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding:1.2rem;border-radius:1.5rem;background:rgba(15,23,42,0.95);border:1px solid rgba(59,130,246,0.18);'><p style='margin:0;color:#94a3b8;'>Signed in as</p><h3 style='margin:.5rem 0 0;color:#ffffff;font-size:1.1rem;'>{user_email}</h3></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding:1.2rem;border-radius:1.5rem;background:rgba(15,23,42,0.95);border:1px solid rgba(16,185,129,0.18);'><p style='margin:0;color:#94a3b8;'>Display name</p><h3 style='margin:.5rem 0 0;color:#ffffff;font-size:1.1rem;'>{st.session_state.current_user}</h3></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding:1.2rem;border-radius:1.5rem;background:rgba(15,23,42,0.95);border:1px solid rgba(244,63,94,0.18);'><p style='margin:0;color:#94a3b8;'>Last updated</p><h3 style='margin:.5rem 0 0;color:#ffffff;font-size:1.1rem;'>{updated_at}</h3></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    profile_memo = profile.get("memo", "")

    st.markdown("<div style='margin-top:1rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;'>", unsafe_allow_html=True)
    stats = [
        ("Profile note", "Saved" if profile_memo else "Empty", "rgba(59,130,246,0.18)"),
        ("Favorite genre", user_prefs.get("Genre", "Not set"), "rgba(16,185,129,0.18)"),
        ("Listening focus", f"{user_prefs.get('Focus', 'Not set')} · {user_prefs.get('Era', 'Not set')} · {user_prefs.get('Tempo', 'Not set')} BPM", "rgba(244,63,94,0.18)"),
    ]
    for title, value, border in stats:
        st.markdown(
            f"<div style='padding:1.2rem;border-radius:1.5rem;background:rgba(15,23,42,0.95);border:1px solid {border};'>"
            f"<p style='margin:0;color:#94a3b8;'>{title}</p>"
            f"<h3 style='margin:.5rem 0 0;color:#ffffff;font-size:1.1rem;'>{value}</h3>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("profile_form"):
        st.markdown("<div style='margin-top:1.5rem;display:grid;grid-template-columns:1.6fr 1fr;gap:1.5rem;'>", unsafe_allow_html=True)
        with st.container():
            display_name = st.text_input("Display name", value=st.session_state.current_user)
            memo = st.text_area(
                "Personal memo",
                value=profile_memo,
                height=210,
                help="Write your musical goals, mood directions, or listening notes for your next session.",
            )
            if profile_memo:
                st.markdown(
                    "<div style='margin-top:1rem;padding:1rem;border-radius:1.25rem;background:rgba(15,23,42,0.8);border:1px solid rgba(100,116,139,0.16);'>"
                    "<strong>Your memo</strong>"
                    f"<p style='margin:.5rem 0 0;color:#cbd5e1;'>{profile_memo}</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='margin-top:1rem;padding:1rem;border-radius:1.25rem;background:rgba(15,23,42,0.8);border:1px solid rgba(100,116,139,0.16);'>"
                    "<p style='margin:0;color:#cbd5e1;'>Add a memo so your next session remembers your mood and goals.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
        with st.container():
            st.markdown("<div style='padding:1.25rem;border-radius:1.5rem;background:rgba(7,10,20,0.88);border:1px solid rgba(100,116,139,0.18);'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#e2e8f0;margin-bottom:0.75rem;'>Profile avatar</h3>", unsafe_allow_html=True)
            st.image(get_avatar_data_uri(user_email), width=200, caption="Current avatar")
            uploaded_file = st.file_uploader(
                "Upload a new avatar",
                type=["png", "jpg", "jpeg"],
                help="Choose an image to personalize your profile.",
            )
            if uploaded_file is not None:
                st.markdown("<div style='margin-top:1rem;color:#cbd5e1;'>Preview</div>", unsafe_allow_html=True)
                st.image(uploaded_file, width=200)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            save = st.form_submit_button("Save profile")
        with col2:
            reset = st.form_submit_button("Reset avatar")

        if save:
            if not display_name.strip():
                st.error("Display name cannot be empty.")
            else:
                if user_email != "Not available":
                    save_user_profile(user_email, memo=memo, avatar_bytes=uploaded_file.read() if uploaded_file else None)
                st.session_state.current_user = display_name.strip()
                st.success("Profile updated successfully.")
                st.rerun()
        if reset:
            if user_email != "Not available":
                avatar_path = _profile_avatar_path(user_email)
                if avatar_path.exists():
                    avatar_path.unlink()
                save_user_profile(user_email, memo=memo)
                st.success("Avatar reset to default.")
                st.rerun()

    st.markdown("<div style='margin-top:1.5rem;padding:1rem;border-radius:1.5rem;background:rgba(15,23,42,0.88);border:1px solid rgba(100,116,139,0.18);'>", unsafe_allow_html=True)
    if st.button("Back to recommendation studio", key="profile_back"):
        st.session_state.current_page = "Home"
        set_query_params({})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def show_admin_page():
    user_email = st.session_state.current_email or ""
    is_admin = check_if_admin(user_email)
    if not is_admin:
        st.error("Access denied — admin credentials required.")
        if st.button("Back to studio"):
            st.session_state.current_page = "Home"
            set_query_params({})
            st.rerun()
        return

    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:1rem;'>", unsafe_allow_html=True)
    if st.button("← Back to studio", key="admin_back_top"):
        st.session_state.current_page = "Home"
        set_query_params({})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='padding:1.2rem;border-radius:1.2rem;background:linear-gradient(135deg, rgba(8,10,20,0.96), rgba(20,28,44,0.96));box-shadow:0 20px 50px rgba(0,0,0,0.28);'>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin:0;color:#fff;font-size:2rem;'>Admin Console</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='margin:.5rem 0 0;color:#94a3b8;'>Manage users, view dataset stats, and perform administrative actions.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # User management
    st.subheader("User Management")
    with st.expander("View and filter users", expanded=True):
        users_list = get_all_users_list()
        df_users = pd.DataFrame(users_list)
        filter_text = st.text_input("Search users (email or name)")
        if filter_text:
            df_users = df_users[df_users["email"].str.contains(filter_text, case=False) | df_users["name"].str.contains(filter_text, case=False)]
        st.dataframe(df_users.reset_index(drop=True))

    st.markdown("---")

    with st.form("add_user_form"):
        st.markdown("### Add new user")
        new_name = st.text_input("Full name", key="admin_new_name")
        new_email = st.text_input("Email", key="admin_new_email")
        new_password = st.text_input("Password", type="password", key="admin_new_password")
        new_is_admin = st.checkbox("Grant admin privileges", key="admin_new_is_admin")
        submit_add = st.form_submit_button("Create user")
        if submit_add:
            if not new_email or not validate_email(new_email):
                st.error("Enter a valid email for the new user.")
            elif user_exists(new_email):
                st.error("A user with that email already exists.")
            else:
                success, msg = register_user(new_name.strip() or new_email, new_email, new_password or "password123")
                if success:
                    if new_is_admin:
                        set_admin_status(new_email, True)
                    st.success(f"Created user {new_email.lower()}.")
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")

    # Delete users
    st.subheader("Delete users")
    users_list = get_all_users_list()
    emails = sorted([u["email"] for u in users_list])
    to_delete = st.multiselect("Select users to delete", options=emails)
    if to_delete:
        if st.button("Delete selected users"):
            for e in to_delete:
                # prevent deleting self
                if e == user_email.lower():
                    st.warning(f"Skipping deletion of current admin account: {e}")
                    continue
                delete_user(e)
            st.success("Selected users deleted.")
            st.rerun()

    st.markdown("---")

    # Manage admin status
    st.subheader("Admin privileges")
    users_list = get_all_users_list()
    admin_options = {u["email"]: u for u in users_list}
    selected_user = st.selectbox("Select user", options=sorted(admin_options.keys()))
    if selected_user:
        current_admin_status = admin_options[selected_user]["is_admin"]
        new_admin_status = st.checkbox(f"Grant admin privileges to {selected_user}", value=current_admin_status)
        if st.button("Update admin status"):
            if new_admin_status != current_admin_status:
                success, msg = set_admin_status(selected_user, new_admin_status)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.info("No changes made.")

    st.markdown("---")

    # Dataset stats
    st.subheader("Dataset statistics")
    try:
        df = load_data()
        st.markdown("**Top artists**")
        top_artists = df['artist'].value_counts().head(10)
        st.bar_chart(top_artists)
        st.markdown("**Top songs**")
        top_songs = df['song_key'].value_counts().head(10)
        st.bar_chart(top_songs)

        st.markdown("**Artist table**")
        artist_table = top_artists.reset_index(name='count').rename(columns={"index": "artist"})
        st.table(artist_table)

        st.markdown("**Song table**")
        song_table = top_songs.reset_index(name='count').rename(columns={"index": "song"})
        st.table(song_table)
    except Exception as e:
        st.error(f"Unable to load dataset for stats: {e}")


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


def normalize_onboarding_genre(genre_value):
    if not genre_value:
        return None
    return genre_value.lower().strip()


def find_genre_seed_song(df_content, genre_value):
    genre_key = normalize_onboarding_genre(genre_value)
    if not genre_key:
        return None
    genre_mask = df_content['genre'].fillna('').str.lower() == genre_key
    filtered = df_content[genre_mask]
    if filtered.empty:
        return None
    filtered = filtered.sort_values(['popularity', 'song_name'], ascending=[False, True])
    return filtered.iloc[0]['song_name']


def apply_onboarding_seed(df_content):
    if st.session_state.get('seed_song_applied'):
        return
    if not st.session_state.onboarding_data:
        return
    genre = st.session_state.onboarding_data.get('Genre')
    if not genre:
        st.session_state.seed_song_applied = True
        return
    default_song = find_genre_seed_song(df_content, genre)
    if default_song:
        st.session_state.song_title = default_song
    st.session_state.seed_song_applied = True


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
if "otp_pending" not in st.session_state:
    st.session_state.otp_pending = False
if "otp_code" not in st.session_state:
    st.session_state.otp_code = None
if "otp_expires" not in st.session_state:
    st.session_state.otp_expires = 0
if "otp_message" not in st.session_state:
    st.session_state.otp_message = ""
if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False
if "onboarding_data" not in st.session_state:
    st.session_state.onboarding_data = {}
if "seed_song_applied" not in st.session_state:
    st.session_state.seed_song_applied = False
if "current_email" not in st.session_state:
    st.session_state.current_email = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if not st.session_state.authenticated:
    show_auth_page()
    st.stop()

if not st.session_state.onboarding_complete:
    show_onboarding_page()
    st.stop()

params = get_query_params()
if params.get("page"):
    page_param = params.get("page")
    st.session_state.current_page = page_param[0] if isinstance(page_param, list) else page_param

if st.session_state.current_page == "Profile":
    show_profile_page()
    st.stop()

if st.session_state.current_page == "Admin":
    show_admin_page()
    st.stop()

st.title(f"Welcome back, {st.session_state.current_user} 🎶")
if st.session_state.onboarding_data:
    prefs = st.session_state.onboarding_data
    with st.container():
        st.markdown("<div style='padding:1.3rem;border-radius:1.5rem;background:linear-gradient(135deg, rgba(16,185,129,0.16), rgba(59,130,246,0.15));border:1px solid rgba(59,130,246,0.18);'>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;'>"
            f"<div style='padding:1rem;border-radius:1rem;background:rgba(15,23,42,0.85);'><strong>Mood</strong><div style='margin-top:0.5rem;color:#d1fae5;'>{prefs['Mood']}</div></div>"
            f"<div style='padding:1rem;border-radius:1rem;background:rgba(15,23,42,0.85);'><strong>Genre</strong><div style='margin-top:0.5rem;color:#c7d2fe;'>{prefs['Genre']}</div></div>"
            f"<div style='padding:1rem;border-radius:1rem;background:rgba(15,23,42,0.85);'><strong>Focus</strong><div style='margin-top:0.5rem;color:#fce7f3;'>{prefs['Focus']}</div></div>"
            f"<div style='padding:1rem;border-radius:1rem;background:rgba(15,23,42,0.85);'><strong>Era</strong><div style='margin-top:0.5rem;color:#c7f9cc;'>{prefs['Era']}</div></div>"
            f"<div style='padding:1rem;border-radius:1rem;background:rgba(15,23,42,0.85);'><strong>Tempo</strong><div style='margin-top:0.5rem;color:#fde68a;'>{prefs['Tempo']} BPM</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

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
    avatar_src = get_avatar_data_uri(st.session_state.current_email or None)
    if avatar_src:
        avatar_html = f'''
            <div style="width:88px;height:88px;border-radius:50%;margin:0 auto 1rem auto;
            background:linear-gradient(135deg,#1db954,#1ed760);display:flex;align-items:center;justify-content:center;
            box-shadow:0 18px 40px rgba(0,0,0,0.25);cursor:pointer;overflow:hidden;">
                <img src="{avatar_src}" width="88" height="88" style="object-fit:cover;display:block;" />
            </div>
        '''
    else:
        avatar_html = '''
            <div style="width:88px;height:88px;border-radius:50%;margin:0 auto 1rem auto;
            background:linear-gradient(135deg,#1db954,#1ed760);display:flex;align-items:center;justify-content:center;
            box-shadow:0 18px 40px rgba(0,0,0,0.25);cursor:pointer;">
                <span style="font-size:2.4rem;color:#ffffff;">🎧</span>
            </div>
        '''
    st.markdown(avatar_html, unsafe_allow_html=True)
    if st.button("Open profile", key="open_profile"):
        st.session_state.current_page = "Profile"
        set_query_params({"page": ["Profile"]})
        st.rerun()
    st.markdown('<div style="text-align:center;color:#94a3b8;margin-bottom:0.75rem;">Tap the avatar or press "Open profile" to edit your account.</div>', unsafe_allow_html=True)
    st.markdown(f"### Signed in as **{st.session_state.current_user}**")
    if st.button("Sign out"):
        st.session_state.authenticated = False
        st.session_state.current_user = "Guest"
        st.session_state.current_email = ""
        st.session_state.current_page = "Home"
        st.session_state.auth_message = "You have signed out successfully."
        st.session_state.onboarding_complete = False
        st.session_state.onboarding_data = {}
        st.session_state.seed_song_applied = False
        st.rerun()

    st.markdown("---")
    st.header("Recommendation controls")
    mode = st.radio("Mode", ["Content-based", "Collaborative", "Hybrid"], index=2)

    if "song_title" not in st.session_state or not st.session_state.seed_song_applied:
        default_song = song_options[0]
        if st.session_state.onboarding_data and st.session_state.onboarding_data.get('Genre'):
            onboard_genre = st.session_state.onboarding_data.get('Genre')
            genre_song = find_genre_seed_song(df_content, onboard_genre)
            if genre_song and genre_song in song_options:
                default_song = genre_song
        st.session_state.song_title = default_song
        st.session_state.seed_song_applied = True
    
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
