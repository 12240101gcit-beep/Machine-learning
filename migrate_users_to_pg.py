"""Migrate local users_db.json into PostgreSQL.

Usage:
  - Ensure Postgres is running and accessible via DATABASE_URL or POSTGRES_* env vars.
  - Optionally copy .env.example to .env and set credentials.
  - Run:
      "c:/Users/tsher/OneDrive/Desktop/Machine learning 2/.venv/Scripts/python.exe" migrate_users_to_pg.py

The script will create the `users` table if it doesn't exist and upsert users by email.
"""
from pathlib import Path
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, select, insert, update, text
from sqlalchemy.exc import SQLAlchemyError

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')

DATABASE_URL = os.getenv('DATABASE_URL', '').strip()
if not DATABASE_URL:
    pg_user = os.getenv('POSTGRES_USER', '').strip()
    pg_pass = os.getenv('POSTGRES_PASSWORD', '').strip()
    pg_host = os.getenv('POSTGRES_HOST', 'localhost').strip()
    pg_port = os.getenv('POSTGRES_PORT', '5432').strip()
    pg_db = os.getenv('POSTGRES_DB', 'music_recommender').strip()
    if pg_user and pg_pass:
        DATABASE_URL = f'postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}'

if not DATABASE_URL:
    print('No DATABASE_URL or POSTGRES_* env vars set. Please configure DB connection in .env or env vars.')
    sys.exit(1)

engine = create_engine(DATABASE_URL, future=True)
metadata = MetaData()
users_table = Table(
    'users', metadata,
    Column('email', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('password', String, nullable=False),
    Column('created_at', DateTime, default=datetime.utcnow),
)

USERS_FILE = BASE / 'users_db.json'
if not USERS_FILE.exists():
    print(f'Local users file not found: {USERS_FILE}')
    sys.exit(1)

with open(USERS_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
users = data.get('users', {})
if not users:
    print('No users found in users_db.json')
    sys.exit(0)

print('Connecting to database...')
try:
    with engine.begin() as conn:
        metadata.create_all(conn)
        print('Ensured users table exists.')

        for email, info in users.items():
            name = info.get('name', email)
            password = info.get('password', '')
            # Upsert logic: if exists, update name/password, else insert
            stmt = select(users_table).where(users_table.c.email == email)
            res = conn.execute(stmt).first()
            if res:
                upd = (
                    update(users_table)
                    .where(users_table.c.email == email)
                    .values(name=name, password=password)
                )
                conn.execute(upd)
                print(f'Updated user: {email}')
            else:
                ins = (
                    insert(users_table).values(
                        email=email,
                        name=name,
                        password=password,
                        created_at=datetime.utcnow(),
                    )
                )
                conn.execute(ins)
                print(f'Inserted user: {email}')

    print('\nMigration complete.')
except SQLAlchemyError as exc:
    print('Database error:', exc)
    sys.exit(1)
