"""Local PostgreSQL configuration independent of API and model imports."""
from __future__ import annotations

import os
from pathlib import Path
import re

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from dotenv import dotenv_values
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[1]


def connection_parameters():
    local = dotenv_values(ROOT/'scene-search-prototype/.env')
    def value(name, default=None):
        return os.environ.get(name, local.get(name, default))
    password = value('DB_PASSWORD')
    if not password:
        raise RuntimeError('DB_PASSWORD missing from local environment configuration')
    return dict(host=value('DB_HOST','127.0.0.1'),port=int(value('DB_PORT','15432')),
                dbname=value('DB_NAME','group8_fyp'),user=value('DB_USER','fyp_user'),
                password=password,connect_timeout=5)


def open_database(*, schema='public', readonly=False):
    conn = psycopg.connect(**connection_parameters(), row_factory=dict_row)
    try:
        conn.read_only = readonly
        conn.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(schema)))
        register_vector(conn)
        return conn
    except Exception:
        conn.close()
        raise


def migrate(*, schema='public'):
    from alembic.config import Config
    from alembic import command
    if not re.fullmatch('[a-zA-Z_][a-zA-Z0-9_]*',schema):
        raise ValueError('Invalid migration schema')
    config = Config(str(ROOT/'alembic.ini'))
    config.set_main_option('script_location',str(ROOT/'migrations'))
    config.attributes['schema'] = schema
    command.upgrade(config,'head')
