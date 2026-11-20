"""Non-destructive baseline and additive Stage 10 migrations."""
from alembic import context
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool

from src.pg_config import connection_parameters

if context.is_offline_mode():
    raise RuntimeError('Online inspection is required to adopt the existing schema')

params = connection_parameters()
url = URL.create('postgresql+psycopg',username=params['user'],password=params['password'],
                 host=params['host'],port=params['port'],database=params['dbname'])
engine = create_engine(url,poolclass=NullPool,connect_args={'connect_timeout':5})
schema = context.config.attributes.get('schema','public')
try:
    with engine.begin() as connection:
        quoted = connection.dialect.identifier_preparer.quote(schema)
        connection.execute(text(f'SET LOCAL search_path TO {quoted}, public'))
        context.configure(connection=connection,version_table_schema=schema)
        with context.begin_transaction():
            context.run_migrations()
finally:
    engine.dispose()
