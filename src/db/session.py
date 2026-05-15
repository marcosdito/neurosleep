"""
Database session and engine management.

Reads the connection URL from the DATABASE_URL environment variable.
Defaults to a sensible local PostgreSQL URL that matches the docker-compose
configuration shipped with the project.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base

DEFAULT_DATABASE_URL = "postgresql+psycopg2://neurosleep:neurosleep@localhost:5432/neurosleep"


def get_engine(database_url: str | None = None) -> Engine:
    """Build (or rebuild) a SQLAlchemy engine from the env var or argument."""
    url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine: Engine | None = None) -> Engine:
    """Create all tables. Idempotent — safe to run multiple times."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""
    engine = engine or get_engine()
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
