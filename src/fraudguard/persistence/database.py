"""Database connection, session management, and table initialization."""

from __future__ import annotations

import logging
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from fraudguard.persistence.models import Base

logger = logging.getLogger(__name__)

DEFAULT_DB_URL = os.getenv("DATABASE_URL", "sqlite:///fraudguard.db")


def get_engine(db_url: str = DEFAULT_DB_URL):
    """Creates SQLAlchemy engine with proper connection arguments."""
    is_sqlite = db_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    return engine


_engine = None
_SessionFactory = None


def init_engine(db_url: str = DEFAULT_DB_URL):
    global _engine, _SessionFactory
    _engine = get_engine(db_url)
    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def init_db(db_url: str = DEFAULT_DB_URL) -> None:
    """Creates all tables if they do not exist."""
    engine = init_engine(db_url)
    logger.info("Initializing database tables on %s", db_url.split("@")[-1] if "@" in db_url else db_url)
    Base.metadata.create_all(bind=engine)


def get_session(db_url: str = DEFAULT_DB_URL) -> Session:
    """Returns a standalone SQLAlchemy session."""
    global _SessionFactory
    if _SessionFactory is None:
        init_engine(db_url)
    return _SessionFactory()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions per request."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()
