from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base
import logging

logger = logging.getLogger(__name__)

# Global variables
engine = None
SessionLocal = None

def init_db(database_url: str):
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    try:
        engine = create_engine(database_url, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def get_session() -> Session:
    """Get database session"""
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    
    return SessionLocal()

def close_db():
    """Close database connection"""
    global engine
    if engine:
        engine.dispose()
        logger.info("Database connection closed")