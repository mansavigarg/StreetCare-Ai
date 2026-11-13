from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config

database_engine = create_engine(config.STREETCARE_DATABASE_URL, connect_args={"check_same_thread": False})
session_factory = sessionmaker(bind=database_engine)
