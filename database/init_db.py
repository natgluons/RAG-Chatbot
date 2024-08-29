from sqlalchemy import create_engine, Column, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class UserInteraction(Base):
    __tablename__ = 'user_interactions'
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    request_count = Column(Integer, default=0)
    last_request_time = Column(DateTime, default=datetime.datetime.now)

DATABASE_URL = "sqlite:///database.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)

def init_db():
    # Example entry to initialize the database
    interaction = UserInteraction(request_count=1, last_request_time=datetime.datetime.now())
    session.add(interaction)
    session.commit()
    print("Database initialized successfully with SQLAlchemy.")

if __name__ == "__main__":
    init_db()
