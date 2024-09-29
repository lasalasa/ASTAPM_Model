from sqlalchemy import Column, Integer, String, DateTime, func
from src.database import Base

class DataSource(Base):
    __tablename__ = "data_source"

    ds_id = Column(Integer, primary_key=True, index=True)
    ds_name = Column(String, unique=True, index=True)
    ds_description = Column(String)
    ds_im_path = Column(String)
    ds_ex_path = Column(String)
    ds_type = Column(String)
    ds_raw_db = Column(String)
    ds_db = Column(String)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
