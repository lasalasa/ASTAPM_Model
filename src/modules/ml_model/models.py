from sqlalchemy import Column, Integer, String, Enum, JSON, ForeignKey, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class MachineLearningModel(Base):
    __tablename__ = "ml_model"
    
    ml_model_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ml_model_name = Column(String(50), unique=True, nullable=False)
    ml_model_type = Column(Enum('SVM', 'LSTM', name="model_type_enum"), nullable=False)  # "SVM" or "LSTM"
    ml_model_description = Column(String(255), nullable=True)
    feature_engineering = Column(Enum('NMF', 'ADSGL', name="feature_engineering_enum"), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationship to train_model
    train_models = relationship("TrainModel", back_populates="ml_model")


class TrainModel(Base):
    __tablename__ = "train_model"
    
    train_model_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ml_model_id = Column(Integer, ForeignKey("ml_model.ml_model_id"), nullable=False)
    training_data = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationship to model
    ml_model = relationship("MachineLearningModel", back_populates="train_models")



# https://docs.sqlalchemy.org/en/20/orm/quickstart.html
