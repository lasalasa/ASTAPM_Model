from pydantic import BaseModel
from datetime import datetime

# code adapted from (SQL (Relational) Databases - FastAPI, n.d.)
class DataSourceBaseSchema(BaseModel):
    ds_name: str
    ds_description: str
    ds_im_path: str
    ds_ex_path: str
    ds_type: str
    ds_raw_db: str
    ds_db: str

class DataSourceCreateSchema(DataSourceBaseSchema):
    pass

class DataSourceUpdateSchema(DataSourceBaseSchema):
    pass

class DataSourceSchema(DataSourceBaseSchema):
    ds_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
# code adapted end