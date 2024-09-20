# pip install mdb-parser
import os
from mdb_parser import MDBParser, MDBTable

# # code adapted from (Mdb-parser, 2022)
class MdbParserExtension:
    def __init__(self, file_path):
        self.db = MDBParser(file_path=file_path)
        pass
    
    def get_tables(self):
        db = self.db
        return db.tables
    
    def read_table(self, table_name):
        db = self.db
        return db.get_table(table_name)
    
# end of adapted code
# TODO REF:https://pypi.org/project/mdb-parser/