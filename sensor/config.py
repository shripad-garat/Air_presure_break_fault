import pymongo
import pandas as pd
import json
from dataclasses import dataclass
# Provide the mongodb localhost url to connect python to mongodb.
import os
@dataclass
class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL")



env_var = EnvironmentVariable()
print(env_var.mongo_db_url)
TARGET_COLUMN = "class"
TARGET_COLUMN_MAPPING = {
    "pos":1,
    "neg":0
}
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)