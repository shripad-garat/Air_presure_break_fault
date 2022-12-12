import pymongo
import pandas as pd
import json

clint = pymongo.MongoClient("mongodb+srv://host:0pQPoWnWMSlypR0N@cluster0.alpnhdp.mongodb.net/?retryWrites=true&w=majority")
data_file_path = "https://raw.githubusercontent.com/yadav-avnish/aps-fault-detection/main/aps_failure_training_set1.csv"
Database_name = "AirPresureFault"
collection_name = "sensor"

if __name__ =="__main__":
    df = pd.read_csv(data_file_path)
    print(f" Rows and columns=: {df.shape}")
    df.reset_index(drop= True,inplace=True)
    json_records = list(json.loads(df.T.to_json()).values())
    print(json_records[0])
    clint[Database_name][collection_name].insert_many(json_records)
