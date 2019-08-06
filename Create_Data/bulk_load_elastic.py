import json
import pandas as pd
from elasticsearch import Elasticsearch


address = "SERVERADDRESS"
data = pd.read_csv(r'C:\Users\Gilad\PycharmProjects\DepressionResearch\Create_Data\SubmissionsDF.csv')
data.to_json('bulk_load.json', orient='index')

print("opening file")
json_data = open('bulk_load.json').read()
data = json.loads(json_data)

es = Elasticsearch(address)
index = 'depression_df'
doc_type = 'submission'

if es.indices.exists(index=index):
    index_counter = es.count(index=index)
else:
    es.indices.create(index=index, ignore=400)
    index_counter = es.count(index=index)

print("Loading Data...")
FLAG = True
while FLAG:
        counter = 1
        count = index_counter['count']
        if count == len(data):
            FLAG = False
        for ind in range(count, len(data)):
            try:
                if counter % 200 == 0:
                    print("Loaded: {} Rows".format(counter))
                es.index(index=index, doc_type=doc_type, id=ind, body=list(data.values())[ind], request_timeout=60)
                counter += 1
            except Exception as e:
                print(e)
                pass
