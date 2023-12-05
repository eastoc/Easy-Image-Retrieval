# Time: 2023.10.13
import json
import numpy as np

def write_json(database, file_name='dataset.json'):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(database, file)
        file.close()
    print('Build dataset.json: Done.')

def load_json(json_dir):
    with open(json_dir, 'rt', encoding='utf-8') as file:
        database = json.load(file)
        print("Successfully loading %s."%(json_dir))
    return database

def cosin_sim(x, y):
    return 1-np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))