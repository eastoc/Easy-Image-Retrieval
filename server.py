import uvicorn 
from fastapi import FastAPI

from typing import List
import time

import urllib.request
from io import BytesIO

app = FastAPI()

#from utils import load_json
#database = load_json('dataset.json')
#map = load_json('map.json')
#nodes = load_json('nodes.json')

Robot = IRSystem()

@app.post("/image-retrieval/CLIP")
async def image_retrieval(req:  dict):
    
    return res

if __name__ == '__main__':
    uvicorn.run(app="server:app", host="0.0.0.0", port=9532, reload=True)