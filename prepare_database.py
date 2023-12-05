import torch
import clip
from PIL import Image
import os
import time
import json
import numpy as np
#from engine import Engine
from backend import dictionary
from utils import write_json

def create_cluster(sparse_data, nclust = 10):
    from sklearn.cluster import k_means_
    from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
    from sklearn.preprocessing import StandardScaler
    # Manually override euclidean
    def euc_dist(X, Y = None, Y_norm_squared = None, squared = False):
        #return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
        return cosine_similarity(X, Y)
    k_means_.euclidean_distances = euc_dist
    
    scaler = StandardScaler(with_mean=False)
    sparse_data = scaler.fit_transform(sparse_data)
    kmeans = k_means_.KMeans(n_clusters = nclust, n_jobs = 20, random_state = 3425)
    _ = kmeans.fit(sparse_data)
    return kmeans.labels_

class encoder():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Model is running with ',self.device)
        self.model, self.preprocess = clip.load("models/ViT-B-32.pt", device=self.device)
    
    def extract_feat(self, img):
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(image).to('cpu').detach().numpy()
        return feat

    def build_dataset(self, imgs_dir):
        imgs_file = os.listdir(imgs_dir)
        start = time.time()
        database = []
        for i, img_file in enumerate(imgs_file):
            data = dict()
            img_dir = os.path.join(imgs_dir, img_file)
            data['embedding'] = self.extract_feat(Image.open(img_dir))[0].tolist()
            data['image'] = img_file
            database.append(data)
        end = time.time()
        duration = (end - start)
        print("time cost: ", duration)
        
        return database



def prepare(imgs_dir):
    model = encoder()
    database = model.build_dataset(imgs_dir)
    write_json(database, 'dataset.json')

#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

#with torch.no_grad():
    #image_features = model.encode_image(image)
    #text_features = model.encode_text(text)
    
    #logits_per_image, logits_per_text = model(image, text)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
#print("image embeddings: ", image_features)
if __name__=='__main__':
    
    imgs_dir = 'datasets/dataset'
    prepare(imgs_dir)
    
    
    