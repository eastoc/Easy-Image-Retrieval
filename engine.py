# Time: 2023.10.16
from backend import dictionary
from img_encoder import encoder
from PIL import Image
import numpy as np
from utils import cosin_sim
#from sklearn.metrics import top_k_accuracy_score
import matplotlib.pyplot as plt
import time

class Engine():
    def __init__(self, database, map=None, nodes=None):
        
        #self.BoW = dictionary(database, n_clusters=100)
        #self.BoW.build()
        self.database = database
        self.map = map
        self.nodes = np.array(nodes['nodes'])
        self.model = encoder()

    def search(self, query_dir, num_res=20):
        query = self.model.extract_feat(Image.open(query_dir))
        
        # find the nearest cluster nodes
        query = query.reshape(512, 1)
        node_id = self.find_cluster(query)
        
        # find the matching embedding in the the nearest clusters
        key_imgs = self.find_keys(query, node_id, num_res=num_res)
    
        return key_imgs
    
    def find_keys(self, query, node_id, num_res):
        key_embeddings = self.map[str(node_id)]
        mat = []
        imgs = []
        for i, data in enumerate(key_embeddings):
            mat.append(np.array(data['embedding']))
            imgs.append(data['image'])

        mat = np.array(mat)
        confi = cosin_sim(mat, query)
        confi = confi.reshape(len(confi))
        sort_ids = np.argsort(confi)
        
        if num_res < len(imgs):
            key_ids = sort_ids[0:num_res]
            key_imgs = []
            for key_id in key_ids:
                key_imgs.append(imgs[key_id])

        elif num_res >= len(imgs):
            key_imgs = imgs

        return key_imgs

    def find_cluster(self, query):
        logits = cosin_sim(self.nodes, query)
        nodes_id = np.argmin(logits)
        print('nodes id:', nodes_id)
        return nodes_id

if __name__=='__main__':
    # 接入后端
    from utils import load_json
    database = load_json('dataset.json')
    map = load_json('map.json')
    nodes = load_json('nodes.json')
    query_dir = 'query/query7.jpg'
    
    # 启动搜索引擎
    t1 = time.time()
    IR = Engine(database, map, nodes)
    # 开始检索
    t2 = time.time()
    key_imgs_dir = IR.search(query_dir, num_res=20)
    t3 = time.time()
    print("engine initial time:", t2-t1)
    print("search time:", t3-t2)
    # 绘制检索结果
    plt.figure(figsize=(5, 4), dpi=1200)

    for i, key_img_dir in enumerate(key_imgs_dir):
        img = Image.open('datasets/dataset/'+key_img_dir)
        plt.subplot(5, 4, i+1)
        plt.axis(False)
        plt.imshow(img)
    plt.savefig('results/test.png')
    plt.show()
    
    
        
    
    
    #import cv2
    #key_img = cv2.imread('datasets/dataset/'+key_img_dir)
    #cv2.imshow('key img',key_img)
    #cv2.waitKey(0)
