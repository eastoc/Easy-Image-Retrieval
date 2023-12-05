# Time: 2023.10.12
import torch
import clip
from PIL import Image
import os
import time

class encoder():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Model is running with ',self.device)
        self.model, self.preprocess = clip.load("models/pth/ViT-B-32.pt", device=self.device)
    
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

#text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

#with torch.no_grad():
    #image_features = model.encode_image(image)
    #text_features = model.encode_text(text)
    
    #logits_per_image, logits_per_text = model(image, text)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
#print("image embeddings: ", image_features)
if __name__=='__main__':
    model = encoder()
    imgs_dir = 'datasets/debug'
    database = model.encode_dataset(imgs_dir)