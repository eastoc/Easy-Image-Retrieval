
import torch
import torch.nn as nn
import clip
from PIL import Image
 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("models/ViT-B-32.pt", device=device)
 
model.float()
model.eval()

image = preprocess(Image.open("images/1.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a bottle"]).to(device)
 
print("text:", text)
 
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
 
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# export to ONNX
 
 
class ImgModelWrapper(nn.Module):
    def __init__(self):
        super(ImgModelWrapper, self).__init__()
        self.model = model
 
    def forward(self, image):
        image_features = model.encode_image(image)
        return image_features
 
 
class TxtModelWrapper(nn.Module):
    def __init__(self):
        super(TxtModelWrapper, self).__init__()
        self.model = model
 
    def forward(self, image):
        text_features = model.encode_text(text)
        return text_features
 
 
img_model = ImgModelWrapper()
txt_model = TxtModelWrapper()
 
torch.onnx.export(img_model,               # model being run
                  image,                         # model input (or a tuple for multiple inputs)
                  "openai_vit_img.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch'}})
torch.onnx.export(txt_model,               # model being run
                  text,                         # model input (or a tuple for multiple inputs)
                  "openai_vit_txt.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch'}})