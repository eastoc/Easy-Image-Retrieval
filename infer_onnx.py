import numpy as np
import torch
import onnxruntime
from PIL import Image
import time
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

MODEL_FILE = 'models/openai_vit_img.onnx'
DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE_INDEX = 0
#DEVICE=f'{DEVICE_NAME}:{DEVICE_INDEX}'
class image_encoder():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocessor = _transform(224)
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.encoder = onnxruntime.InferenceSession(MODEL_FILE, providers=self.providers)

    def infer(self, img):
        #img = self.preprocessor(img).unsqueeze(0).numpy()
        inputs = {self.encoder.get_inputs()[0].name: img}
        output = self.encoder.get_outputs()[0].name
        res = self.encoder.run([output], inputs)[0]
        return res

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize((n_px, n_px)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def test():
    #create_model()
    import os
    imgs_dir = 'images'
    imgs_file = os.listdir('images')
    model = image_encoder()
    t1 = time.time()
    imgs = []
    for i, img_file in enumerate(imgs_file):
        image = model.preprocessor(Image.open("images/"+img_file))
        image = np.expand_dims(image, axis=0)
        imgs.append(image)
    n = len(imgs)
    imgs = np.concatenate(imgs, axis=0)
    print(imgs.shape)
    res = model.infer(imgs)
    t2 = time.time()
    #print(res)
    print(f'Done. ({(1E3 * (t2 - t1)/n):.1f}ms) Inference.', 'FPS:', 1/((t2-t1)/n))

if __name__ == "__main__":
    test()   