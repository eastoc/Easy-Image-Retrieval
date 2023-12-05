import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import matplotlib.pyplot as plt
from PIL import Image

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "models/trt/openai_vit_img.engine"
input_file  = "input.jpg"
output_file = "output.ppm"

def _convert_image_to_rgb(image):
        return image.convert("RGB")

def _transform():
    return Compose([
        Resize((224, 224)),
        _convert_image_to_rgb,
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

class encoder_trt():
    def __init__(self, engine_dir):
        self.preprocessor = _transform()
        self.model = self.load_engine(engine_dir)
    # For torchvision models, input images are loaded in to a range of [0, 1] and
    # normalized using mean = [0.485, 0.456, 0.406] and stddev = [0.229, 0.224, 0.225].
    
    def preprocess(self, image):
        # Mean normalization
        image = self.preprocessor(image).astype('float16')
        return image

    def load_engine(self, engine_file_path):
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def infer(self, engine, input_file, output_file):
        print("Reading input image from file {}".format(input_file))
        with Image.open(input_file) as img:
            input_image = preprocess(img)
            image_width = img.width
            image_height = img.height

        with engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
            # Allocate host and device buffers
            bindings = []
            for binding in engine:
                binding_idx = engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                if engine.binding_is_input(binding):
                    input_buffer = np.ascontiguousarray(input_image)
                    input_memory = cuda.mem_alloc(input_image.nbytes)
                    bindings.append(int(input_memory))
                else:
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            # Synchronize the stream
            stream.synchronize()

        with postprocess(np.reshape(output_buffer, (image_height, image_width))) as img:
            print("Writing output image to file {}".format(output_file))
            img.convert('RGB').save(output_file, "PPM")

if __name__ == '__main__':
    model = encoder_trt(engine_file)
    