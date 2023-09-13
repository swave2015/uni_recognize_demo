import sys
sys.path.append('./LAVIS')

import onnxruntime as ort
import cv2
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image
from transformers import XLMRobertaTokenizer
from lavis.models import load_model_and_preprocess
import torch

device = 'cpu'
txt = "an image of a white cat"
raw_image = Image.open("./test_imgs/00000004.jpg").convert("RGB")
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

onnx_path = '/data/xcao/code/uni_recognize_demo/algorithms/onnx_export/blip2_itm_vision_encoder.onnx'
ort_session = ort.InferenceSession(onnx_path)

inputs = {
    'input_img': img.cpu().numpy()
}

ort_outs = ort_session.run(None, inputs)
print(f"img_ort_outs: {ort_outs[0]}")