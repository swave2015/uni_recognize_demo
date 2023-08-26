from beit3_model import BEIT3Model
import torch
import os
import cv2
from demo_utils import resize_or_pad

device = "cuda" if torch.cuda.is_available() else "cpu"
beit3Model = BEIT3Model(device)
test_img_dir = '/data/xcao/code/uni_recognize_demo/test_misc/out_frames'
input_size = 384
test_prompt = ["an image of a dog"]
beit3Model.infer_text(test_prompt)

for filename in os.listdir(test_img_dir):
    filepath = os.path.join(test_img_dir, filename)
    image = cv2.imread(filepath)
    processed_img = resize_or_pad(image, input_size)
    score = beit3Model.infer_img(processed_img)
    value = score.detach().item()
    print(f"filename: {filename}, score: {value}")
    # if value > 0.6:
    #     print(f"filename: {filename}, score: {value}")
    # feature = beit3Model.extract_img_features(filepath)
    # print('feature: ', feature.shape)