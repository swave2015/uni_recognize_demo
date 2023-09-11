import sys
sys.path.append('./LAVIS')

import torch
from PIL import Image
import time
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import onnxruntime

raw_image = Image.open("./test_imgs/00000004.jpg").convert("RGB")

# setup device to use
# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
device = 'cpu'

caption = "Series of images: a person sitting"
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

def test_model_with_prompt(prompt):
    itm = False
    itc = False
    txt = text_processors["eval"](prompt)
    test_onnx_infer = True
    start_time = time.time()
    if itm:
        text_token = model.tokenizer(txt, 
                                    padding="max_length",
                                    truncation=True, 
                                    max_length=model.max_txt_len,
                                    return_tensors="pt").to(device)
        print(f"input_text_token_shape: {text_token.input_ids.shape}")
        print(f"input_text_token_attention_mask_shape: {text_token.attention_mask.shape}")
        print(f"input_text_token_atten_mask: {text_token.attention_mask}")
        onnx_model_path = '/data/xcao/code/uni_recognize_demo/algorithms/onnx_export/blip2_itm_constant.onnx'
        torch.onnx.export(model,
                          (img, text_token.input_ids, text_token.attention_mask),
                          onnx_model_path,
                          export_params=False,
                          opset_version=13,
                          do_constant_folding=True,
                          input_names=['input_img','token_input_ids','token_attention_mask'],
                          output_names = ['output']
                          )

        # itm_output = model(img, text_token.input_ids, text_token.attention_mask)
        # itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        # print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')

    if itc:
        itc_score = model({"image": img, "text_input": txt}, match_head='itc')
        print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

    if test_onnx_infer:
        pass
        

prompts = [
    "an image of a white cat",
    # Add more prompts here
]

for prompt in prompts:
    test_model_with_prompt(prompt)