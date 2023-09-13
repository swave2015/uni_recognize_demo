import sys
sys.path.append('./LAVIS')

import onnxruntime as ort
import torch
from PIL import Image
import time
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

raw_image = Image.open("./test_imgs/00000004.jpg").convert("RGB")

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

caption = "Series of images: a person sitting"
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

def test_model_with_prompt(prompt):
    itm = True
    itc = False
    txt = text_processors["eval"](prompt)
    start_time = time.time()
    if itm:
        text_token = model.tokenizer(txt, 
                                    padding="max_length",
                                    truncation=True, 
                                    max_length=model.max_txt_len,
                                    return_tensors="pt").to(device)
        print(f"input_text_token_shape: {text_token.input_ids.shape}")
        print(f"input_text_token_attention_mask_shape: {text_token.attention_mask.shape}")

        onnx_path = '/data/xcao/code/uni_recognize_demo/algorithms/blip2_onnx_full_model_export/blip2_itm_constant.onnx'
        ort_session = ort.InferenceSession(onnx_path)
        inputs = {
            'input_img': img.cpu().numpy(),
            'token_input_ids': text_token.input_ids.cpu().numpy(),
            'token_attention_mask': text_token.attention_mask.cpu().numpy()
        }

        ort_outs = ort_session.run(None, inputs)
        print(f"img_ort_outs: {ort_outs[0]}")

        # itm_output = model(img, text_token.input_ids, text_token.attention_mask)

        # # itm_output = model({"image": img, "text_input": txt}, match_head="itm")
        # print(f"itm_output_print: {itm_output}")
        # itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        # print(f"itm_scores: {itm_scores}")
        # print(f"itm_scores_shape: {itm_scores.shape}")
        # print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
        # end_time = time.time()
        # elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        # print(f"input prompt {prompt}")
        # print(f"blip2ITM infer took {elapsed_time:.2f} ms to complete.")

    if itc:
        itc_score = model({"image": img, "text_input": txt}, match_head='itc')
        print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

prompts = [
    "an image of a white cat",
    # Add more prompts here
]

for prompt in prompts:
    test_model_with_prompt(prompt)