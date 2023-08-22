import torch
from PIL import Image
import time
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

raw_image = Image.open("/data1/caoxh/zero_short_action_vid/test_imgs/input.jpg").convert("RGB")

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

caption = "Series of images: a person sitting"
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

def test_model_with_prompt(prompt):
    txt = text_processors["eval"](prompt)
    start_time = time.time()
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    end_time = time.time()
    elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"input prompt {prompt}")
    print(f"blip2ITM infer took {elapsed_time:.2f} ms to complete.")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    print(f'The image and text are matched with a probability of {itm_scores[:, 1].item():.3%}')
    itc_score = model({"image": img, "text_input": txt}, match_head='itc')
    print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)

prompts = [
    "A series of images of person is standing",
    "A Collection of pictures of person is standing",
    "A Set of photos of person is standing",
    # Add more prompts here
]

for prompt in prompts:
    test_model_with_prompt(prompt)