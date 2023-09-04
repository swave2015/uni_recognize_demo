import onnxruntime as ort
import cv2
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image
from transformers import XLMRobertaTokenizer
import torch

test_img_path = '/data/xcao/code/uni_recognize_demo/test_misc/out_frames/frame_2.png'
test_prompt = ['a picture of a dog', 'a picture of a cat']
img_onnx_path = '/data/xcao/code/uni_recognize_demo/algorithms/onnx_export/beit3_retrival_coco_img_constant.onnx'
text_onnx_path = '/data/xcao/code/uni_recognize_demo/algorithms/onnx_export/beit3_retrival_coco_text_constant.onnx'
input_size = 384
input_img = cv2.imread(test_img_path)
transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
input_img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
input_img = transform(input_img).unsqueeze(0).numpy()
print(f"input_img_shape: {input_img.shape}")
ort_session = ort.InferenceSession(img_onnx_path)
inputs = {
    'input_img': input_img
}
ort_outs = ort_session.run(None, inputs)
print(f"img_ort_outs: {ort_outs[0].shape}")

tokenizer = XLMRobertaTokenizer('/data/xcao/code/uni_recognize_demo/algorithms/model_weights/beit3.spm')
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
max_len = 64

def get_tokens(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    max_len = 64
    token_ids = [bos_token_id] + token_ids[:] + [eos_token_id]
    num_tokens = len(token_ids)
    token_ids = token_ids + [pad_token_id] * (max_len - num_tokens)
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).numpy()
    num_tokens = len(token_ids)
    token_ids = token_ids + [pad_token_id] * (max_len - num_tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
    padding_mask_tensor = torch.tensor(padding_mask).unsqueeze(0).numpy()

    return token_ids_tensor, padding_mask_tensor


ort_session_text = ort.InferenceSession(text_onnx_path)

token_ids_tensor0,  padding_mask_tensor0 = get_tokens(test_prompt[0])
token_ids_tensor1,  padding_mask_tensor1 = get_tokens(test_prompt[1])
input_names = [input.name for input in ort_session_text.get_inputs()]
inputs_text0 = {
    'input_text': token_ids_tensor0,
    'input_mask': padding_mask_tensor0
}
ort_outs_text0 = ort_session_text.run(None, inputs_text0)

inputs_text1 = {
    'input_text': token_ids_tensor1,
    'input_mask': padding_mask_tensor1
}
ort_outs_text1 = ort_session_text.run(None, inputs_text1)

score0 = ort_outs[0] @ ort_outs_text0[0].T
score1 = ort_outs[0] @ ort_outs_text1[0].T
print(f"score0: {score0}, score1: {score1}")