import torch
from timm.models import create_model
import modeling_finetune
import utils
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image
import cv2
import time

class BEIT3Model:
    def __init__(self, 
                device):
        self.device = device
        self.input_size = 384
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        self.model_config = 'beit3_large_patch16_384_retrieval'
        self.model_path = '/data/xcao/code/uni_recognize_demo/algorithms/model_weights/beit3_large_patch16_384_coco_retrieval.pth'
        self.drop_path = 0.1
        self.vocab_size = 64010
        self.max_len = 64
        self.checkpoint_activations = None
        self.tokenizer = XLMRobertaTokenizer('/data/xcao/code/uni_recognize_demo/algorithms/model_weights/beit3.spm')
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        self.beit3_model = create_model(
            self.model_config,
            pretrained=False,
            drop_path_rate=self.drop_path,
            vocab_size=self.vocab_size,
            checkpoint_activations=self.checkpoint_activations,
        )

        self.load_model()
        self.beit3_model.to(self.device)
        self.beit3_model.eval()
        self.language_cls = None
        print('BEiT3 model load success')

    def load_model(self):
        utils.load_model_and_may_interpolate(self.model_path, self.beit3_model, 'model|module', '')
        
    def infer_text(self, prompt_list):
        prompt_features = []
        prompt_prefix = "a series of images of "
        for prompt in prompt_list:
            start_time = time.time()
            print(f"load prompt: {prompt}")
            tokens = self.tokenizer.tokenize(prompt_prefix + prompt)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids = [self.bos_token_id] + token_ids[:] + [self.eos_token_id]
            num_tokens = len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * (self.max_len - num_tokens)
            token_ids_tensor = torch.tensor(token_ids).to(self.device).unsqueeze(0)
            padding_mask = [0] * num_tokens + [1] * (self.max_len - num_tokens)
            padding_mask_tensor = torch.tensor(padding_mask).to(self.device).unsqueeze(0)
            _, language_cls = self.beit3_model(
                text_description=token_ids_tensor, 
                padding_mask=padding_mask_tensor, 
                only_infer=True
            )
            prompt_features.append(language_cls.detach().cpu()[0])
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"text infer took {elapsed_time:.2f} ms to complete.")

        self.language_cls = torch.stack(prompt_features)
    
    def infer_img(self, input_img):
        input_img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        input_img = self.transform(input_img).unsqueeze(0).to(self.device)
        vision_cls, _ = self.beit3_model(image=input_img, only_infer=True)
        vision_cls = vision_cls.cpu()
        scores = vision_cls @ self.language_cls.t()
        
        return scores

