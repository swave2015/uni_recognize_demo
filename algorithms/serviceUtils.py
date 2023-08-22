import io
import base64
import torch

def base64_to_tensor(base64_string):
    buffer = io.BytesIO(base64.b64decode(base64_string))
    tensor = torch.load(buffer)
    return tensor