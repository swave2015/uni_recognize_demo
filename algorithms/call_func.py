import requests
import torch
import base64
import io

def tensor_to_base64(tensor):
    """Convert a PyTorch tensor to a base64 string."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Sample tensor data
tensor = torch.randn((1, 3, 384, 384))  # Replace with your tensor data if needed
tensor_base64 = tensor_to_base64(tensor)

data = {
    "tensor": tensor_base64,
    "frame_id": "sample_frame_id",  # Optional
    "tracker_id": "sample_tracker_id"  # Optional
}

# Send a POST request
response = requests.post("http://127.0.0.1:3000/beit3", json=data)
print(response.json())
