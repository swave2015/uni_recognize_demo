from flask import Flask, jsonify, request
import torch.multiprocessing as mp
import torch
import time
from beit3_model import BEIT3Model
import io
import base64
import time

app = Flask(__name__)

# Define the worker function
def worker(data_queue, prompt_list, device):
    beit3_model = BEIT3Model(prompt_list)
    while True:
        item = data_queue.get()
        if item is None:  # End signal
            break
        tensor = item["tensor"].to(device).unsqueeze(0)
        print('process_tensor_shape: ', tensor.shape)
        frame_id = item["frame_id"]
        tracker_id = item["tracker_id"]
        infer_start_time = time.time()
        beit3_model.infer_img(tensor)
        infer_end_time = time.time()
        elapsed_time = (infer_end_time - infer_start_time) * 1000
        print(f"beit3 infer took {elapsed_time:.2f} ms to complete.")

        print(f"Worker received tensor with sum: {frame_id}. Metadata: {tracker_id}")

def base64_to_tensor(base64_string):
    buffer = io.BytesIO(base64.b64decode(base64_string))
    tensor = torch.load(buffer)
    return tensor

@app.route('/beit3', methods=['POST'])
def enqueue_item():
    tensor_data = request.json["tensor"]
    frame_id = request.json.get("frame_id", None)  # default to None if not provided
    tracker_id = request.json.get("tracker_id", None)  # default to None if not provided
    
    # Convert the base64 string back to a PyTorch tensor
    tensor = base64_to_tensor(tensor_data)

    # Put the data in the queue for processing
    data_queue.put({"tensor": tensor, "frame_id": frame_id, "tracker_id": tracker_id})
    
    return jsonify({"retcode": "0000"})


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')  # Recommended start method for PyTorch

    data_queue = mp.Queue()  # Create a queue

    # Start the worker process
    prompt_list = ['person', 'cat', 'dog', 'car', 'microwave']
    p = mp.Process(target=worker, args=(data_queue, prompt_list, device))
    p.start()

    try:
        app.run(host='0.0.0.0', port=3000)
    finally:
        q.put(None)  # Send end signal to worker
        p.join()     # Wait for worker process to finish
