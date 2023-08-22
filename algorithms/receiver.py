from flask import Flask, request, jsonify
import torch
import base64
import io
import multiprocessing
from beit3_model import BEIT3Model
import torch.multiprocessing as mp

app = Flask(__name__)

# Create a queue to hold data

device = "cuda" if torch.cuda.is_available() else "cpu"
data_queue = torch.multiprocessing.Queue()

def process_data(data_queue, prompt_list):
    # beit3_model = BEIT3Model(prompt_list)
    while True:
        print('into while')
        # print('data_queue: ', data_queue)
        print('process_data_qsize: ', data_queue.qsize())
        data = data_queue.get()
        if data is None:
            break  
        tensor = data["tensor"].cuda()
        print('process_tensor_shape: ', tensor.shape)
        frame_id = data["frame_id"]
        tracker_id = data["tracker_id"]
        # scores = beit3_model.infer_img(tensor)
        # print('out_scores: ', scores)
        
        # Here, you process the tensor and other data as needed
        # print(f"Processed tensor: {tensor}, Frame ID: {frame_id}, Tracker ID: {tracker_id}")

# Start the data processing process
prompt_list = ['person', 'cat', 'dog', 'car', 'microwave']

# data_process = multiprocessing.Process(target=process_data, args=(prompt_list, ))
# data_process.start()


def base64_to_tensor(base64_string):
    buffer = io.BytesIO(base64.b64decode(base64_string))
    tensor = torch.load(buffer)
    return tensor


@app.route("/beit3", methods=["POST"])
def receive_data_to_queue():
    tensor_data = request.json["tensor"]
    frame_id = request.json.get("frame_id", None)  # default to None if not provided
    tracker_id = request.json.get("tracker_id", None)  # default to None if not provided
    
    # Convert the base64 string back to a PyTorch tensor
    tensor = base64_to_tensor(tensor_data)

    # Put the data in the queue for processing
    data_queue.put({"tensor": tensor, "frame_id": frame_id, "tracker_id": tracker_id})
    print('data_queue_size_print: ', data_queue.qsize())
    return jsonify({"retcode": "0000"})



if __name__ == "__main__":
    # print(torch.multiprocessing.get_start_method())
    mp.set_start_method('spawn', force=True)
    p = mp.Process(target=process_data, args=(data_queue, prompt_list), daemon=True)
    p.start()
    try:
        app.run(host="0.0.0.0", port=3000)
    finally:
        # This will end the data processing loop and stop the process
        data_queue.put(None)
        p.join()
