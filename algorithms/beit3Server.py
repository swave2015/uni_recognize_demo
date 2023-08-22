import grpc
from concurrent import futures
from grpc_protos import beit3service_pb2
from grpc_protos import beit3service_pb2_grpc
from serviceUtils import base64_to_tensor
import torch
from beit3_model import BEIT3Model
import torch.multiprocessing as mp
import time
import io
import numpy as np
import cv2
import json
from lavis.models import load_model_and_preprocess
from PIL import Image


class CallBackClient:

    def __init__(self):
        self.channel = grpc.insecure_channel('192.168.31.244:5050')
        self.stub = beit3service_pb2_grpc.ClientServiceStub(self.channel)

    def callback(self, result):
        # Create the request
        request = beit3service_pb2.CallBackResult(
            result=result
        )

        # Start timing
        start_time = time.time()

        # Send the request
        response = self.stub.ReceiveInferResult(request)
        print('response_code: ', response)

        # End timing
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        # print(f"elapsed_time: {elapsed_time}")

        return response.retcode, elapsed_time

def worker(data_queue, prompt_list, device0, device1):
    beit3_model = BEIT3Model(prompt_list, device0)
    blip2_model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device1, is_eval=True)
    ITM_prefix = "A series of images of "
    print('BLIP2 model load success')
    callbackClient = CallBackClient()
    while True:
        item = data_queue.get()
        if item is None:  # End signal
            break
        # tensor = item["tensor"].to(device).unsqueeze(0)
        image = item["image"]
        # print('process_tensor_shape: ', tensor.shape)
        frame_id = item["frame_id"]
        tracker_id = item["tracker_id"]
        infer_start_time = time.time()
        scores = beit3_model.infer_img(image)
        values, indices = torch.topk(scores, 1)
        top1_index = indices[0, 0].item()
        print(f"infer_result: {prompt_list[top1_index]}")
        infer_end_time = time.time()
        elapsed_time = (infer_end_time - infer_start_time) * 1000
        print(f"beit3 infer took {elapsed_time:.2f} ms to complete.")

        infer_start_time = time.time()
        opencv_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(opencv_image_rgb)
        img = vis_processors["eval"](pil_image).unsqueeze(0).to(device1)
        txt = text_processors["eval"](ITM_prefix + prompt_list[top1_index])
        itm_output = blip2_model({"image": img, "text_input": txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        item_score = itm_scores[:, 1].item()
        call_back_score = "{:.2f}".format(item_score)
        infer_end_time = time.time()
        elapsed_time = (infer_end_time - infer_start_time) * 1000
        print(f"blip2 infer took {elapsed_time:.2f} ms to complete.")


        callbackData = {
            "frame_id": frame_id,
            "tracker_id": tracker_id,
            "event": prompt_list[top1_index],
            "conf": call_back_score
        }

       
        callback_json = json.dumps(callbackData)
        callbackClient.callback(callback_json)


class Beit3ServiceServicer(beit3service_pb2_grpc.Beit3ServiceServicer):
    
    def __init__(self, data_queue):
        self.data_queue = data_queue

    def EnqueueItem(self, request, context):
        # serialized_tensor  = request.serialized_tensor
        image = np.frombuffer(request.image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # tensor_data = torch.load(io.BytesIO(serialized_tensor))
        frame_id = request.frame_id
        tracker_id = request.tracker_id

        # print(f"tensor: {serialized_tensor}, frame_id: {frame_id}, tracker_id: {tracker_id}")

        # tensor = base64_to_tensor(tensor_data)
        self.data_queue.put({"image": image, "frame_id": frame_id, "tracker_id": tracker_id})

        return beit3service_pb2.Beit3Response(retcode="0000")

if __name__ == "__main__":
    device0 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')
    data_queue = mp.Queue()
    prompt_file_path = "/data1/caoxh/zero_short_action_vid/config/prompt_custom.txt"
    prompt_list = []
    with open(prompt_file_path, "r") as prompt_file:
        for line in prompt_file:
            prompt_list.append(line.strip()) 

    print(prompt_list)
    # prompt_list = ['person is standing', 'person is working', 'person is walking', 'car', 'microwave', 
    #                'person is waving hands', 'person is delivering pakcages', 'person is sitting on a chair',
    #                'person is eating']
    p = mp.Process(target=worker, args=(data_queue, prompt_list, device0, device1))
    p.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    beit3service_pb2_grpc.add_Beit3ServiceServicer_to_server(Beit3ServiceServicer(data_queue), server)
    server.add_insecure_port('[::]:5056')
    server.start()
    server.wait_for_termination()
