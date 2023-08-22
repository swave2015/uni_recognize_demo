import sys
sys.path.insert(0, '../../multimodal_exp')

import grpc
from grpc_protos import beit3service_pb2
from grpc_protos import beit3service_pb2_grpc
import io
import torch
import time
import cv2
from demo_utils import resize_or_pad
from TrackerManager import TrackerManager
from ultralytics import YOLO
from concurrent import futures
from PIL import Image, ImageDraw, ImageFont
from demo_utils import caption_multi_line
import numpy as np
import json
from multiprocessing import Process, Manager, Queue
import redis


def display_frames(display_queue, redis_client):
    # ... Initialization code ...
    cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Video Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    rgb_color = (84, 198, 247)
    bgr_color = rgb_color[::-1]
    warm_up_frame_counter = 60
    caption_font = ImageFont.truetype("../miscellaneous/fonts/Arial.ttf", 20)
    tracker_id_event_dict = {}
    while True:
        if display_queue.qsize() < warm_up_frame_counter:
            continue
        else:
            frame_data = display_queue.get()
            ori_frame = frame_data[2]
            frame_id = frame_data[0]
            tracker_list = frame_data[1]
            pil_image = Image.fromarray(cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            for tracker in tracker_list:
                tracker_id = tracker[0]
                x1, y1, x2, y2 = tracker[1]
                draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
                result = redis_client.hget(frame_id, tracker_id)
                if result is None:
                    # Handle the case where there's no value associated with the keys in Redis
                    print(f"No data found for frame_id: {frame_id}, tracker_id: {tracker_id}")
                else:
                    event = result.decode('utf-8')
                    redis_client.hdel(frame_id, tracker_id)
                    tracker_id_event_dict.setdefault(tracker_id, event)
                    print('evetn_grpc_return: ', event)
                draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
                
                caption = tracker_id_event_dict.get(tracker_id, "None")
                caption_multi_line((x1, y1), caption,
                                                    pil_image, caption_font, 
                                                    rgb_color, (0, 0), split_len=100)
                

            current_tracker_ids = set(tracker[0] for tracker in tracker_list)
            keys_to_delete = [key for key in tracker_id_event_dict.keys() if key not in current_tracker_ids]
            for key in keys_to_delete:
                del tracker_id_event_dict[key]

            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            cv2.imshow('Video Feed', opencv_image)
            cv2.waitKey(100)

class GRPCClient:

    def __init__(self):
        self.channel = grpc.insecure_channel('192.168.31.28:5056')
        self.stub = beit3service_pb2_grpc.Beit3ServiceStub(self.channel)

    def send_data_to_server(self, image, frame_id, tracker_id):
        # Create the request
        request = beit3service_pb2.Beit3Request(
            image=image,
            frame_id=frame_id,
            tracker_id=tracker_id
        )

        # Start timing
        start_time = time.time()

        # Send the request
        response = self.stub.EnqueueItem(request)
        print('response_code: ', response)

        # End timing
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        # print(f"elapsed_time: {elapsed_time}")

        return response.retcode, elapsed_time
    
class ClientServiceServicer(beit3service_pb2_grpc.ClientServiceServicer):
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def ReceiveInferResult(self, request, context):
        # Process the received infer result
        print(f"infer result: {request.result}")
        parsed_data = json.loads(request.result)
        frame_id = parsed_data['frame_id']
        tracker_id = parsed_data['tracker_id']
        event = parsed_data['event']
        conf = parsed_data['conf']
        self.redis_client.hset(frame_id, tracker_id, event + ' ' + conf)
        # if frame_id not in self.callback:
        #     self.callback[frame_id] = {}
        # self.callback[frame_id][tracker_id] = event
        # print('self.callback: ', self.callback)
        
        return beit3service_pb2.Beit3Response(retcode="0000")

if __name__ == "__main__":
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    client_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    beit3service_pb2_grpc.add_ClientServiceServicer_to_server(ClientServiceServicer(redis_client), client_server)
    client_server.add_insecure_port('[::]:5050')
    client_server.start()
    display_queue = Queue()
    display_process = Process(target=display_frames, args=(display_queue, redis_client))
    display_process.start()
    client = GRPCClient()
    yolo_model_path = '/home/caoxh/multimodal_exp/models_weights/yolov8x.pt'
    yolo_model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    input_size = 384
    frame_counter = 0
    trackerManager = TrackerManager()
    sample_rate = 5
    target_cls = [0]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ori_frame = frame.copy()
        pil_image = Image.fromarray(cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB))
        
        results = yolo_model.predict(frame, half=True, imgsz=640, conf=0.3, verbose=False)
        boxes = results[0].boxes
        target_boxes = []
        for box in boxes:
            if box.cls.cpu() in target_cls:
                x1, y1, x2, y2 = box.xyxy.cpu().int()[0]
                x1 = x1.item()
                y1 = y1.item()
                x2 = x2.item()
                y2 = y2.item()
                target_boxes.append([x1, y1, x2, y2]) 
        trackerManager.update_trackers(target_boxes, merge=True)
        if frame_counter % sample_rate == 0:
            trackerManager.updateTrackerClipImg(frame)
            if len(trackerManager.trackers) > 0:
                for tracker in trackerManager.trackers:
                    if tracker.clipImg is not None:
                        processed_img = resize_or_pad(tracker.clipImg, input_size)
                        ret, buffer = cv2.imencode('.jpg', processed_img)
                        image = buffer.tobytes()
                        retcode, elapsed = client.send_data_to_server(image, frame_counter, tracker.id)
                        print(f"gRPC call elapsed: {elapsed:.2f} ms")
        tracker_id_list = []
        for tracker in trackerManager.trackers:
            x1, y1 = tracker.x1, tracker.y1
            x2, y2 = tracker.x2, tracker.y2
            tracker_tuple = (tracker.id, (x1, y1, x2, y2))
            tracker_id_list.append(tracker_tuple)
            # draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
            
            # if frame_counter in callback and tracker.id in callback[frame_counter]:
            #     tracker.caption_show = callback[frame_counter][tracker.id]

            # tracker_image_pil = caption_multi_line((x1, y1), tracker.caption_show,
            #                                     pil_image, caption_font, 
            #                                     rgb_color, (0, 0), split_len=10)
            
        # opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Video Feed', opencv_image)
        frame_data = (frame_counter, tracker_id_list, ori_frame)
        display_queue.put(frame_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1