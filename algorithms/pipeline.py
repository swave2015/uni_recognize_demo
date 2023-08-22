import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from beit3_model import BEIT3Model
from lavis.models import load_model_and_preprocess
import cv2
from TrackerManager import TrackerManager
from demo_utils import resize_or_pad
import time
from demo_utils import caption_multi_line
import numpy as np
import threading
import os
import shutil

class VideoAnalyzer:
    def __init__(self, src_path, yolo_model_path, prompt_file_path, frame, font_path, task_queue, device="cuda"):
        self.src_path = src_path
        self.yolo_model_path = yolo_model_path
        self.prompt_file_path = prompt_file_path
        self.frame = frame
        self.caption_font = ImageFont.truetype(font_path, 20)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        # Load models
        self.yolo_model = YOLO(self.yolo_model_path)
        self.beit3_model = BEIT3Model(self.device)
        self.blip2_model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=self.device, is_eval=True)
        print('BLIP2 model load success')
        self.target_cls = [792]
        self.caption_font = ImageFont.truetype("/data/xcao/code/uni_recognize_demo/algorithms/miscellaneous/fonts/Arial.ttf", 20)
        self.task_queue = task_queue

    def analyze_video(self, prompt_list, src_file, save_dir):
        print(f"src_file: {src_file}")
        print(f"prompt_list: {prompt_list}")
        tmp_dir_name = os.path.dirname(src_file)
        save_name = os.path.basename(src_file)
        save_name_without_ext = os.path.splitext(save_name)[0]
        output_path = os.path.join(save_dir, save_name_without_ext + '.mp4')
        tmp_save_path = os.path.join(tmp_dir_name, save_name_without_ext + '.mp4') 
        print(f"tmp_save_path: {tmp_save_path}")
        print(f"processed_save_path: {output_path}")
        self.beit3_model.infer_text(prompt_list)
        cap = cv2.VideoCapture(src_file)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        video_out = cv2.VideoWriter(tmp_save_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        sample_rate = 15
        frame_counter = 0
        input_size = 384
        rgb_color = (84, 198, 247)
        ITM_prefix = "A series of images of "
        trackerManager = TrackerManager()
        while True:
            infer_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            pil_image_ori = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image_ori)
            results = self.yolo_model.predict(frame, half=True, imgsz=640, conf=0.3, verbose=False)
            boxes = results[0].boxes
            target_boxes = []
            for box in boxes:
                if box.cls.cpu() in self.target_cls:
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
                    x1, y1 = tracker.x1, tracker.y1
                    x2, y2 = tracker.x2, tracker.y2
                    draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
                    if tracker.clipImg is not None and frame_counter % sample_rate == 0:
                        processed_img = resize_or_pad(tracker.clipImg, input_size)
                        scores = self.beit3_model.infer_img(processed_img)
                        values, indices = torch.topk(scores, 1)
                        top1_index = indices[0, 0].item()
                        opencv_image_rgb = cv2.cvtColor(tracker.clipImg, cv2.COLOR_BGR2RGB)
                        img = self.vis_processors["eval"](Image.fromarray(opencv_image_rgb)).unsqueeze(0).to(self.device)
                        print(f"top1_index_for_infer: {top1_index}")
                        txt = self.text_processors["eval"](ITM_prefix + prompt_list[top1_index])
                        itm_output = self.blip2_model({"image": img, "text_input": txt}, match_head="itm")
                        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                        item_score = itm_scores[:, 1].item()
                        caption = prompt_list[top1_index] + ' ' + "{:.2f}".format(item_score)
                        tracker.caption_show = caption
                        caption_multi_line((x1, y1), tracker.caption_show,
                                                        pil_image_ori, self.caption_font, 
                                                        rgb_color, (0, 0), split_len=100)
                    else:
                        caption_multi_line((x1, y1), tracker.caption_show,
                                                        pil_image_ori, self.caption_font, 
                                                        rgb_color, (0, 0), split_len=100)

            opencv_image = cv2.cvtColor(np.array(pil_image_ori), cv2.COLOR_RGB2BGR)
            infer_end_time = time.time()
            elapsed_time = (infer_end_time - infer_start_time) * 1000
            print(f"total infer elapse {elapsed_time:.2f} ms to complete.")
            video_out.write(opencv_image)
            frame_counter += 1
        
        cap.release()
        video_out.release()
        shutil.move(tmp_save_path, output_path)

    def analyze_task(self):
        while True:
            item = self.task_queue.get()
            file_dir = item[0]
            prompt_list = item[1]['person']
            save_dir = item[2]
            for filename in os.listdir(file_dir):
                 file_ext = os.path.splitext(filename)[1].lower()
                 if file_ext in ['.mp4', '.avi', '.mkv', '.ts', '.mov']:
                        video_path = os.path.join(file_dir, filename)
                        print(f"Found video: {video_path}")
                        self.analyze_video(prompt_list, video_path, save_dir)
                            
    def start_analyze(self):
        thread = threading.Thread(target=self.analyze_task)
        thread.start() 
        print("start_video_analyzing")
