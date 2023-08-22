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

device = "cuda" if torch.cuda.is_available() else "cpu"
src_path = '/data/xcao/code/uni_recognize_demo/algorithms/test_images/package_test.mp4'
yolo_model_path = '/data/xcao/code/uni_recognize_demo/algorithms/model_weights/last.pt'
frame = '/data/xcao/code/uni_recognize_demo/algorithms/test_images/16923479689894.png'
caption_font = ImageFont.truetype("/data/xcao/code/uni_recognize_demo/algorithms/miscellaneous/fonts/Arial.ttf", 20)
yolo_model = YOLO(yolo_model_path)
prompt_file_path = "/data/xcao/code/uni_recognize_demo/algorithms/config/prompt_custom.txt"
prompt_list = []
with open(prompt_file_path, "r") as prompt_file:
    for line in prompt_file:
        prompt_list.append(line.strip()) 
print(prompt_list)
beit3_model = BEIT3Model(prompt_list, device)
blip2_model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)
print('blip2_load_success...................')
target_cls = [792]
trackerManager = TrackerManager()

# Initialize video capture and get original frame rate
cap = cv2.VideoCapture(src_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Initialize VideoWriter for saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter('output.mp4', fourcc, frame_rate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

sample_rate = 15
frame_counter = 0
input_size = 384
rgb_color = (84, 198, 247)
ITM_prefix = "A series of images of "
frame_rate = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

while True:
    infer_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    pil_image_ori = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image_ori)
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
                    scores = beit3_model.infer_img(processed_img)
                    values, indices = torch.topk(scores, 1)
                    top1_index = indices[0, 0].item()
                    opencv_image_rgb = cv2.cvtColor(tracker.clipImg, cv2.COLOR_BGR2RGB)
                    img = vis_processors["eval"](Image.fromarray(opencv_image_rgb)).unsqueeze(0).to(device)
                    txt = text_processors["eval"](ITM_prefix + prompt_list[top1_index])
                    itm_output = blip2_model({"image": img, "text_input": txt}, match_head="itm")
                    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
                    item_score = itm_scores[:, 1].item()
                    call_back_score = "{:.2f}".format(item_score)
                    print(f"call_back_score: {call_back_score}")
                    x1, y1 = tracker.x1, tracker.y1
                    x2, y2 = tracker.x2, tracker.y2
                    draw.rectangle([(x1, y1), (x2, y2)], outline=rgb_color, width=2)
                    caption = prompt_list[top1_index]
                    caption_multi_line((x1, y1), caption,
                                                    pil_image_ori, caption_font, 
                                                    rgb_color, (0, 0), split_len=100)
    opencv_image = cv2.cvtColor(np.array(pil_image_ori), cv2.COLOR_RGB2BGR)
    video_out.write(opencv_image)


    infer_end_time = time.time()
    elapsed_time = (infer_end_time - infer_start_time) * 1000
    print(f"total infer elapse {elapsed_time:.2f} ms to complete.")

cap.release()
video_out.release()