import sys
sys.path.append('./algorithms')
sys.path.append('./algorithms/LAVIS')

from task_monitor.upload_monitor import UploadMonitor
from algorithms.pipeline import VideoAnalyzer
import multiprocessing
import torch
from queue import Queue
    
if __name__ == "__main__":
    data_queue = Queue()
    monitor = UploadMonitor(data_queue)
    monitor.start_monitoring()

    analyzer = VideoAnalyzer(
        src_path='/data/xcao/code/uni_recognize_demo/algorithms/test_images/package_test.mp4',
        yolo_model_path='/data/xcao/code/uni_recognize_demo/algorithms/model_weights/last.pt',
        prompt_file_path="/data/xcao/code/uni_recognize_demo/algorithms/config/prompt_custom.txt",
        frame='/data/xcao/code/uni_recognize_demo/algorithms/test_images/16923479689894.png',
        font_path="/data/xcao/code/uni_recognize_demo/algorithms/miscellaneous/fonts/Arial.ttf",
        task_queue=data_queue
    )

    analyzer.start_analyze()