import sys
sys.path.append('./algorithms')

from task_monitor.upload_monitor import UploadMonitor
from algorithms.pipeline import VideoAnalyzer
import multiprocessing
import torch
from queue import Queue
    
if __name__ == "__main__":
    # main()
    # mp.set_start_method('forkserver')
    data_queue = Queue()
    # # data_queue = multiprocessing.Queue()
    monitor = UploadMonitor(data_queue)
    monitor.start_monitoring()

    # for task in monitor.tasks:
    #     process = mp.Process(target=monitor.wait_for_all_uploads_to_complete, args=(task,))
    # process.start()


    # for task in monitor.tasks:
    #     process = mp.Process(target=monitor.wait_for_all_uploads_to_complete, args=(task,))
    #     process.start()

    # def start_monitoring(self):
    #     mp.set_start_method('spawn')        
    #     for task in self.tasks:
    #         process = mp.Process(target=self.wait_for_all_uploads_to_complete, args=(task,))
    #         process.start()
    #         print(f"start task monitor: {task['type']}")
    # monitor.start_monitoring()
    
    analyzer = VideoAnalyzer(
        src_path='/data/xcao/code/uni_recognize_demo/algorithms/test_images/package_test.mp4',
        yolo_model_path='/data/xcao/code/uni_recognize_demo/algorithms/model_weights/last.pt',
        prompt_file_path="/data/xcao/code/uni_recognize_demo/algorithms/config/prompt_custom.txt",
        frame='/data/xcao/code/uni_recognize_demo/algorithms/test_images/16923479689894.png',
        font_path="/data/xcao/code/uni_recognize_demo/algorithms/miscellaneous/fonts/Arial.ttf",
        task_queue=data_queue
    )

    analyzer.start_analyze()
    # process = mp.Process(target=analyzer.analyze_task, args=(1,))
    # process.start()
    # # monitor.start_monitoring()
    # print("start_video_analyzing")