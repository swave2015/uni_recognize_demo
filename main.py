from upload_monitor import UploadMonitor
import multiprocessing

if __name__ == "__main__":
    queue = multiprocessing.Queue()
    monitor = UploadMonitor(queue)
    monitor.start_monitoring()
    while True:
        item = queue.get()
        print(item)