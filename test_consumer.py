import torch.multiprocessing as mp
import torch
from consumer import Consumer

if __name__ == "__main__":
    mp.set_start_method('fork')
    queue = mp.Queue()
    consumer = Consumer(queue)
    process = mp.Process(target=consumer.test_while)
    process.start()