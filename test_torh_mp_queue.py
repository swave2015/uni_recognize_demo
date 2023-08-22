import torch
import torch.multiprocessing as mp
import time

# Producer function: generate random tensors and place them in the queue
def producer(queue, num_tensors):
    for _ in range(num_tensors):
        tensor = torch.randn(2, 2)
        queue.put(tensor)
        time.sleep(0.1)  # Simulate some delay for producing the tensor
    queue.put(None)  # Sentinel value to signal the consumer to terminate

# Consumer function: retrieve tensors from the queue and compute their sum
def consumer(queue):
    tensor_sum = torch.zeros(2, 2)
    while True:
        tensor = queue.get()
        if tensor is None:  # Check for the sentinel value
            break
        tensor_sum += tensor
    print(f"Sum of tensors:\n{tensor_sum}")

def main():
    num_tensors = 5

    # Create a shared queue
    queue = mp.Queue()

    # Start the producer and consumer processes
    producer_process = mp.Process(target=producer, args=(queue, num_tensors))
    consumer_process = mp.Process(target=consumer, args=(queue,))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

if __name__ == '__main__':
    main()
