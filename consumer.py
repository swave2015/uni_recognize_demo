import torch

class Consumer:
    def __init__(self, queue):
        self.queue = queue
        self.device = torch.device("cuda:0")

    def test_while(self):
        while True:
            print('111111')
            tensor = torch.randn(1000, 1000).to(self.device)
            result = (tensor + tensor).sum()
            print('res: ', result)
            continue
