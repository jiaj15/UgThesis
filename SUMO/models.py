import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import PARAMETER as p

#parameter
EPOCH=p.EPOCH
BATCH_SIZE=p.BATCH_SIZE
LR=p.LR

torch.manual_seed(1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size= 4,
                stride =2

            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size= 2,
                stride =1

            ),
            nn.ReLU()
        )



        Conv2d(1, 16, 2)
        self.conv2 = nn.Conv2d(16, 32, 1)






