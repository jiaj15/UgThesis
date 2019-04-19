import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import PARAMETER as p
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import os
import time

# parameter
# EPOCH=p.EPOCH
# BATCH_SIZE=p.BATCH_SIZE
# LR=p.LR

torch.manual_seed(1)


class CNNv2(nn.Module):
    def __init__(self):
        super(CNNv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                # input shape 2*12*12
                in_channels=3,
                out_channels=48,
                kernel_size=5,
                stride=1,
                padding=2
                # output shape 32,12,12

            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                # input shape 32, 12, 12
                in_channels=48,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1

                # output shape 64, 4 ,4

            ),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(57612, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 12 * p.WIDTH * 2)

        )

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        # print x1.size()
        x1 = self.conv2(x1)
        # print x1.size()
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), -1)
        # print x.size()
        # y=self.fc(x1,x2)
        y = self.fc(x)
        # print y.size()
        return y


class sumoDataSetv2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sumodata_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        print "init"

    def __len__(self):
        return len(self.sumodata_frame)

    def __getitem__(self, idx):
        # print "get items"
        vel_matrix_name = self.sumodata_frame.iloc[idx, 0]
        vel_matrix = pd.read_csv(vel_matrix_name, index_col=0, dtype=np.float64)
        Vmatrice = torch.from_numpy(vel_matrix.values)

        # Vmatrice = torch.reshape(torch.from_numpy(vel_matrix.values),[1,25,12])
        pos_matrix_name = self.sumodata_frame.iloc[idx, 1]
        pos_matrix = pd.read_csv(pos_matrix_name, index_col=0, dtype=np.float64)
        Pmatrice = torch.from_numpy(pos_matrix.values)

        tim_matrix_name = self.sumodata_frame.iloc[idx, 2]
        tim_matrix = pd.read_csv(tim_matrix_name, index_col=0, dtype=np.float64)
        Tmatrice = torch.from_numpy(tim_matrix.values)
        # Pmatrice = torch.reshape(torch.from_numpy(pos_matrix.values),[1,25,12])

        vehicle_matrix = torch.stack([Vmatrice, Pmatrice, Tmatrice])

        # print(vehicle_matrix)
        sam_vel_matrix_name = self.sumodata_frame.iloc[idx, 3]
        sam_vel_matrix = pd.read_csv(sam_vel_matrix_name, index_col=0, dtype=np.float64)
        SVmatrice = torch.from_numpy(sam_vel_matrix.values)

        # Vmatrice = torch.reshape(torch.from_numpy(vel_matrix.values),[1,25,12])
        sam_pos_matrix_name = self.sumodata_frame.iloc[idx, 4]
        sam_pos_matrix = pd.read_csv(sam_pos_matrix_name, index_col=0, dtype=np.float64)
        SPmatrice = torch.from_numpy(sam_pos_matrix.values)

        sam_tim_matrix_name = self.sumodata_frame.iloc[idx, 5]
        sam_tim_matrix = pd.read_csv(sam_tim_matrix_name, index_col=0, dtype=np.float64)
        STmatrice = torch.from_numpy(sam_tim_matrix.values)

        # Pmatrice = torch.reshape(torch.from_numpy(pos_matrix.values),[1,25,12])

        sam_vehicle_matrix = torch.stack([SPmatrice, STmatrice])

        signal_name = self.sumodata_frame.iloc[idx, 6]
        signal = pd.read_csv(signal_name, index_col=0, dtype=np.float64)
        signal_matrix = torch.from_numpy(signal.values)

        # image = io.imread(img_name)
        # landmarks = self.sumodata_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        #

        sample = {'vehicle': vehicle_matrix, 'Signal': signal_matrix, 'sample': sam_vehicle_matrix}

        if self.transform:
            sample = self.transform(sample)

        return sample


def trainNet(LR, BATCH_SIZE, EPOCH):
    net = CNNv2()
    cnn = net.double()
    validation_split = .2

    dataset = sumoDataSetv2(root_dir="data/output2/", csv_file="data/output2/sumo_trainingset.csv")
    # testset = sumoDataSet(root_dir="data/output/", csv_file="data/output/sumo_trainingset.csv")
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=valid_sampler)

    # train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
    #                     shuffle=True, num_workers=4)

    print 'data loaded'

    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    # loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    Losses = []
    ValidationLosses = []

    for i in range(EPOCH):
        train_loss = 0
        val_loss = 0
        j = 0

        for step, data in enumerate(train_loader, 0):
            vehicle_data = data['vehicle']
            signal_data = data['Signal']
            sample_data = data['sample']
            # print(sample_data.size())

            output = cnn(vehicle_data, signal_data)
            output = torch.reshape(output, [output.size()[0], 2, 12, p.WIDTH])
            loss = loss_func(output, sample_data)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # print loss.data.numpy()

            train_loss = train_loss + loss.data.numpy()

            j = j + 1
        Losses.append(train_loss / j)
        print train_loss / j
        j = 0
        # Losses.append(loss.data.numpy())
        for step, data in enumerate(validation_loader, 0):
            vehicle_data = data['vehicle']
            signal_data = data['Signal']
            sample_data = data['sample']
            # print(sample_data.size())

            output = cnn(vehicle_data, signal_data)
            output = torch.reshape(output, [output.size()[0], 2, 12, 50])
            loss = loss_func(output, sample_data)
            optimizer.zero_grad()  # clear gradients for this training step
            # loss.backward()  # backpropagation, compute gradients
            # optimizer.step()  # apply gradients
            # if step%2==0:
            #     test_output=cnn
            j = j + 1
            val_loss = val_loss + loss.data.numpy()
        print "testing"

        # print loss.data.numpy()
        ValidationLosses.append(val_loss / j)
        print val_loss / j

    torch.save(cnn.state_dict(), 'data/model/model2_params2.pkl')
    return Losses, ValidationLosses


def run_eval(partial_info):
    model = CNNv2()
    model.load_state_dict(torch.load('data/model2_params2_03.pkl'))
    model.eval()
    output = model(partial_info)
    output = torch.reshape(output, [2, 12, 50])

    return output


if __name__ == "__main__":
    # cnn=CNNv2()
    # X1=torch.tensor()
    # LRS = [0.3,0.1,0.01,0.001]
    # plt.figure()
    #
    # for LR in LRS:
    #     loss, val = trainNet(LR, p.BATCH_SIZE, 50)
    #     plt.plot(loss, label='training, learning rate='+str(LR))
    #     plt.plot(val,  label='testing , learning rate='+str(LR))
    #     # x = len(loss)
    #     # xo = len(val)
    #     # xn = range(x, x / xo)
    #     # plt.plot(range(x), loss, label='training')
    #     # plt.plot(xn, val, label='validation')
    # plt.legend()
    # plt.show()
    run_eval()
