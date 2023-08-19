import torchvision
import torch
import numpy as np
import os
import pickle
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
from torch.utils.tensorboard import SummaryWriter
import scipy.stats
import math

device = torch.device('cuda')

batch_size = 64
K=3     # sequence length


driver_class=4

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(4, 10, 2)
        self.conv2 = nn.Conv2d(10, 20, 2)
        #4*5*5->20*2*2->80*1->40*1

        self.fc1 = nn.Linear(80, 60)
        self.fc2 = nn.Linear(60, 40)

    def forward(self, x):
        input_size = x.size(0)   # input_size=batch_size

        x = self.conv1(x)  # in: batch*4*5*5, out: batch*10*4*4  (5-2+1)
        x = F.relu(x)

        x = F.max_pool2d(x,kernel_size = (2,2),stride=1)  # in: batch*10*4*4, out: batch*10*3*3  (4-2+1)/1

        x = self.conv2(x)  # in: batch*10k*3*3, out: batch*20*2*2 (3-2+1)
        x = F.relu(x)

        x = x.view(input_size,-1)  # in: batch20*2*2, out: batch*80

        x = self.fc1(x)  # in: batch*80k  out:batch*60k
        x = F.relu(x)

        x = self.fc2(x)        # in:batch*60  out:batch*40
        return F.log_softmax(x,dim=1)


class FC1(nn.Module):
    def __init__(self):
        super(FC1, self).__init__()

        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):

        # in: batch*8  out:batch*6
        x = self.fc1(x)
        x = F.relu(x)

        # in:batch*6 out:batch*4
        x = self.fc2(x)
        return F.relu(x)


class FC2(nn.Module):
    def __init__(self):
        super(FC2, self).__init__()

        self.fc3 = nn.Linear(25, 10)  # 从（2+23）*1->n*1
        self.fc4 = nn.Linear(10, 6)


    def forward(self, x):

        # in: batch*25  out:batch*10
        x = self.fc3(x)
        x = F.relu(x)

        # in:batch*10 out:batch*6
        x = self.fc4(x)
        return F.relu(x)

class FrontNet(nn.Module):  #fc+cnn  batch*50
    def __init__(self):
        super(FrontNet, self).__init__()
        self.CNN = CNN()
        self.FC1 = FC1()
        self.FC2= FC2()

    def forward(self,fs,fc):
        fm_d = fs[:, 0:25]
        ft = fs[:, 25:125]
        ft_data = torch.zeros(batch_size, 4, 5, 5) #batchsize*channel*size*size
        clu_num = 0

        for i in range(25, 125, 4):
            row = int(clu_num / 5)
            col = int(clu_num % 5)

            ft_data[:, 0, row, col] = ft[:, i - 25]
            ft_data[:, 1, row, col] = ft[:, i + 1 - 25]
            ft_data[:, 2, row, col] = ft[:, i + 2 - 25]
            ft_data[:, 3, row, col] = ft[:, i + 3 - 25]
            clu_num += 1

        #GPU加速
        ft_data=ft_data.cuda()   #batchsize*channel*size*size
        fc=fc.cuda()             #batchsize*6
        fm_d=fm_d.cuda()         #batchsize*25

        first=self.CNN(ft_data) #batch*40
        second=self.FC1(fc) #batch*4
        third=self.FC2(fm_d) #batch*6

        return torch.cat((first, second, third), 1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size      # feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # 初始参数设置
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)

        seq_len = input_seq.shape[1]  # 5=K

        # input(batch_size, seq_len, input_size)
        ipt_size=50
        input_seq = input_seq.view(self.batch_size, seq_len, ipt_size)

        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))  # output(batch，5，1*64)
        output = output.contiguous().view(self.batch_size * seq_len, self.hidden_size)
        pred = self.linear(output)  # pred(batch*5, 9)
        pred = pred.view(self.batch_size, seq_len, -1)  # (batch*5*9)

        pred = pred[:, -1, :]  # (batch*9)
        return pred


class MyNets(nn.Module):
    def __init__(self):
        super(MyNets, self).__init__()

        self.fn1=FrontNet()
        self.fn2=FrontNet()
        self.fn3 = FrontNet()


        self.LSTM = LSTM(input_size=50, hidden_size=64, num_layers=3, output_size=9, batch_size=batch_size)


    def forward(self, input_data):  # input_data=batchsize*665
        LSTM_input = torch.zeros(batch_size,K,50)


        fs=input_data[:,0:125]
        fc=input_data[:,375:381]
        oc1=self.fn1(fs,fc)
        fs = input_data[:,125:250]
        fc = input_data[:,381:387]
        oc2 = self.fn2(fs, fc)
        fs = input_data[:,250:375]
        fc = input_data[:,387:393]
        oc3 = self.fn3(fs, fc)

        #lstm
        LSTM_input[:,0,:]=oc1
        LSTM_input[:,1,:] = oc2
        LSTM_input[:,2,:] = oc3


        LSTM_input=LSTM_input.cuda()  # batch*seq*feature
        output = self.LSTM(LSTM_input)
        output=F.softmax(output,dim=1)

        return output


if __name__=='__main__':
    #tensorboard
    #writer = SummaryWriter('logs')

    #preparing testing set
    file=open('data/LSTMTestingSetClass'+str(driver_class)+'.pkl','rb')
    
    raw_data=pickle.load(file)
    print('Read All Files')

    total_num=len(raw_data[0])
    input=np.zeros((total_num,131*K),dtype=float)
    output=np.zeros((total_num,9*K+1),dtype=float)
    for i in range(total_num):
        fs=raw_data[0][i]
        fc=raw_data[1][i]

        input[i,0:125*K]=fs
        input[i,125*K:131*K]=fc

        label=[]
        for seq in range(K):
            temp=raw_data[2][i][seq][0:9]
            label.extend(temp)

        output[i, 0:9*K] =label
        output[i,9*K]=raw_data[2][i][seq][9]


    inp=torch.from_numpy(input)
    out=torch.from_numpy(output)
    inp=inp.to(torch.float32)
    out=out.to(torch.float32)
    dataset = TensorDataset(inp, out)
    print("Start Testing...")

    net=MyNets()
    net=torch.load('Pre_Class'+str(driver_class)+'.pt')

    
    net=net.cuda()

    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    test_sample_num=0
    acc1_num=0
    acc2_num=0
    acc3_num=0
    acc4_num=0
    acc5_num=0

    pre_action = np.zeros(shape=(72, 9))
    real_action= np.zeros(shape=(72, 9))
    action_list=[0]*9
    reala_list=[0]*9

    for i_batch, batch_data in enumerate(data_loader):
        if len(batch_data[1]) < batch_size:
            break

        outcome= net(batch_data[0])
        
        label_temp = batch_data[1][:, 9*(K-1):9*K]
        real_action_list = batch_data[1][:, 9*K]  # ground truth

        for iter in range(batch_size):
            label = label_temp[iter]
            real_a = real_action_list[iter]
            label_a=torch.argmax(label_temp[iter]).item()

            pred = outcome[iter].tolist()
            #print(pred)
            a_hat = pred.index(max(pred))

            if a_hat==real_a:
                acc1_num+=1

            pred=np.array(pred)
            index_list=np.argsort(-pred)

            if real_a.item() in index_list[0:3]:
                acc3_num+=1
            
            if real_a.item() in index_list[0:2]:
                acc2_num+=1

            if real_a.item() in index_list[0:4]:
                acc4_num+=1

            if real_a.item() in index_list[0:5]:
                acc5_num+=1
                

            grid = batch_data[0][iter, 131*K-4]  #grid stores real cluster id
            grid_num = grid.item()
            pre_action[int(grid_num)][int(a_hat)] += 1
            real_action[int(grid_num)][int(real_a)] += 1
            action_list[a_hat]+=1
            reala_list[int(real_a.item())]+=1



        test_sample_num += batch_size
    print(real_action)
    print(pre_action)
    print('real')
    print(reala_list)
    print('predict')
    print(action_list)
    print('Acc@1='+str(float(acc1_num/test_sample_num)))
    print('Acc@2='+str(float(acc2_num/test_sample_num)))
    print('Acc@3=' + str(float(acc3_num / test_sample_num)))
    print('Acc@4=' + str(float(acc4_num / test_sample_num)))
    print('Acc@5=' + str(float(acc5_num / test_sample_num)))
