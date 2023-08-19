import datetime as dt
import os
import pickle
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

batch_size = 64

l_r = 1e-3
K = 3
eps = 0.001

if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(4, 10, 2)
        self.conv2 = nn.Conv2d(10, 20, 2)
        self.fc1 = nn.Linear(80, 60)
        self.fc2 = nn.Linear(60, 40)

    def forward(self, x):
        input_size = x.size(0)  # input_size=batch_size

        x = self.conv1(x)  # in: batch*4*5*5, out: batch*10*4*4  (5-2+1)
        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=(2, 2), stride=1)  # in: batch*10*4*4, out: batch*10*3*3  (4-2+1)/1

        x = self.conv2(x)  # in: batch*10k*3*3, out: batch*20*2*2 (3-2+1)
        x = F.relu(x)

        x = x.view(input_size, -1)  # in: batch20*2*2, out: batch*80

        x = self.fc1(x)  # in: batch*80k  out:batch*60k
        x = F.relu(x)

        x = self.fc2(x)  # in:batch*60  out:batch*40
        return F.log_softmax(x, dim=1)


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

        self.fc3 = nn.Linear(25, 10)
        self.fc4 = nn.Linear(10, 6)

    def forward(self, x):
        # in: batch*25  out:batch*10
        x = self.fc3(x)
        x = F.relu(x)

        # in:batch*10 out:batch*6
        x = self.fc4(x)
        return F.relu(x)


class FrontNet(nn.Module):  # fc+cnn  batch*50
    def __init__(self):
        super(FrontNet, self).__init__()
        self.CNN = CNN()
        self.FC1 = FC1()
        self.FC2 = FC2()

    def forward(self, fs, fc):
        fm_d = fs[:, 0:25]
        ft = fs[:, 25:125]
        ft_data = torch.zeros(batch_size, 4, 5, 5)  # batchsize*channel*size*size
        clu_num = 0

        for i in range(25, 125, 4):
            row = int(clu_num / 5)
            col = int(clu_num % 5)

            ft_data[:, 0, row, col] = ft[:, i - 25]
            ft_data[:, 1, row, col] = ft[:, i + 1 - 25]
            ft_data[:, 2, row, col] = ft[:, i + 2 - 25]
            ft_data[:, 3, row, col] = ft[:, i + 3 - 25]
            clu_num += 1

        ft_data = ft_data.to(device)  # batchsize*channel*size*size
        fc = fc.to(device)  # batchsize*6
        fm_d = fm_d.to(device)  # batchsize*25

        first = self.CNN(ft_data)  # batch*40
        second = self.FC1(fc)  # batch*4
        third = self.FC2(fm_d)  # batch*6

        return torch.cat((first, second, third), 1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size  # feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)

        seq_len = input_seq.shape[1]

        # input(batch_size, seq_len, input_size)
        ipt_size = 50
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

        self.fn1 = FrontNet()
        self.fn2 = FrontNet()
        self.fn3 = FrontNet()

        self.LSTM = LSTM(input_size=50, hidden_size=64, num_layers=3, output_size=9, batch_size=batch_size)

    def forward(self, input_data):  # input_data=batchsize*665
        LSTM_input = torch.zeros(batch_size, K, 50)

        fs = input_data[:, 0:125]
        fc = input_data[:, 375:381]
        oc1 = self.fn1(fs, fc)
        fs = input_data[:, 125:250]
        fc = input_data[:, 381:387]
        oc2 = self.fn2(fs, fc)
        fs = input_data[:, 250:375]
        fc = input_data[:, 387:393]
        oc3 = self.fn3(fs, fc)
        # fs = input_data[:,375:500]
        # fc = input_data[:,643:649]
        # oc4 = self.fn4(fs, fc)
        # fs=input_data[:,500:625]
        # fc=input_data[:,649:655]
        # oc5=self.fn5(fs,fc)

        # lstm
        LSTM_input[:, 0, :] = oc1
        LSTM_input[:, 1, :] = oc2
        LSTM_input[:, 2, :] = oc3
        # LSTM_input[:,3,:] = oc4
        # LSTM_input[:,4,:] = oc5

        LSTM_input = LSTM_input.to(device)  # batch*seq*feature
        output = self.LSTM(LSTM_input)
        output = F.softmax(output, dim=1)

        return output


def timestamp_datetime(value):
    d = datetime.fromtimestamp(value)
    t = dt.datetime(d.year, d.month, d.day, d.hour, d.minute, 0)
    return t


def string_datetime(value):
    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def string_pdTimestamp(value):
    d = string_datetime(value)
    t = pd.Timestamp(d.year, d.month, d.day, d.hour, d.minute)
    return t


def return_timestamp(tim):
    s_t = time.strptime(tim, " %Y/%m/%d %H:%M:%S")
    timeStamp = int(time.mktime(s_t))
    return timeStamp


def ReadMap(input_file_path):
    reader = pd.read_csv(input_file_path, chunksize=1000)
    Map = []
    for chunk in reader:
        Map.append(chunk)
    Map = pd.concat(Map)
    Map = Map.drop(["Unnamed:0"], axis=1)
    Map = Map.values
    Map = Map.astype('int64')
    return Map


def ReadCostMap(input_file_path):
    reader = pd.read_csv(input_file_path, header=None, chunksize=1000)
    Map = []
    for chunk in reader:
        Map.append(chunk)
    Map = pd.concat(Map)
    return Map


def ReadPath(input_file_path):
    reader = pd.read_csv(input_file_path, chunksize=1000)
    Path = []
    for chunk in reader:
        Path.append(chunk)
    Path = pd.concat(Path)
    Path = Path.drop(["Unnamed: 0"], axis=1)
    Path = Path.values
    Path = Path.astype('int64')
    return Path


def ReadNode(input_file_path):
    reader = pd.read_csv(input_file_path, chunksize=1000)
    Node = []
    for chunk in reader:
        Node.append(chunk)
    Node = pd.concat(Node)
    return Node


def ReadNodeIDList(input_file_path):
    NodeIDList = []
    with open(input_file_path, 'r') as f:
        data = f.readlines()

        for line in data:
            odom = line.split()
            odom = int(odom[0])
            NodeIDList.append(odom)
    return NodeIDList


def ReadOrder(input_file_path):
    reader = pd.read_csv(input_file_path, chunksize=1000)
    Order = []
    for chunk in reader:
        Order.append(chunk)
    Order = pd.concat(Order)
    Order = Order.drop(
        columns=['End_time', 'PointS_Longitude', 'PointS_Latitude', 'PointE_Longitude', 'PointE_Latitude'])
    Order["Start_time"] = Order["Start_time"].apply(timestamp_datetime)

    Order_Num = 25000
    Order.drop(labels=list(range(Order_Num, len(Order))), axis=0, inplace=True)

    Order = Order.sort_values(by="Start_time")
    Order["ID"] = range(0, Order.shape[0])
    Order = Order.values
    return Order


def ReadResetOrder(input_file_path):
    reader = pd.read_csv(input_file_path, chunksize=1000)
    Order = []
    for chunk in reader:
        Order.append(chunk)
    Order = pd.concat(Order)
    Order = Order.values
    return Order


def Creat_Obey(driver_num):
    outcome = np.zeros(driver_num)
    bound = [0.0583, 0.0117, 0.3058, 0.3867, 0.3970, 0.5912, 0.6990, 0.7638, 0.8759, 0.9407]
    for i in range(driver_num):
        coin = random.randrange(0, 10000, 1) / 10000
        for j in range(10):
            if coin < bound[j]:
                outcome[i] = j * 0.1
                break
            if j == 9:
                outcome[i] = 1
    return outcome


def ReadDriver(input_file_path):
    reader = pd.read_csv(input_file_path, chunksize=1000)
    Driver = []
    for chunk in reader:
        Driver.append(chunk)
    Driver = pd.concat(Driver)
    Driver["Start_time"] = [return_timestamp(i) for i in Driver["Start_time"]]
    Driver["Start_time"] = Driver["Start_time"].apply(timestamp_datetime)
    Rand_Obey = Creat_Obey(Driver.shape[0])
    Driver.insert(loc=3, column="Obey", value=Rand_Obey)
    Driver = Driver.values
    return Driver


def ReadRidePreference(input_file_path):
    RP = pd.read_csv(os.path.join(input_file_path, 'RidePreference.csv'), header=None).values
    return RP


def ReadPreNet(input_file_path):
    ClassNet0 = MyNets()
    ClassNet0 = torch.load(os.path.join(input_file_path, 'Pre_Class0.pt'), map_location=torch.device('cpu'))
    ClassNet0 = ClassNet0.to(device)

    ClassNet1 = MyNets()
    ClassNet1 = torch.load(os.path.join(input_file_path, 'Pre_Class1.pt'), map_location=torch.device('cpu'))
    ClassNet1 = ClassNet1.to(device)

    ClassNet2 = MyNets()
    ClassNet2 = torch.load(os.path.join(input_file_path, 'Pre_Class2.pt'), map_location=torch.device('cpu'))
    ClassNet2 = ClassNet2.to(device)

    ClassNet3 = MyNets()
    ClassNet3 = torch.load(os.path.join(input_file_path, 'Pre_Class3.pt'), map_location=torch.device('cpu'))
    ClassNet3 = ClassNet3.to(device)

    ClassNet4 = MyNets()
    ClassNet4 = torch.load(os.path.join(input_file_path, 'Pre_Class4.pt'), map_location=torch.device('cpu'))
    ClassNet4 = ClassNet4.to(device)

    ClassNetList = [ClassNet0, ClassNet1, ClassNet2, ClassNet3, ClassNet4]

    return ClassNetList


def ReadClassList(input_file_path):
    ClassList = pd.read_csv(os.path.join(input_file_path, 'clustering_outcome.csv'), header=None).values[:, 1]
    return ClassList


def ReadHomeLocation(input_file_path):
    HomeLocation = pd.read_csv(os.path.join(input_file_path, 'home_info.csv')).values
    driver_id = HomeLocation[:, 0]
    lo_la = HomeLocation[:, 1:3]
    return lo_la


def ReadPoILocation(input_file_path):
    file = open(os.path.join(input_file_path, 'poi_location.pkl'), 'rb')
    PoILocation = pickle.load(file)
    return PoILocation


def ReadVisFre(input_file_path):
    VisFre = pd.read_csv(os.path.join(input_file_path, 'DriverFamilarity.csv'), header=None).values
    return VisFre


def ReadAllFiles():
    NodePath = os.path.join(os.path.dirname(sys.path[0]), "data", "Node.csv")
    NodeIDListPath = os.path.join(os.path.dirname(sys.path[0]), "data", "NodeIDList.txt")

    path = os.path.join(os.path.dirname(sys.path[0]), "data")
    order_set = []

    for file in os.listdir(os.path.join(path, "Order_List")):
        if file[-3:] == "csv":
            order_set.append(file)

    OrdersPath = os.path.join(os.path.dirname(sys.path[0]), "data", "Order_List", random.choice(order_set))
    VehiclesPath = os.path.join(os.path.dirname(sys.path[0]), "data", "Vehicle_List", 'Vehicle_List') + OrdersPath[-8:]

    print("using: ", OrdersPath, VehiclesPath)
    MapPath = os.path.join(os.path.dirname(sys.path[0]), "data", "AccurateMap.csv")
    PrePath = os.path.join(os.path.dirname(sys.path[0]), 'data', 'PreData')

    Node = ReadNode(NodePath)
    NodeIDList = ReadNodeIDList(NodeIDListPath)
    Orders = ReadOrder(OrdersPath)
    Vehicles = ReadDriver(VehiclesPath)
    Map = ReadCostMap(MapPath)

    ClassNetList = ReadPreNet(PrePath)
    ClassDict = ReadClassList(PrePath)
    HomeLocation = ReadHomeLocation(PrePath)
    PoILocation = ReadPoILocation(PrePath)
    VisFreDict = ReadVisFre(PrePath)
    RidePreference = ReadRidePreference(PrePath)

    return Node, NodeIDList, Orders, Vehicles, Map, ClassNetList, ClassDict, HomeLocation, PoILocation, VisFreDict, RidePreference


def ReadAllFiles_TEST(idx):
    NodePath = os.path.join(os.path.dirname(sys.path[0]), "data", "Node.csv")
    NodeIDListPath = os.path.join(os.path.dirname(sys.path[0]), "data", "NodeIDList.txt")

    path = os.path.join(os.path.dirname(sys.path[0]), "data")
    order_set = []

    for file in os.listdir(os.path.join(path, "Order_List")):
        if file[-3:] == "csv":
            order_set.append(file)

    OrdersPath = os.path.join(os.path.dirname(sys.path[0]), "data", "Order_List", order_set[idx])
    VehiclesPath = os.path.join(os.path.dirname(sys.path[0]), "data", "Vehicle_List", 'Vehicle_List') + OrdersPath[-8:]

    print("using: ", OrdersPath, VehiclesPath)
    MapPath = os.path.join(os.path.dirname(sys.path[0]), "data", "AccurateMap.csv")
    PrePath = os.path.join(os.path.dirname(sys.path[0]), 'data', 'PreData')

    Node = ReadNode(NodePath)
    NodeIDList = ReadNodeIDList(NodeIDListPath)
    Orders = ReadOrder(OrdersPath)
    Vehicles = ReadDriver(VehiclesPath)
    Map = ReadCostMap(MapPath)

    ClassNetList = ReadPreNet(PrePath)
    ClassDict = ReadClassList(PrePath)
    HomeLocation = ReadHomeLocation(PrePath)
    PoILocation = ReadPoILocation(PrePath)
    VisFreDict = ReadVisFre(PrePath)
    RidePreference = ReadRidePreference(PrePath)

    return Node, NodeIDList, Orders, Vehicles, Map, ClassNetList, ClassDict, HomeLocation, PoILocation, VisFreDict, RidePreference
