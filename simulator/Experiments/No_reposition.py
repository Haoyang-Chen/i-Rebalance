# -*-coding: utf-8 -*-
# @Time : 2022/7/31 09:53
# @Author : Chen Haoyang   SEU
# @File : Greedy.py
# @Software : PyCharm


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import torch.nn as nn
import torch
import seaborn as sns
from simulator.simulator import *
from tensorboardX import SummaryWriter
from haversine import haversine
from sklearn.preprocessing import minmax_scale

PLOT = False
DISPATCH = True

KAPPA = 8


def Act2Cluster(self, action, cluster):
    """
    return a cluster given an action(0-8) and current cluster
    """
    rand_cluster = cluster.Neighbor[random.randrange(0, len(cluster.Neighbor))]
    try:
        if action <= 2:
            dest = self.Clusters[cluster.ID + self.NumGrideWidth - 1 + action]
        elif 2 < action <= 5:
            dest = self.Clusters[cluster.ID - 4 + action]
        elif 5 < action <= 8:
            dest = self.Clusters[cluster.ID - self.NumGrideWidth - 7 + action]
        else:
            raise 'wrong action'
    except:
        dest = rand_cluster

    if cluster.ID < self.NumGrideWidth and 5 < action <= 8:
        dest = rand_cluster
    if cluster.ID >= self.NumGrideWidth * (self.NumGrideHeight - 1) and action <= 2:
        dest = rand_cluster
    if cluster.ID % self.NumGrideWidth == 0 and action % 3 == 0:
        dest = rand_cluster
    if cluster.ID % self.NumGrideWidth == self.NumGrideWidth - 1 and action % 3 == 2:
        dest = rand_cluster

    return dest


def Getidx(Cluster_ID, Goal_ID):
    index = Cluster_ID - Goal_ID
    if abs(index) > 10:
        return 4
    C2ADict = {0: 4, -1: 5, 1: 3, 9: 7, -9: 1, 10: 6, 8: 8, -8: 0, -10: 2}

    return C2ADict[index]


def get_pre_old(self, vehicle):  # 司机

    driver_id = vehicle.ID
    cluster_id = vehicle.Cluster.ID
    origin_id = cluster_id
    # center_id = self.DriverClusteringInst[driver_id]
    center_id = 0
    # center_id = random.randrange(0,10)
    center_info = self.DriverClusteringData[int(center_id)]

    outcome = [0] * 9
    outcome[4] = center_info[self.NumGrideHeight - 1 - int(cluster_id / self.NumGrideWidth)][
        int(cluster_id % self.NumGrideWidth)]

    C2ADict = {0: 4, -1: 5, 1: 3, 9: 7, -9: 1, 10: 6, 8: 8, -8: 0, -10: 2}

    for c in vehicle.Cluster.Neighbor:
        cluster_id = c.ID
        index = C2ADict[origin_id - cluster_id]
        outcome[index] = center_info[self.NumGrideHeight - 1 - int(cluster_id / self.NumGrideWidth)][
            int(cluster_id % self.NumGrideWidth)]

    return outcome


def DemandPredictFunction(self):
    """
    Here you can implement your own order forecasting method
    to provide efficient and accurate help for Dispatch method
    传当前time slot内对应的订单数量
    """
    self.DemandExpect = torch.zeros(self.ClustersNumber)
    DE = torch.zeros(self.ClustersNumber)
    for order in self.Orders:
        if self.RealExpTime + self.TimePeriods <= order.ReleasTime < self.RealExpTime + 2 * self.TimePeriods:
            self.DemandExpect[self.NodeID2Cluseter[order.PickupPoint].ID] += 1
            cluster = self.NodeID2Cluseter[order.PickupPoint]
            rodedist = self.RoadDist(order.PickupPoint, order.DeliveryPoint)
            value = self.GetValue(rodedist)
            cluster.potential += value
        if self.RealExpTime <= order.ReleasTime <= self.RealExpTime + self.TimePeriods:
            DE[self.NodeID2Cluseter[order.PickupPoint].ID] += 1

    self.DE_mat = DE.reshape((1, 1, self.NumGrideHeight, self.NumGrideWidth))
    trans = torch.nn.Conv2d(1, 1, (5, 5), stride=1, padding=2, bias=False)
    trans.weight.data = torch.Tensor([[[[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1]]]])
    self.DE_mat = trans(self.DE_mat)
    self.DE_mat = self.DE_mat.view(-1, self.DE_mat.shape[2] * self.DE_mat.shape[3]).squeeze()
    return self.DE_mat


def TEST(self, rand, pre, a, reject_rate, avg_wait, pre_reject, num_disp, idxx):
    self.CreateAllInstantiate_TEST(idxx // int(EPS / 6))
    self.Reset()
    EpisodeStartTime = dt.datetime.now()
    self.RealExpTime = self.Orders[0].ReleasTime
    self.NowOrder = self.Orders[0]

    total_reward = 0
    self.NowOrder = self.Orders[0]

    EndTime = self.Orders[-1].ReleasTime

    reject = 0

    step = 0

    action_sta = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    while self.RealExpTime <= EndTime:
        # print(self.RealExpTime)

        SOV = 0

        self.UpdateFunction()
        self.MatchFunction()
        self.SupplyExpectFunction()
        self.DemandPredictFunction(step)
        self.IdleTimeCounterFunction()
        self.Refresh_Pre()

        if PLOT:
            demand_mat = np.zeros((self.NumGrideHeight, self.NumGrideWidth), dtype=int)
            supply_mat = np.zeros((self.NumGrideHeight, self.NumGrideWidth), dtype=int)

            for cluster in self.Clusters:
                demand_mat[int(self.NumGrideHeight - 1 - cluster.ID // self.NumGrideWidth)][
                    cluster.ID % self.NumGrideWidth] = int(
                    self.DemandExpect[cluster.ID])
                supply_mat[int(self.NumGrideHeight - 1 - cluster.ID // self.NumGrideWidth)][
                    cluster.ID % self.NumGrideWidth] = self.SupplyExpect[
                                                           cluster.ID] + len(
                    cluster.IdleVehicles)

            plt.cla()
            plt.clf()
            plt.figure(figsize=(21, 14))
            plt.subplot(2, 3, 1)
            plt.title(str(self.RealExpTime) + '----' + str(self.RealExpTime + self.TimePeriods))
            sns.heatmap(demand_mat, center=0, annot=True, linewidths=.5, fmt="d", cbar=False, cmap='Blues')
            plt.xlabel('Demand_Before')
            plt.subplot(2, 3, 2)
            sns.heatmap(supply_mat, center=0, annot=True, linewidths=.5, fmt="d", cbar=False, cmap='OrRd')
            plt.xlabel('Supply_Before')
            plt.subplot(2, 3, 3)
            sns.heatmap(supply_mat - demand_mat, center=0, annot=True, linewidths=.5, fmt="d", cbar=False,
                        cmap='icefire')
            plt.xlabel('S-D_Before')

        if DISPATCH:
            for cluster in self.Clusters:

                if cluster.ID % 9 == 0:
                    self.Refresh_Pre()

                for vehicle in cluster.IdleVehicles:

                    if pre:
                        if NEW_PRE:
                            PreList = self.PreList[vehicle.ID]
                            pre_idxs = [PreList[5], PreList[6], PreList[7], PreList[4], PreList[8], PreList[0],
                                        PreList[3], PreList[2], PreList[1]]
                        else:
                            pre_idxs = get_pre_old(self, vehicle)
                        # pre_idxs = [round(x, 3) for x in pre_idxs]
                        pre_idxs = np.array(pre_idxs)

                        idxs = pre_idxs.argsort()
                        s = 0
                        for i in idxs:
                            pre_idxs[i] = s
                            s += 1

                        idxs = idxs[::-1]
                        # print('raw_pre: ',pre_idxs)
                        pre_idxs = minmax_scale(pre_idxs, (0, 1))

                        # print('pre_idxs: ',pre_idxs)
                        # print('action rank: ',idxs)

                    # print(pre_idxs)
                    idxs = idxs[0:int(len(idxs) / 2):1]
                    np.random.shuffle(idxs)
                    action = idxs[0]
                    action_sta[action] += 1
                    self.DispatchNum += 1

                    vehicle.Cluster = Act2Cluster(self, action, vehicle.Cluster)

                    if vehicle.Cluster == cluster:
                        self.StayNum += 1

                    RandomNode = random.choice(vehicle.Cluster.Nodes)
                    RandomNode = RandomNode[0]
                    vehicle.DeliveryPoint = RandomNode

                    ScheduleCost = self.RoadCost(vehicle.LocationNode, RandomNode)
                    self.TotallyDispatchCost += ScheduleCost

                    vehicle.Cluster.VehiclesArrivetime[vehicle] = min(self.RealExpTime + np.timedelta64(
                        MINUTES * ScheduleCost), self.RealExpTime + self.TimePeriods)

                    self.SupplyExpect[vehicle.Cluster.ID] += 1

                cluster.IdleVehicles.clear()

        self.RealExpTime += self.TimePeriods

        if PLOT:
            demand_dest = np.zeros((self.NumGrideHeight, self.NumGrideWidth), dtype=int)
            for order in self.Orders:
                if self.RealExpTime < order.ReleasTime < self.RealExpTime + self.TimePeriods:
                    c_id = self.NodeID2Cluseter[order.DeliveryPoint].ID
                    demand_dest[int(c_id / self.NumGrideWidth)][c_id % self.NumGrideWidth] += 1

            for cluster in self.Clusters:
                supply_mat[int(self.NumGrideHeight - 1 - cluster.ID // self.NumGrideWidth)][
                    cluster.ID % self.NumGrideWidth] = self.SupplyExpect[
                    cluster.ID]

            plt.subplot(2, 3, 4)
            sns.heatmap(demand_dest, center=0, annot=True, linewidths=.5, fmt="d", cbar=False, cmap='Blues')
            plt.xlabel('Demand_Dest')
            plt.subplot(2, 3, 5)
            sns.heatmap(supply_mat, center=0, annot=True, linewidths=.5, fmt="d", cbar=False, cmap='OrRd')
            plt.xlabel('Supply_After')
            plt.subplot(2, 3, 6)
            sns.heatmap(supply_mat - demand_mat, center=0, annot=True, linewidths=.5, fmt="d", cbar=False,
                        cmap='icefire')
            plt.xlabel('S-D_After')
            plt.savefig(str(self.RealExpTime) + '.png')

        for i in self.Orders:
            if self.RealExpTime - self.TimePeriods <= i.ReleasTime < self.RealExpTime:
                if i.ArriveInfo != "Reject":
                    SOV += self.GetValue(i.OrderValue)

        a[idxx][step] += SOV
        step += 1

    SumOrderValue = 0
    OrderValueNum = 0
    for i in self.Orders:
        if i.ArriveInfo != "Reject":
            SumOrderValue += self.GetValue(i.OrderValue)
            OrderValueNum += 1
        else:
            self.TotallyWaitTime += 20

    s = sum(action_sta)
    print(action_sta)
    action_sta = [x / s for x in action_sta]
    print("action distribution: ", action_sta)
    print("Total Order value: " + str(SumOrderValue))
    print("----------------------------Experiment over----------------------------")
    if DISPATCH:
        print('Total reward: ', total_reward)
        avg_reward = total_reward / self.DispatchNum
        self.AVGreward = avg_reward
        print('Average reward: ', avg_reward)
        print('preference reject rate: ', reject / self.DispatchNum)
        pre_reject[idxx] = reject / self.DispatchNum

    print("Number of Reject: " + str(self.RejectNum))
    print("Reject rate: " + str(self.RejectNum / self.OrderNum))
    reject_rate[idxx] = self.RejectNum / self.OrderNum

    if DISPATCH:
        print("Number of Dispatch: " + str(self.DispatchNum - self.StayNum))
        num_disp[idxx] = self.DispatchNum - self.StayNum
        print('Number of Stay; ' + str(self.StayNum))

    if self.DispatchNum != 0:
        print("Average Dispatch Cost: " + str(self.TotallyDispatchCost / self.DispatchNum))
    if (len(self.Orders) - self.RejectNum) != 0:
        print("Average waiting time: " + str(self.TotallyWaitTime / (len(self.Orders))))
        avg_wait[idxx] = self.TotallyWaitTime / (len(self.Orders))

    print("Total Order value: " + str(SumOrderValue))
    return


DispatchMode = "Simulation"
DemandPredictionMode = "None"
ClusterMode = "Grid"

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
parser.add_argument("-n", "--name", default='test00', help="Name of the run")
args = parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")
save_path = os.path.join("../","Models","DQN_npre", "_", "saves_episode", "VeRL0-" + args.name)
# writer_pre_rand = SummaryWriter('cc/Greedy_司机有偏好/')

EPS=6*10

TOV = np.zeros((EPS, 108))
reject_rate = np.zeros(EPS)
avg_wait = np.zeros(EPS)
pre_reject = np.zeros(EPS)
num_disp = np.zeros(EPS)
for i in range(EPS):
    EXPSIM = Simulation(
        ClusterMode=ClusterMode,
        DemandPredictionMode=DemandPredictionMode,
        DispatchMode=DispatchMode,
        VehiclesNumber=500,
        TimePeriods=TIMESTEP,
        LocalRegionBound=LocalRegionBound,
        SideLengthMeter=SideLengthMeter,
        VehiclesServiceMeter=VehiclesServiceMeter,
        NeighborCanServer=NeighborCanServer,
        FocusOnLocalRegion=FocusOnLocalRegion
    )

    rand = False
    pre = True
    TEST(EXPSIM, rand, pre, TOV, reject_rate, avg_wait, pre_reject, num_disp, i)

TOV = TOV.sum(0)
TOV /= EPS
TOV = [int(TOV[x] // 1 + (TOV[x] % 1 > 0)) for x in range(108)]
step_TOV = [TOV[x] for x in range(108)]
hour_TOV = [TOV[6 * x] + TOV[6 * x + 1] + TOV[6 * x + 2] + TOV[6 * x + 3] + TOV[6 * x + 4] + TOV[6 * x + 5] for x in
            range(18)]
print("step_TOV: ", step_TOV)
print("hour_TOV: ", hour_TOV)
print("day_TOV: ", sum(hour_TOV))
print("reject_rate: ", sum(reject_rate) / EPS)
print("avg_wait: ", sum(avg_wait) / EPS)
print("pre_reject: ", sum(pre_reject) / EPS)
print("num_dispatch: ", sum(num_disp) / EPS)

# step_TOV:  [2417, 4949, 5047, 4342, 3272, 3578, 4061, 4571, 3640, 3669, 3706, 3894, 4246, 4379, 4840, 5549, 5593, 5447, 5372, 5947, 4946, 5626, 5073, 5536, 5919, 5342, 4950, 5809, 5986, 5802, 6104, 5879, 6091, 6268, 5557, 6925, 5799, 6361, 5605, 5740, 7386, 5793, 6254, 6785, 7314, 6185, 8408, 5493, 6280, 6584, 7066, 5532, 6848, 6268, 6271, 6383, 7521, 7175, 7328, 6333, 6639, 5400, 6393, 5842, 6261, 6259, 5418, 6863, 6321, 6544, 6912, 5887, 6828, 6000, 6498, 7015, 5535, 6330, 6662, 6361, 6203, 6270, 7038, 5953, 6312, 5984, 5715, 6291, 5413, 6342, 6370, 5885, 6312, 6512, 6342, 5444, 5651, 6883, 6574, 5827, 4665, 6282, 5450, 6784, 5648, 4623, 5233, 1089]
# hour_TOV:  [23605, 25249, 24871, 23464, 22791, 23225, 23541, 23726, 23534, 24734, 26614, 28501, 30054, 31180, 32748, 32854, 32931, 32411]
# day_TOV:  486033
# reject_rate:  0.48821365784322274
# avg_wait:  0.9645781643625269
# pre_reject:  0.047092344615637924
# num_dispatch:  3451.85
