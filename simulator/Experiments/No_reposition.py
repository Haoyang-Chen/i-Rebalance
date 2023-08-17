# -*-coding: utf-8 -*-
# @Time : 2022/7/31 09:53
# @Author : Chen Haoyang   SEU
# @File : Greedy.py
# @Software : PyCharm


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import seaborn as sns
from simulator.simulator import *

PLOT = False
DISPATCH = True

KAPPA = 8


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
                        PreList = self.PreList[vehicle.ID]
                        pre_idxs = [PreList[5], PreList[6], PreList[7], PreList[4], PreList[8], PreList[0],
                                    PreList[3], PreList[2], PreList[1]]
                        pre_idxs = np.array(pre_idxs)

                        idxs = pre_idxs.argsort()
                        s = 0
                        for i in idxs:
                            pre_idxs[i] = s
                            s += 1

                        idxs = idxs[::-1]

                    idxs = idxs[0:int(len(idxs) / 2):1]
                    np.random.shuffle(idxs)
                    action = idxs[0]
                    action_sta[action] += 1
                    self.DispatchNum += 1

                    vehicle.Cluster = self.Act2Cluster(action, vehicle.Cluster)

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
save_path = os.path.join("../", "Models", "DQN_npre", "_", "saves_episode", "VeRL0-" + args.name)

EPS = 6 * 10

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
