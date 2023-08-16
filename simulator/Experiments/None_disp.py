# -*-coding: utf-8 -*-
# @Time : 2022/4/5 10:36
# @Author : Chen Haoyang   SEU
# @File : Test_raw.py
# @Software : PyCharm

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import torch.nn as nn
import torch
import seaborn as sns
from simulator.simulator import *
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plot

PLOT = False


def TEST(self, a, reject_rate, avg_wait, idxx):
    self.Reset()
    EpisodeStartTime = dt.datetime.now()
    self.RealExpTime = self.Orders[0].ReleasTime  # - self.TimePeriods
    self.NowOrder = self.Orders[0]

    EndTime = self.Orders[-1].ReleasTime  # + 3 * self.TimePeriods

    sum = 0
    step = 0

    while self.RealExpTime <= EndTime:

        # print("step: ", step)
        # print("time: ", self.RealExpTime)
        # print(self.RealExpTime)

        SOV = 0

        self.UpdateFunction()
        # for cluster in self.Clusters:
        #     print(len(cluster.IdleVehicles))
        vehicles = []
        for cluster in self.Clusters:
            for v in cluster.IdleVehicles:
                vehicles.append(v)
            cluster.IdleVehicles = []
        for order in self.Orders:
            if len(vehicles) == 0:
                break
            if self.RealExpTime < order.ReleasTime < self.RealExpTime + self.TimePeriods:
                o_cluster = self.NodeID2Cluseter[order.PickupPoint]
                vehicles[0].LocationNode = order.PickupPoint
                vehicles[0].Cluster = o_cluster
                o_cluster.IdleVehicles.append(vehicles[0])
                vehicles = vehicles[1::]
        if len(vehicles):
            self.Clusters[0].IdleVehicles += vehicles

        self.MatchFunction()
        if PLOT:
            demand_dest = np.zeros((self.NumGrideHeight, self.NumGrideWidth), dtype=int)
            # supply_mat = np.zeros((self.NumGrideHeight, self.NumGrideWidth), dtype=int)
            # for cluster in self.Clusters:
            #     supply_mat[int(cluster.ID / self.NumGrideWidth)][cluster.ID % self.NumGrideWidth] = int(
            #         len(cluster.IdleVehicles))
            #     cluster.IdleVehicles.clear()
            for order in self.Orders:
                if self.RealExpTime - self.TimePeriods < order.ReleasTime < self.RealExpTime:
                    c_id = self.NodeID2Cluseter[order.DeliveryPoint].ID
                    demand_dest[int(c_id / self.NumGrideWidth)][c_id % self.NumGrideWidth] += 1
            plt.cla()
            plt.clf()
            plt.figure(figsize=(21, 14))
            plt.title(str(self.RealExpTime) + '----' + str(self.RealExpTime + self.TimePeriods))
            sns.heatmap(demand_dest, center=0, annot=True, linewidths=.5, fmt="d", cbar=False, cmap='Blues')
            plt.xlabel('demand_dest')
            plt.savefig(str(self.RealExpTime) + '.png')

        self.RealExpTime += self.TimePeriods
        for i in self.Orders:
            if self.RealExpTime - self.TimePeriods < i.ReleasTime < self.RealExpTime:
                if i.ArriveInfo != "Reject":
                    SOV += self.GetValue(i.OrderValue)

        # writer_non.add_scalar('Order_Value', 2*SOV, step)
        a[idxx][step] += SOV
        step += 1

    SumOrderValue = 0
    OrderValueNum = 0
    for i in self.Orders:
        if i.ArriveInfo != "Reject":
            SumOrderValue += self.GetValue(i.OrderValue)
            OrderValueNum += 1

    print("----------------------------Experiment over----------------------------")
    print("Number of Reject: " + str(self.RejectNum))
    print("Number of order: " + str(self.OrderNum))
    print("Reject rate: " + str(self.RejectNum / self.OrderNum))
    reject_rate[idxx] = self.RejectNum / self.OrderNum
    print("Total Order value: " + str(SumOrderValue))
    if (len(self.Orders) - self.RejectNum) != 0:
        print("Average waiting time: " + str(self.TotallyWaitTime / (len(self.Orders))))
        avg_wait[idxx] = self.TotallyWaitTime / (len(self.Orders))
    return


DispatchMode = "Simulation"
DemandPredictionMode = "None"
ClusterMode = "Grid"

writer_non = SummaryWriter('cc/不进行调度/')
TOV = np.zeros((1, 108))
reject_rate = np.zeros(1)
avg_wait = np.zeros(1)
for i in range(1):
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
    EXPSIM.CreateAllInstantiate()
    TEST(EXPSIM, TOV, reject_rate, avg_wait, i)

TOV = TOV.sum(0)
TOV = [int(TOV[x] // 1 + (TOV[x] % 1 > 0)) for x in range(108)]
step_TOV = [TOV[x] for x in range(108)]
hour_TOV = [TOV[6 * x] + TOV[6 * x + 1] + TOV[6 * x + 2] + TOV[6 * x + 3] + TOV[6 * x + 4] + TOV[6 * x + 5] for x in
            range(18)]

print("step_TOV: ", step_TOV)
print("hour_TOV: ", hour_TOV)
print("day_TOV: ", sum(hour_TOV))
print("reject_rate: ", sum(reject_rate))
print("avg_wait: ", sum(avg_wait))
