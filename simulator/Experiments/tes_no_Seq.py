# -*-coding: utf-8 -*-
# @Time : 2022/5/6 19:50 下午
# @Author : Chen Haoyang   SEU
# @File : RL_algo.py
# @Software : PyCharm
import random
import sys
import os

# sys.path.append(os.path.dirname(sys.path[0]))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from simulator.simulator import *

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from tensorboardX import SummaryWriter
from haversine import haversine

import numpy as np
import argparse
from sklearn.preprocessing import minmax_scale

EPISODES = 10000

HIDDEN_SIZE = 64

ENTROPY_BETA = 0.01

KAPPA = 8
GAMMA = 0

EPSILON = 0.8
EPSILON_EP = 100

LEARNING_RATE = 0.001

BATCH_SIZE = 10


def DispatchFunction(self, state, actor):
    self.DispatchNum += 1
    state_v = torch.tensor(state, dtype=torch.float32).to(device)
    output = actor(state_v)
    action_probs = F.softmax(output, dim=0)
    action = torch.multinomial(action_probs, 1).item()
    return action


class Disp_Actor(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * 2, output_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


def Disp_GetStateFunction(self, cluster, pre_idxs):
    DemandExpect = np.array(self.DemandExpect)
    DemandExpect = np.array(np.random.normal(DemandExpect, DemandExpect * 0.2) + 0.5, dtype=int)
    DemandExpect = np.maximum(DemandExpect, 0)

    state = np.zeros((KAPPA + 1) * 2)
    neighbour = cluster.Neighbor
    state[8] = int(self.SupplyExpect[cluster.ID]) - int(DemandExpect[cluster.ID])
    state[9] = int(pre_idxs[4])
    for nc in neighbour:
        id = nc.ID
        idx = self.Getidx(cluster.ID, id)
        state[idx * 2] = int(self.SupplyExpect[id]) - int(DemandExpect[id])
        state[idx * 2 + 1] = pre_idxs[idx]
    return state


def RewardFunction(self, vehicle, cluster, pre_idxs, action):
    supply = int(self.SupplyExpect[vehicle.Cluster.ID]) - 1
    demand = int(self.DemandExpect[vehicle.Cluster.ID])
    gap = demand - supply
    diff = [gap, int(self.DemandExpect[cluster.ID]) - int(self.SupplyExpect[cluster.ID])]
    for nc in cluster.Neighbor:
        if nc.ID == vehicle.Cluster.ID:
            continue
        s = int(self.SupplyExpect[nc.ID])
        d = int(self.DemandExpect[nc.ID])
        diff.append(d - s)
    mean = sum(diff) / len(diff)
    std = statistics.stdev(diff)
    balance_reward = (gap - mean) / (std + 0.01)
    pre_reward = pre_idxs[action]
    return 2 * balance_reward + pre_reward


def move(self, vehicle, action, cluster, pre_idxs, idxs):
    vehicle.Cluster = self.Act2Cluster(action, vehicle.Cluster)

    num_supply = self.SupplyExpect[vehicle.Cluster.ID]
    num_demand = self.DemandExpect[vehicle.Cluster.ID]
    if num_supply > num_demand:
        expected_income = vehicle.Cluster.potential / num_supply
    else:
        expected_income = vehicle.Cluster.potential / num_demand

    idx = self.Getidx(cluster.ID, vehicle.Cluster.ID)
    ranking = minmax_scale(pre_idxs, (1, 9))
    ranking = [10 - x for x in ranking]
    ranking = ranking[idx]

    result = -0.3892 * ranking + 0.2932 * expected_income - 1.3959 + 1.8379 * vehicle.Obey
    vehicle.Auto = False
    incentive = 0
    if result < 0:
        vehicle.Auto = True
        incentive += (0 - result) / 0.2932
        idxs = idxs[0:int(len(idxs) / 2):1]
        self.reject += 1
        np.random.shuffle(idxs)
        vehicle.Cluster = self.Act2Cluster(idxs[0], cluster)

    if vehicle.Cluster == cluster:
        self.StayExpect[cluster.ID] += 1

    RandomNode = random.choice(vehicle.Cluster.Nodes)
    RandomNode = RandomNode[0]
    vehicle.DeliveryPoint = RandomNode

    ScheduleCost = self.RoadCost(vehicle.LocationNode, RandomNode)
    self.TotallyDispatchCost += self.RoadDist(vehicle.LocationNode, RandomNode)

    vehicle.Cluster.VehiclesArrivetime[vehicle] = min(self.RealExpTime + np.timedelta64(
        MINUTES * ScheduleCost), self.RealExpTime + self.TimePeriods)

    self.SupplyExpect[vehicle.Cluster.ID] += 1
    return incentive


def TEST(self, a, reject_rate, avg_wait, pre_reject, num_disp, idxx, reject_rate_h, idle_taxi_h,
         order_num_h, reward_h, num_pre_reject, AOV, disp_cost, incentives):
    disp_actor = Disp_Actor().to(device)
    disp_actor_name = 'episode_4999actor'
    disp_actor_path = os.path.join(save_path2, disp_actor_name)
    disp_actor.load_state_dict(torch.load(disp_actor_path, map_location=torch.device('cpu')))

    seq = [31, 40, 21, 22, 23, 30, 32, 39, 41, 48, 49, 50, 11, 12, 13, 14, 15, 20, 24, 29, 33, 38, 42, 47, 51, 56, 57,
           58, 59, 60, 1, 2, 3, 4, 5, 6, 7, 10, 16, 19, 25, 28, 34, 37, 43, 46, 52, 55, 61, 64, 65, 66, 67, 68, 69, 70,
           0, 8, 9, 17, 18, 26, 27, 35, 36, 44, 45, 53, 54, 62, 63, 71]

    self.CreateAllInstantiate_TEST(idxx // int(EPS / 6))
    self.Reset()
    EpisodeStartTime = dt.datetime.now()
    self.RealExpTime = self.Orders[0].ReleasTime
    self.NowOrder = self.Orders[0]

    total_reward = 0
    step = 0
    self.reject = 0
    total_batch_reward = 0

    EndTime = self.Orders[-1].ReleasTime

    incentive = 0
    while self.RealExpTime <= EndTime:
        SOV = 0

        self.UpdateFunction()
        self.MatchFunction()
        self.SupplyExpectFunction()
        self.DemandPredictFunction(step)
        self.IdleTimeCounterFunction()
        self.Refresh_Pre()

        pre_rej_count = 0
        cluster_counter = 0
        step_reward = 0

        for idx in seq:
            cluster = self.Clusters[idx]

            cluster_counter += 1
            if cluster_counter % 9 == 0:
                self.Refresh_Pre()

            random.shuffle(cluster.IdleVehicles)

            for vehicle in cluster.IdleVehicles:
                idle_taxi_h[idxx][step] += 1
                PreList = self.PreList[vehicle.ID]
                pre_idxs = np.array(
                    [PreList[5], PreList[6], PreList[7], PreList[4], PreList[8], PreList[0], PreList[3],
                     PreList[2], PreList[1]])

                idxs = pre_idxs.argsort()
                s = 0
                for i in idxs:
                    pre_idxs[i] = s
                    s += 1

                idxs = idxs[::-1]
                pre_idxs = minmax_scale(pre_idxs, (0, 1))

                disp_state = Disp_GetStateFunction(self, cluster, pre_idxs)
                disp_action = DispatchFunction(self, disp_state, disp_actor)
                ince = move(self, vehicle, disp_action, cluster, pre_idxs, idxs)
                if ince != 0:
                    incentive += ince
                    pre_rej_count += 1
                reward = RewardFunction(self, vehicle, cluster, pre_idxs, disp_action)

                if vehicle.Cluster == cluster:
                    self.StayNum += 1

                total_reward += reward
                step_reward += reward
                reward_h[idxx][step] += reward

            cluster.IdleVehicles.clear()

        num_pre_reject[idxx][step] += pre_rej_count
        self.RealExpTime += self.TimePeriods

        count = 0
        for i in self.Orders:
            if self.RealExpTime - self.TimePeriods <= i.ReleasTime < self.RealExpTime:
                if i.ArriveInfo != "Reject":
                    SOV += self.GetValue(i.OrderValue)
                    reject_rate_h[idxx][step] += 1
                count += 1
        reject_rate_h[idxx][step] = (count - reject_rate_h[idxx][step]) / count
        order_num_h[idxx][step] += count

        a[idxx][step] += SOV
        step += 1

    SumOrderValue = 0
    OrderValueNum = 0
    for i in self.Orders:
        if i.ArriveInfo != "Reject":
            SumOrderValue += self.GetValue(i.OrderValue)
            OrderValueNum += 1

    print("Total Order value: " + str(SumOrderValue))
    print('Auto Order Value: ' + str(self.AutoOrderValue))
    print('Dispatch Value: ' + str(SumOrderValue - self.AutoOrderValue))
    AOV[idxx] += SumOrderValue - self.AutoOrderValue
    print('Incentive: ', incentive)
    incentives[idxx] += incentive
    print("----------------------------Experiment over----------------------------")

    print('Total reward: ', total_reward)
    avg_reward = total_reward / self.DispatchNum
    self.AVGreward = avg_reward
    print('Average reward: ', avg_reward)
    print('preference reject rate: ', self.reject / self.DispatchNum)
    pre_reject[idxx] = self.reject / self.DispatchNum

    print("Number of Orders: " + str(len(self.Orders)))
    print("Number of Reject: " + str(self.RejectNum))
    print("Reject rate: " + str(self.RejectNum / self.OrderNum))
    reject_rate[idxx] = self.RejectNum / self.OrderNum

    print("Number of Dispatch: " + str(self.DispatchNum - self.StayNum))
    num_disp[idxx] = self.DispatchNum - self.StayNum
    print('Number of Stay; ' + str(self.StayNum))

    if self.DispatchNum != 0:
        print("Average Dispatch Cost: " + str(self.TotallyDispatchCost / self.DispatchNum))
        print("Dispatch Cost: " + str(self.TotallyDispatchCost))
        disp_cost[idxx] += self.TotallyDispatchCost
    if (len(self.Orders) - self.RejectNum) != 0:
        print("Average waiting time: " + str(self.TotallyWaitTime / (len(self.Orders))))
        avg_wait[idxx] = self.TotallyWaitTime / (len(self.Orders))

    print("Total Order value: " + str(SumOrderValue))
    print("Episode Run time : " + str(dt.datetime.now() - EpisodeStartTime))
    return


DispatchMode = "Simulation"
DemandPredictionMode = "None"
ClusterMode = "Grid"

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
parser.add_argument("-n", "--name", default='A2C_test', help="Name of the run")
args = parser.parse_args()
device = torch.device("cuda:2" if args.cuda else "cpu")
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_path = os.path.join("../","Models","A2C_pre_nsr", "_", "saves_episode", "Seq-" + args.name)
save_path2 = os.path.join("../","Models","A2C_pre_nsr", "_", "saves_episode", "Disp-" + args.name)
# os.makedirs(save_path, exist_ok=True)

EPS = 6 * 10

TOV = np.zeros((EPS, 108))
AOV = np.zeros(EPS)

num_pre_reject = np.zeros((EPS, 108))

reject_rate_h = np.zeros((EPS, 108))
reject_rate = np.zeros(EPS)

idle_taxi_h = np.zeros((EPS, 108))
order_num_h = np.zeros((EPS, 108))

reward_h = np.zeros((EPS, 108))

avg_wait = np.zeros(EPS)
pre_reject = np.zeros(EPS)
num_disp = np.zeros(EPS)

disp_cost = np.zeros(EPS)
incentives = np.zeros(EPS)

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
        FocusOnLocalRegion=FocusOnLocalRegion,
    )

    TEST(EXPSIM, TOV, reject_rate, avg_wait, pre_reject, num_disp, i, reject_rate_h, idle_taxi_h,
         order_num_h, reward_h, num_pre_reject, AOV, disp_cost, incentives)

# print(AOV)
num_pre_reject = num_pre_reject.sum(0)
num_pre_reject /= EPS
num_pre_reject = [int(x) for x in num_pre_reject]
num_pre_reject = [
    num_pre_reject[6 * x] + num_pre_reject[6 * x + 1] + num_pre_reject[6 * x + 2] + num_pre_reject[6 * x + 3] +
    num_pre_reject[6 * x + 4] + num_pre_reject[6 * x + 5] for x in
    range(18)]

TOV_ = [[round(
    TOV[i][6 * x] + TOV[i][6 * x + 1] + TOV[i][6 * x + 2] + TOV[i][6 * x + 3] + TOV[i][6 * x + 4] + TOV[i][6 * x + 5])
    for x in
    range(18)] for i in range(EPS)]
print(TOV_)

# print([T.sum() for T in TOV_])

TOV = TOV.sum(0)
reject_rate_h = reject_rate_h.sum(0)
TOV /= EPS

AOV = AOV.sum()
AOV /= EPS

disp_cost = disp_cost.sum()
disp_cost /= EPS

incentives = incentives.sum()
incentives /= EPS

idle_taxi_h = idle_taxi_h.sum(0)
idle_taxi_h /= EPS
idle_taxi_h = [int(x) for x in idle_taxi_h]
idle_taxi_h = [(idle_taxi_h[6 * x] + idle_taxi_h[6 * x + 1] + idle_taxi_h[6 * x + 2] + idle_taxi_h[6 * x + 3] +
                idle_taxi_h[6 * x + 4] +
                idle_taxi_h[6 * x + 5]) for x in range(18)]

order_num_h = order_num_h.sum(0)
order_num_h /= EPS
order_num_h = [int(x) for x in order_num_h]

# print(reward_h)
reward_h = reward_h.sum(0)
reward_h /= EPS
reward_h = [(reward_h[6 * x] + reward_h[6 * x + 1] + reward_h[6 * x + 2] + reward_h[6 * x + 3] + reward_h[6 * x + 4] +
             reward_h[6 * x + 5]) / 6 for x in range(18)]
reward_h = [int(x) for x in reward_h]

reject_rate_h /= EPS
reject_rate_h_ = [
    (reject_rate_h[6 * x] + reject_rate_h[6 * x + 1] + reject_rate_h[6 * x + 2] + reject_rate_h[6 * x + 3] +
     reject_rate_h[6 * x + 4] + reject_rate_h[6 * x + 5]) / 6 for x in range(18)]

reject_rate_h_ = [round(x, 2) for x in reject_rate_h_]

order_num_h = [(order_num_h[6 * x] + order_num_h[6 * x + 1] + order_num_h[6 * x + 2] + order_num_h[6 * x + 3] +
                order_num_h[6 * x + 4] +
                order_num_h[6 * x + 5]) for x in range(18)]

order_catched = [order_num_h[x] * (1 - reject_rate_h_[x]) for x in range(18)]
order_catched = [int(x) for x in order_catched]

TOV = [int(TOV[x] // 1 + (TOV[x] % 1 > 0)) for x in range(108)]
step_TOV = [TOV[x] for x in range(108)]
hour_TOV = [TOV[6 * x] + TOV[6 * x + 1] + TOV[6 * x + 2] + TOV[6 * x + 3] + TOV[6 * x + 4] + TOV[6 * x + 5] for x in
            range(18)]

print("step_TOV: ", step_TOV)
print("hour_TOV: ", hour_TOV)
print("day_TOV: ", sum(hour_TOV))

print("day_DOV: ", AOV)

print("dispatch cost: ", disp_cost)
print("incentive: ", incentives)

print("reject_rate_h: ", reject_rate_h_)
print("reject_rate: ", sum(reject_rate) / EPS)

print("reward_h: ", reward_h)
print("reward: ", sum(reward_h))

print("idle_taxi_h: ", idle_taxi_h)
print("num_pre_reject: ", num_pre_reject)

print("order_sum_h: ", order_num_h)
print("order_catched: ", order_catched)
print("num_order: ", sum(order_catched))

print("avg_wait: ", sum(avg_wait) / EPS)
print("pre_reject: ", sum(pre_reject) / EPS)
print("num_dispatch: ", sum(num_disp) / EPS)
