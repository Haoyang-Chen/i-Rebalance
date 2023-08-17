# -*-coding: utf-8 -*-
# @Time : 2022/7/31 09:53
# @Author : Chen Haoyang   SEU
# @File : Greedy.py
# @Software : PyCharm


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
from simulator.simulator import *
import seaborn as sns
from sklearn.preprocessing import minmax_scale

PLOT = False
DISPATCH = True

KAPPA = 8

DM = 'survey'


def TEST(self, rand, pre, a, reject_rate, avg_wait, pre_reject, num_disp, idxx, reject_rate_h, idle_taxi_h,
         order_num_h, reward_h, num_pre_reject, AOV, disp_cost, incentives):
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
    positive = 0
    negative = 0

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
            self.Supply_obs = torch.zeros(self.ClustersNumber)
            for i in range(self.ClustersNumber):
                self.Supply_obs[i] += self.SupplyExpect[i]

            for cluster in self.Clusters:

                if cluster.ID % 9 == 0:
                    self.Refresh_Pre()

                for vehicle in cluster.IdleVehicles:
                    idle_taxi_h[idxx][step] += 1

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
                        pre_idxs = minmax_scale(pre_idxs, (0, 1))

                    neighbours = cluster.Neighbor
                    demand = []
                    supply = []

                    for nc in neighbours:
                        demand.append(len(nc.Orders))
                        supply.append(len(nc.IdleVehicles))

                    diff = [demand[i] - supply[i] for i in range(len(demand))]
                    avg_diff = sum(diff) / len(diff)
                    diff = np.array(diff)
                    std_diff = diff.std()
                    diff = [2*(x - avg_diff) / std_diff for x in diff]
                    for i_diff in range(len(diff)):
                        diff[i_diff] += pre_idxs[self.Getidx(cluster.ID, neighbours[i_diff].ID)]

                    idx = diff.index(max(diff))
                    action = self.Getidx(cluster.ID, neighbours[idx].ID)

                    self.DispatchNum += 1

                    if len(cluster.Orders) - self.Supply_obs[cluster.ID] > max(diff):
                        action = 4

                    vehicle.Cluster = self.Act2Cluster(action, vehicle.Cluster)
                    self.Supply_obs[vehicle.Cluster.ID] += 1

                    if pre:
                        if DM == 'sample':
                            if random.randrange(0, 10000) / 10000 > pre_idxs[
                                self.Getidx(cluster.ID, vehicle.Cluster.ID)]:
                                reject += 1
                                temp = sorted(pre_idxs)
                                temp = temp[::-1]
                                for i in range(9):
                                    if random.randrange(0, 10000) / 10000 <= temp[i]:
                                        vehicle.Cluster = self.Act2Cluster(idxs[i], cluster)
                                        break
                                    vehicle.Cluster = cluster

                        if DM == 'top4':
                            idxs = idxs[0:int(len(idxs) / 2):1]
                            idx = self.Getidx(cluster.ID, vehicle.Cluster.ID)
                            exist = False
                            for i in idxs:
                                if i == idx:
                                    exist = True
                            if exist == False:
                                reject += 1
                                np.random.shuffle(idxs)
                                vehicle.Cluster = self.Act2Cluster(idxs[0], cluster)

                        if DM == 'survey':
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
                            if result < 0:
                                incentive += (0 - result) / 0.2932
                                vehicle.Auto = True
                                reject += 1
                                np.random.shuffle(idxs)
                                vehicle.Cluster = self.Act2Cluster(idxs[0], cluster)
                                if (self.DemandExpect[vehicle.Cluster.ID] - self.SupplyExpect[vehicle.Cluster.ID]) <= 0:
                                    negative += 1
                                else:
                                    positive += 1

                    if vehicle.Cluster == cluster:
                        self.StayExpect[cluster.ID] += 1
                        self.StayNum += 1

                    RandomNode = random.choice(vehicle.Cluster.Nodes)
                    RandomNode = RandomNode[0]
                    vehicle.DeliveryPoint = RandomNode

                    ScheduleCost = self.RoadCost(vehicle.LocationNode, RandomNode)
                    self.TotallyDispatchCost += self.RoadDistance(vehicle.LocationNode, RandomNode)

                    vehicle.Cluster.VehiclesArrivetime[vehicle] = min(self.RealExpTime + np.timedelta64(
                        MINUTES * ScheduleCost), self.RealExpTime + self.TimePeriods)

                    self.SupplyExpect[vehicle.Cluster.ID] += 1

                cluster.IdleVehicles.clear()

        num_pre_reject[idxx][step] += pre_rej_count
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

        count = 0
        for i in self.Orders:
            if self.RealExpTime - self.TimePeriods <= i.ReleasTime < self.RealExpTime:
                if i.ArriveInfo != "Reject":
                    dist = i.OrderValue
                    SOV += self.GetValue(dist)
                    reject_rate_h[idxx][step] += 1
                count += 1
        reject_rate_h[idxx][step] = (count - reject_rate_h[idxx][step]) / count
        order_num_h[idxx][step] += count

        a[idxx][step] += SOV
        step += 1

    print('positive: ', positive)
    print('negative: ', negative)
    SumOrderValue = 0
    OrderValueNum = 0
    for i in self.Orders:
        if i.ArriveInfo != "Reject":
            SumOrderValue += self.GetValue(i.OrderValue)
            OrderValueNum += 1
        else:
            self.TotallyWaitTime += 20
    print('Incentive: ', incentive)
    print("Total Order value: " + str(SumOrderValue))
    print('Auto Order Value: ' + str(self.AutoOrderValue))
    print('Dispatch Value: ' + str(SumOrderValue - self.AutoOrderValue))
    AOV[idxx] += SumOrderValue - self.AutoOrderValue
    print('Incentive: ', incentive)
    incentives[idxx] += incentive
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
        print("Dispatch Cost: " + str(self.TotallyDispatchCost))
        disp_cost[idxx] += self.TotallyDispatchCost
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
        FocusOnLocalRegion=FocusOnLocalRegion
    )
    EXPSIM.CreateAllInstantiate()

    rand = False
    pre = True
    TEST(EXPSIM, rand, pre, TOV, reject_rate, avg_wait, pre_reject, num_disp, i, reject_rate_h, idle_taxi_h,
         order_num_h, reward_h, num_pre_reject, AOV, disp_cost, incentives)

num_pre_reject = num_pre_reject.sum(0)
num_pre_reject /= EPS
num_pre_reject = [int(x) for x in num_pre_reject]
num_pre_reject = [
    num_pre_reject[6 * x] + num_pre_reject[6 * x + 1] + num_pre_reject[6 * x + 2] + num_pre_reject[6 * x + 3] +
    num_pre_reject[6 * x + 4] + num_pre_reject[6 * x + 5] for x in
    range(18)]

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

print("avg_wait: ", sum(avg_wait) / EPS)
print("pre_reject: ", sum(pre_reject) / EPS)
print("num_dispatch: ", sum(num_disp) / EPS)
print("num_order: ", sum(order_catched))
