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
from ortools.graph.python import min_cost_flow

PLOT = False
DISPATCH = True

KAPPA = 8

DM = 'survey'

LAMBDA = 0.8
GAMMA = -1.5


def Can_Weight(self, cluster, order_num):
    step_start = self.RealExpTime // self.TimePeriods
    step_end = step_start + 4
    num_dest = 0
    for vehicle, time in list(cluster.VehiclesArrivetime.items()):
        if time <= self.RealExpTime:
            num_dest += 1
    orig_sum = 0
    for i in range(step_start, step_end):
        orig_sum += order_num[i][cluster.ID]
    return orig_sum - LAMBDA * num_dest


def Des_Weight(self, cluster, vehicle):
    T = self.RoadCost(vehicle.LocationNode, cluster.Nodes[0])
    N = len(cluster.IdleVehicles)
    return (T * N) ** GAMMA


def Score(self, cluster, vehicle, order_num):
    step_start = self.RealExpTime // self.TimePeriods
    step_end = step_start + 4
    T = self.RoadCost(vehicle.LocationNode, cluster.Nodes[0])
    orig_sum = 0
    for i in range(step_start, step_end):
        orig_sum += order_num[i][cluster.ID]
    return T / orig_sum


def Capacity(Can_Weights, A, i):
    W = Can_Weights[i]
    Sum_W = sum(Can_Weights)
    return W / Sum_W * A


def GetStateFunction(self, vehicle_id, cluster, pre_idxs):
    """
    for each dispatched taxi i do
        observe state s^i_t
        Store tuple (s,a,r,s) into M
    """
    state = []
    cCluster = cluster

    state.append(cCluster.ID)
    state.append(int(self.StayExpect[cCluster.ID] + self.SupplyExpect[cCluster.ID]))
    state.append(int(self.DemandExpect[cCluster.ID]))
    state.append(pre_idxs[4])

    nClusters = cCluster.Neighbor

    for neighbour in nClusters:
        state.append(self.StayExpect[neighbour.ID] + self.SupplyExpect[neighbour.ID])
        state.append(int(self.DemandExpect[neighbour.ID]))

        state.append(pre_idxs[self.Getidx(cluster.ID, neighbour.ID)])

    while len(state) < (KAPPA + 1) * 3 + 1:
        state.append(0)

    if len(state) > (KAPPA + 1) * 3 + 1:
        state = state[:(KAPPA + 1) * 3 + 1]

    return state


def TEST(self, rand, pre, a, reject_rate, avg_wait, pre_reject, num_disp, idxx, reject_rate_h, idle_taxi_h,
         order_num_h, reward_h, num_pre_reject, AOV, disp_cost, incentives):
    self.CreateAllInstantiate_TEST(idxx // int(EPS / 6))
    self.Reset()
    EpisodeStartTime = dt.datetime.now()
    self.RealExpTime = self.Orders[0].ReleasTime
    self.NowOrder = self.Orders[0]


    EndTime = self.Orders[-1].ReleasTime

    step = 0
    reject = 0
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

            clusters_list = []
            end_nodes = []
            start_nodes = []

            for cluster in self.Clusters:
                if len(cluster.Orders) - len(cluster.IdleVehicles) < 0:
                    clusters_list.append(cluster.ID)
            for i in clusters_list:
                for j in range(len(self.Clusters[i].Neighbor)):
                    if len(self.Clusters[i].Neighbor[j].Orders) - len(self.Clusters[i].Neighbor[j].IdleVehicles) > 0:
                        end_nodes.append(self.Clusters[i].Neighbor[j].ID)
                        start_nodes.append(i)
            num_start = len(list(set(start_nodes)))
            start_nodes_new = [72] * num_start + start_nodes + list(set(end_nodes))
            end_nodes_new = list(set(start_nodes)) + end_nodes + [73] * len(list(set(end_nodes)))
            cost = [1] * len(start_nodes_new)
            supply_list = list(set(start_nodes + end_nodes))
            SD_positive = []
            SD_negative = []
            for i in supply_list:
                if len(self.Clusters[i].Orders) - len(self.Clusters[i].IdleVehicles) < 0:
                    SD_positive.append(-len(self.Clusters[i].Orders) + len(self.Clusters[i].IdleVehicles))
                else:
                    SD_negative.append(len(self.Clusters[i].Orders) - len(self.Clusters[i].IdleVehicles))
            capacity = SD_positive + [1000] * (len(start_nodes_new) - len(SD_positive))

            SD = [sum(SD_positive)] + [0] * len(supply_list) + [-sum(SD_positive)]

            start_nodes_new = np.array(start_nodes_new)
            end_nodes_new = np.array(end_nodes_new)
            cost = np.array(cost)
            capacity = np.array(capacity)
            smcf = min_cost_flow.SimpleMinCostFlow()
            all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
                start_nodes_new, end_nodes_new, capacity, cost)

            # Add supply for each nodes.
            smcf.set_nodes_supply(np.array([72] + supply_list + [73]), SD)
            status = smcf.solve()
            if status != smcf.OPTIMAL:
                print('There was an issue with the min cost flow input.')
                print(f'Status: {status}')
                exit(1)
            solution_flows = smcf.flows(all_arcs)
            a1 = []
            b1 = []
            c1 = []
            for i in range(num_start, len(solution_flows) - len(list(set(end_nodes)))):
                a1.append(start_nodes_new[i])
                b1.append(end_nodes_new[i])
                c1.append(solution_flows[i])
            d = []
            d.append(a1)
            d.append(b1)
            d.append(c1)

            min_cost_expect = np.zeros((72, 72))
            for iter in range(len(d[0])):
                min_cost_expect[d[0][iter], d[1][iter]] = d[2][iter]

            flow_record = np.zeros((72, 72))

            for cluster in self.Clusters:

                if cluster.ID % 9 == 0:
                    self.Refresh_Pre()

                for vehicle in cluster.IdleVehicles:
                    self.DispatchNum += 1
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

                    state = GetStateFunction(self, vehicle.ID, cluster, pre_idxs)

                    neighbours = cluster.Neighbor
                    already_dispatch = 0
                    for nc in neighbours:
                        if flow_record[cluster.ID, nc.ID] < min_cost_expect[cluster.ID, nc.ID]:
                            action = self.Getidx(cluster.ID, nc.ID)
                            vehicle.Cluster = self.Act2Cluster(action, vehicle.Cluster)
                            flow_record[cluster.ID, nc.ID] += 1
                            already_dispatch = 1

                    if already_dispatch == 0:
                        action = 4
                        vehicle.Cluster = self.Act2Cluster(action, vehicle.Cluster)

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
                                vehicle.Auto = True
                                incentive += (0 - result) / 0.2932
                                idxs = idxs[0:int(len(idxs) / 2):1]
                                reject += 1
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
                    self.TotallyDispatchCost += ScheduleCost

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
                    SOV += self.GetValue(i.OrderValue)
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

    print("Total Order value: " + str(SumOrderValue))
    print('Auto Order Value: ' + str(self.AutoOrderValue))
    print("Average Dispatch Cost: " + str(self.TotallyDispatchCost / self.DispatchNum))
    print("Dispatch Cost: " + str(self.TotallyDispatchCost))
    disp_cost[idxx] += self.TotallyDispatchCost
    AOV[idxx] += SumOrderValue - self.AutoOrderValue
    print('Incentive: ', incentive)
    incentives[idxx] += incentive
    print("----------------------------Experiment over----------------------------")
    if DISPATCH:
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
    print("Episode Run time : " + str(dt.datetime.now() - EpisodeStartTime))
    return


DispatchMode = "Simulation"
DemandPredictionMode = "None"
ClusterMode = "Grid"

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
parser.add_argument("-n", "--name", default='test00', help="Name of the run")
args = parser.parse_args()
device = torch.device("cuda:3" if args.cuda else "cpu")

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

    rand = False
    pre = True
    TEST(EXPSIM, rand, pre, TOV, reject_rate, avg_wait, pre_reject, num_disp, i, reject_rate_h, idle_taxi_h,
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
