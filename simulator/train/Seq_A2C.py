# -*-coding: utf-8 -*-
# @Time : 2022/5/6 19:50 下午
# @Author : Chen Haoyang   SEU
# @File : Seq_A2C.py
# @Software : PyCharm

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


def LearningFunction(self, minibatch, actor, critic, actor_opt, critic_opt, device='cpu'):
    states = torch.stack([x[0] for x in minibatch])
    actions = torch.stack([x[1] for x in minibatch])
    rewards = torch.stack([x[2] for x in minibatch])

    value_output = critic(states)
    value_output = value_output.squeeze()

    mask = (actions != -1)

    td_error = rewards - value_output
    td_error = td_error.view(len(td_error), -1)

    critic_loss = torch.mean(td_error ** 2)

    actor_loss = - torch.mean(td_error * actions * mask)

    actor_opt.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_opt.step()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    return


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


class Disp_Critic(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * 2, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


class Seq_Actor(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * (BATCH_SIZE + 1), output_size=BATCH_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, output_size),
            nn.Sigmoid()
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


class Seq_Critic(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * (BATCH_SIZE + 1), output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.normal_(layer.weight)

    def forward(self, x):
        return self.net(x)


def GetStateFunction(self, cluster, pre_idxs):
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


def Seq_GetStateFunction(self, mat_diff, pre_idxs_batch):
    state = []
    mat_diff = mat_diff.tolist()
    state += mat_diff
    for pre in pre_idxs_batch:
        state += pre.tolist()
    for i in range((KAPPA + 1) * (BATCH_SIZE + 1) - len(state)):
        state.append(0)
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


def SeqFunc(self, state, seq_actor, vlist):
    state_v = torch.tensor(state, dtype=torch.float32).to(device)
    output = seq_actor(state_v).tolist()
    vvdict = {}
    for i in range(len(vlist)):
        vvdict[vlist[i]] = output[i]
    res_list = [pair[0] for pair in dict(sorted(vvdict.items(), key=lambda x: x[1], reverse=True)).items()]
    return res_list, output


def train(self):
    disp_actor = Disp_Actor().to(device)
    seq_actor = Seq_Actor().to(device)
    seq_critic = Seq_Critic().to(device)
    seq_actor_opt = optim.Adam(seq_actor.parameters(), lr=LEARNING_RATE, eps=1e-3)
    seq_critic_opt = optim.Adam(seq_critic.parameters(), lr=LEARNING_RATE)
    disp_actor_name = 'episode_4971actor'
    disp_actor_path = os.path.join(save_path2, disp_actor_name)
    disp_actor.load_state_dict(torch.load(disp_actor_path, map_location=torch.device('cpu')))
    seq = [31, 40, 21, 22, 23, 30, 32, 39, 41, 48, 49, 50, 11, 12, 13, 14, 15, 20, 24, 29, 33, 38, 42, 47, 51, 56, 57,
           58, 59, 60, 1, 2, 3, 4, 5, 6, 7, 10, 16, 19, 25, 28, 34, 37, 43, 46, 52, 55, 61, 64, 65, 66, 67, 68, 69, 70,
           0, 8, 9, 17, 18, 26, 27, 35, 36, 44, 45, 53, 54, 62, 63, 71]

    writer = SummaryWriter(comment='-Seq_' + args.name)
    best_TOV = 0

    minibatch = []

    for episode in range(EPISODES):
        print('Now running episide: ', episode)
        self.CreateAllInstantiate()
        self.Reset()
        self.RealExpTime = self.Orders[0].ReleasTime
        self.NowOrder = self.Orders[0]

        total_reward = 0
        step = 0
        reject = 0
        total_batch_reward = 0

        EndTime = self.Orders[-1].ReleasTime

        while self.RealExpTime <= EndTime:

            self.UpdateFunction()
            self.MatchFunction()
            self.SupplyExpectFunction()
            self.DemandPredictFunction(step)
            self.IdleTimeCounterFunction()
            self.Refresh_Pre()

            cluster_counter = 0
            step_reward = 0

            v_count = 0
            v_disp = 0

            idle_vehicles = []
            for cluster in self.Clusters:
                v_count += len(cluster.IdleVehicles)
                idle_vehicles += cluster.IdleVehicles

            for idx in seq:
                cluster = self.Clusters[idx]

                cluster_counter += 1
                if cluster_counter % 9 == 0:
                    self.Refresh_Pre()

                ######################################################################
                v_num = len(cluster.IdleVehicles)

                batch_num = v_num // BATCH_SIZE
                last_batch_size = v_num % BATCH_SIZE
                if last_batch_size:
                    batch_num += 1
                if batch_num == 1 and last_batch_size == 0:
                    last_batch_size = BATCH_SIZE

                for i_batch in range(batch_num):
                    batch_reward = 0
                    mask = 0
                    if i_batch == batch_num - 1:
                        mask = last_batch_size

                    if cluster.IdleVehicles:
                        mat_diff = GetStateFunction(self, cluster, [0, 0, 0, 0, 0, 0, 0, 0, 0])
                        mat_diff = np.array(mat_diff[0::2])

                        pre_idxs_batch = []
                        idxs_batch = []
                        vlist = []
                        for v in range(BATCH_SIZE):
                            if mask and v == mask:
                                break
                            v_id = cluster.IdleVehicles[v].ID
                            PreList = self.PreList[v_id]
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
                            pre_idxs_batch.append(pre_idxs)
                            idxs_batch.append(idxs)
                            vlist.append(cluster.IdleVehicles[v])
                        seq_state = Seq_GetStateFunction(self, mat_diff, pre_idxs_batch)
                        seq_list, seq_action = SeqFunc(self, seq_state, seq_actor, vlist)

                        if i_batch == batch_num - 1:
                            for h in range(last_batch_size, len(seq_action)):
                                seq_action[h] = -1

                        for vehicle in seq_list:

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

                            state = GetStateFunction(self, cluster, pre_idxs)
                            action = DispatchFunction(self, state, disp_actor)

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

                            result = -0.24256738 * ranking + 0.1716329 * expected_income - 0.11379395
                            exist = True
                            if result < 0:
                                exist = False
                                idxs = idxs[0:int(len(idxs) / 2):1]
                                reject += 1
                                np.random.shuffle(idxs)
                                vehicle.Cluster = self.Act2Cluster(idxs[0], cluster)

                            if vehicle.Cluster == cluster:
                                self.StayExpect[cluster.ID] += 1

                            RandomNode = random.choice(vehicle.Cluster.Nodes)
                            RandomNode = RandomNode[0]
                            vehicle.DeliveryPoint = RandomNode

                            ScheduleCost = self.RoadCost(vehicle.LocationNode, RandomNode)
                            self.TotallyDispatchCost += ScheduleCost

                            vehicle.Cluster.VehiclesArrivetime[vehicle] = min(self.RealExpTime + np.timedelta64(
                                MINUTES * ScheduleCost), self.RealExpTime + self.TimePeriods)

                            self.SupplyExpect[vehicle.Cluster.ID] += 1

                            reward = RewardFunction(self, vehicle, cluster, pre_idxs, action)

                            total_reward += reward
                            step_reward += reward
                            batch_reward += reward

                        batch_reward /= len(seq_list)
                        total_batch_reward += batch_reward
                        if self.RealExpTime != EndTime:
                            state_v = torch.tensor(seq_state, dtype=torch.float32).to(device)
                            action_v = torch.tensor(seq_action, dtype=torch.float32).to(device)
                            reward_v = torch.tensor(batch_reward, dtype=torch.float32).to(device)

                            minibatch.append([state_v, action_v, reward_v])
                            if len(minibatch) == 10:
                                LearningFunction(self, minibatch, seq_actor, seq_critic, seq_actor_opt, seq_critic_opt,
                                                 device)
                                minibatch = []
                        cluster.IdleVehicles = cluster.IdleVehicles[BATCH_SIZE::]

                cluster.IdleVehicles.clear()

            if episode % 50 == 0:
                writer.add_scalar('Episode_' + str(episode) + '_reward_per_step', step_reward, step)

            step += 1
            self.RealExpTime += self.TimePeriods

        if episode % 5 == 0:
            name = 'episode_' + str(episode)
            fname = os.path.join(save_path, name)
            torch.save(seq_actor.state_dict(), fname + "actor")
            torch.save(seq_critic.state_dict(), fname + "critic")

        SumOrderValue = 0
        OrderValueNum = 0
        for i in self.Orders:
            if i.ArriveInfo != "Reject":
                SumOrderValue += self.GetValue(i.OrderValue)
                OrderValueNum += 1

        print("Experiment over")
        print("Episode: " + str(episode))
        print('Total reward: ', total_reward)
        avg_reward = total_reward / self.DispatchNum
        print('Average reward: ', avg_reward)
        print('reject rate: ', reject / self.DispatchNum)
        print('Total Batch Reward:', total_batch_reward)

        writer.add_scalar('Total Batch Reward', total_batch_reward, episode)
        writer.add_scalar('PRE_reject', reject / self.DispatchNum, episode)
        writer.add_scalar('avg_reward_per_ep', avg_reward, episode)
        writer.add_scalar('total_reward_per_ep', total_reward, episode)

        if best_TOV is None or best_TOV < SumOrderValue:
            if best_TOV is not None:
                print('Best TOV updated: %.3f -> %.3f' % (best_TOV, SumOrderValue))
                name = 'best_%+.3f_%d.dat' % (SumOrderValue, episode)
                fname = os.path.join(save_path, name)
                torch.save(seq_actor.state_dict(), fname + "actor")
                torch.save(seq_critic.state_dict(), fname + "critic")
            best_TOV = SumOrderValue

        print("Date: " + str(self.Orders[0].ReleasTime.month) + "/" + str(self.Orders[0].ReleasTime.day))
        print("Weekend or Workday: " + self.WorkdayOrWeekend(self.Orders[0].ReleasTime.weekday()))
        if self.ClusterMode != "Grid":
            print("Number of Clusters: " + str(len(self.Clusters)))
        elif self.ClusterMode == "Grid":
            print("Number of Grids: " + str((self.NumGrideWidth * self.NumGrideHeight)))
        print("Number of Vehicles: " + str(len(self.Vehicles)))
        print("Number of Orders: " + str(len(self.Orders)))
        print("Number of Reject: " + str(self.RejectNum))

        writer.add_scalar("Number of Reject: ", self.RejectNum, episode)
        print('Number of Stay; ' + str(self.StayNum))

        writer.add_scalar("Number of Dispatch: ", self.DispatchNum - self.StayNum, episode)
        writer.add_scalar("Average Dispatch Cost: ", self.TotallyDispatchCost / self.DispatchNum, episode)
        writer.add_scalar("Average wait time: ", self.TotallyWaitTime / (len(self.Orders) - self.RejectNum), episode)

        print("Total Order value: " + str(SumOrderValue))
        writer.add_scalar("Total Order value: ", SumOrderValue, episode)

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
save_path = os.path.join("../", "Models", "A2C_pre_nsr", "_", "saves_episode", "Seq-" + args.name)
save_path2 = os.path.join("../", "Models", "A2C_pre_nsr", "_", "saves_episode", "VeRL0-" + args.name)
os.makedirs(save_path, exist_ok=True)

EXPSIM = Simulation(
    ClusterMode=ClusterMode,
    DemandPredictionMode=DemandPredictionMode,
    DispatchMode=DispatchMode,
    VehiclesNumber=VehiclesNumber,
    TimePeriods=TIMESTEP,
    LocalRegionBound=LocalRegionBound,
    SideLengthMeter=SideLengthMeter,
    VehiclesServiceMeter=VehiclesServiceMeter,
    NeighborCanServer=NeighborCanServer,
    FocusOnLocalRegion=FocusOnLocalRegion,
)
train(EXPSIM)
