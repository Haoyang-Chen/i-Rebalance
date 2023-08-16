# -*-coding: utf-8 -*-
# @Time : 2022/5/6 19:50 下午
# @Author : Chen Haoyang   SEU
# @File : RL_algo.py
# @Software : PyCharm

import sys
import os

# sys.path.append(os.path.dirname(sys.path[0]))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from simulator.simulator import *

import torch
import torch.nn as nn
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
GAMMA = 0.98

EPSILON = 0.8
EPSILON_EP = 100

LEARNING_RATE = 0.001


def LearningFunction(self, minibatch, actor, critic, actor_opt, critic_opt, device='cpu'):
    states = torch.stack([x[0] for x in minibatch])
    actions = torch.Tensor([x[1] for x in minibatch]).long()
    rewards = torch.stack([x[2] for x in minibatch])
    next_states = torch.stack([x[3] for x in minibatch])
    policy_output = actor(states)
    action_probs = nn.functional.softmax(policy_output, dim=1)

    value_output = critic(states)
    value_output = value_output.squeeze()

    next_value_output = critic(next_states)
    next_value_output = next_value_output.squeeze()
    advantage = rewards + GAMMA * next_value_output - value_output

    # actor_loss = - torch.mean(torch.log(action_probs[range(10), actions]) * advantage)
    critic_loss = torch.mean(advantage ** 2)

    policy_dist = torch.distributions.Categorical(action_probs)
    entropy = policy_dist.entropy()

    actor_loss = - torch.mean(torch.log(action_probs[range(len(minibatch)), actions]) * advantage - ENTROPY_BETA * entropy)

    actor_opt.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_opt.step()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    return


def DispatchFunction(self, state, actor, ep, epsilon):
    self.DispatchNum += 1
    state_v = torch.tensor(state, dtype=torch.float32).to(device)
    output = actor(state_v)
    action_probs = F.softmax(output, dim=0)
    action = torch.multinomial(action_probs, 1).item()
    if ep < EPSILON_EP and random.randrange(0, 10000) / 10000 < epsilon:
        action = random.randrange(0, 9)
    # print(action)
    return action


class Actor(nn.Module):
    def __init__(self, input_size=(KAPPA + 1), output_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            # nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            # nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_size=(KAPPA + 1), output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            # nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            # nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


def GetStateFunction(self,cluster):
    DemandExpect = np.array(self.DemandExpect)
    DemandExpect = np.array(np.random.normal(DemandExpect, DemandExpect * 0.2) + 0.5, dtype=int)
    DemandExpect = np.maximum(DemandExpect, 0)

    state = np.zeros((KAPPA + 1))
    neighbour = cluster.Neighbor
    state[4] = int(self.SupplyExpect[cluster.ID]) - int(DemandExpect[cluster.ID])
    for nc in neighbour:
        id = nc.ID
        idx = self.Getidx(cluster.ID, id)
        state[idx] = int(self.SupplyExpect[id]) - int(DemandExpect[id])
    return state


def RewardFunction(self, vehicle, cluster):
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
    return 2 * balance_reward


def train(self):
    actor = Actor().to(device)
    critic = Critic().to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=LEARNING_RATE, eps=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    seq = [31, 40, 21, 22, 23, 30, 32, 39, 41, 48, 49, 50, 11, 12, 13, 14, 15, 20, 24, 29, 33, 38, 42, 47, 51, 56, 57,
           58, 59, 60, 1, 2, 3, 4, 5, 6, 7, 10, 16, 19, 25, 28, 34, 37, 43, 46, 52, 55, 61, 64, 65, 66, 67, 68, 69, 70,
           0, 8, 9, 17, 18, 26, 27, 35, 36, 44, 45, 53, 54, 62, 63, 71]

    writer = SummaryWriter(comment='-VeRL_' + args.name)
    best_TOV = 0

    minibatch = []

    for episode in range(EPISODES):
        print('Now running episide: ', episode)
        self.CreateAllInstantiate()
        self.Reset()
        self.RealExpTime = self.Orders[0].ReleasTime
        self.NowOrder = self.Orders[0]

        # epsilon = EPSILON - episode * (EPSILON / EPSILON_EP)
        epsilon = 0

        total_reward = 0
        step = 0
        reject = 0

        EndTime = self.Orders[-1].ReleasTime

        while self.RealExpTime <= EndTime:

            self.UpdateFunction()

            self.MatchFunction()

            ##############################################
            self.SupplyExpectFunction()
            self.DemandPredictFunction(step)
            self.IdleTimeCounterFunction()
            ##############################################

            cluster_counter = 0
            step_reward = 0

            self.Refresh_Pre()

            v_count = 0
            v_disp = 0

            idle_vehicles = []
            for cluster in self.Clusters:
                v_count += len(cluster.IdleVehicles)
                idle_vehicles += cluster.IdleVehicles

            for exp in minibatch:
                temp_vehicle = self.Vehicles[exp[5]]
                temp_cluster = temp_vehicle.Cluster
                if temp_vehicle not in idle_vehicles:
                    exp[4] = True
                    exp[2]+=5
                else:
                    temp_next_state = GetStateFunction(self, temp_cluster)
                    temp_next_state_v = torch.tensor(temp_next_state, dtype=torch.float32).to(device)
                    exp[3] = temp_next_state_v
            if len(minibatch) > 1:
                LearningFunction(self, minibatch, actor, critic, actor_opt, critic_opt,
                                 device)
                minibatch = []

            for idx in seq:
                cluster = self.Clusters[idx]

                cluster_counter += 1
                if cluster_counter % 9 == 0:
                    self.Refresh_Pre()


                for vehicle in cluster.IdleVehicles:

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

                    state = GetStateFunction(self, cluster)
                    action = DispatchFunction(self, state, actor, episode, epsilon)

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

                    if result < 0:
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

                    new_state = GetStateFunction(self, cluster)

                    reward = RewardFunction(self, vehicle, cluster)

                    total_reward += reward
                    step_reward += reward

                    if self.RealExpTime != EndTime:
                        done = False
                        state_v = torch.tensor(state, dtype=torch.float32).to(device)
                        action_v = torch.tensor(action, dtype=torch.float32).to(device)
                        reward_v = torch.tensor(reward, dtype=torch.float32).to(device)
                        new_state_v = torch.tensor(new_state, dtype=torch.float32).to(device)

                        minibatch.append([state_v, action_v, reward_v, new_state_v, done, self.Vehicles.index(vehicle)])


                cluster.IdleVehicles.clear()

            if episode % 50 == 0:
                writer.add_scalar('Episode_' + str(episode) + '_reward_per_step', step_reward, step)

            step += 1
            self.RealExpTime += self.TimePeriods

        if episode % 5 == 0:
            name = 'episode_' + str(episode)
            fname = os.path.join(save_path, name)
            torch.save(actor.state_dict(), fname + "actor")
            torch.save(critic.state_dict(), fname + "critic")

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

        writer.add_scalar('PRE_reject', reject / self.DispatchNum, episode)
        writer.add_scalar('avg_reward_per_ep', avg_reward, episode)
        writer.add_scalar('total_reward_per_ep', total_reward, episode)

        if best_TOV is None or best_TOV < SumOrderValue:
            if best_TOV is not None:
                print('Best TOV updated: %.3f -> %.3f' % (best_TOV, SumOrderValue))
                name = 'best_%+.3f_%d.dat' % (SumOrderValue, episode)
                fname = os.path.join(save_path, name)
                torch.save(actor.state_dict(), fname + "actor")
                torch.save(critic.state_dict(), fname + "critic")
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
device = torch.device("cuda:3" if args.cuda else "cpu")
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_path = os.path.join("../","Models","Disp_A2C_NP", "_", "saves_episode", "VeRL0-" + args.name)
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
EXPSIM.CreateAllInstantiate()
train(EXPSIM)
