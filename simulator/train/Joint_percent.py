# -*-coding: utf-8 -*-
# @Time : 2022/5/6 19:50 下午
# @Author : Chen Haoyang   SEU
# @File : RL_algo.py
# @Software : PyCharm

import os

# sys.path.append(os.path.dirname(sys.path[0]))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from simulator.simulator import *

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

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
    actions = torch.stack([x[1] for x in minibatch])
    rewards = torch.stack([x[2] for x in minibatch])
    next_states = torch.stack([x[3] for x in minibatch])

    value_output = critic(states)
    value_output = value_output.squeeze()

    next_value_output = critic(next_states)
    next_value_output = next_value_output.squeeze()

    td_error = rewards - value_output + GAMMA * next_value_output

    td_error = td_error.view(len(td_error), -1)

    critic_loss = torch.mean(td_error ** 2)

    actor_loss = - torch.mean(td_error * actions)

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
    action = [float(x / output.sum()) for x in output]
    return action


class Actor(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * 2, output_size=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * 2, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


def GetStateFunction(self, cluster):
    DemandExpect = np.array(self.DemandExpect)
    DemandExpect = np.array(np.random.normal(DemandExpect, DemandExpect * 0.2) + 0.5, dtype=int)
    DemandExpect = np.maximum(DemandExpect, 0)

    state = np.zeros((KAPPA + 1) * 2)
    neighbour = cluster.Neighbor
    state[8] = int(self.SupplyExpect[cluster.ID] + len(cluster.IdleVehicles))
    state[9] = int(DemandExpect[cluster.ID])
    for nc in neighbour:
        id = nc.ID
        idx = self.Getidx(cluster.ID, id)
        state[idx * 2] = int(self.SupplyExpect[id] + len(nc.IdleVehicles))
        state[idx * 2 + 1] = int(DemandExpect[id])
    return state


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

        total_reward = 0
        step = 0
        reject = 0

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

            for idx in seq:
                cluster = self.Clusters[idx]

                cluster_counter += 1
                if cluster_counter % 9 == 0:
                    self.Refresh_Pre()

                state = GetStateFunction(self, cluster)
                actions = DispatchFunction(self, state, actor)
                num_disp = [round(len(cluster.IdleVehicles) * x) for x in actions]

                n_before = []
                for neighbor in cluster.Neighbor:
                    n_before.append(self.SupplyExpect[neighbor.ID] - self.DemandExpect[neighbor.ID])
                var_before = torch.tensor(n_before, dtype=torch.float32).var()

                for vehicle in cluster.IdleVehicles:
                    action = num_disp.index(max(num_disp))

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

                    self.move(vehicle, action, cluster, pre_idxs, idxs)
                    num_disp[action] -= 1

                new_state = GetStateFunction(self, cluster)

                n_after = []
                for neighbor in cluster.Neighbor:
                    n_after.append(self.SupplyExpect[neighbor.ID] - self.DemandExpect[neighbor.ID])
                var_after = torch.tensor(n_after, dtype=torch.float32).var()

                reward = float(var_before - var_after)
                total_reward += reward
                step_reward += reward

                if self.RealExpTime != EndTime:
                    done = False
                    state_v = torch.tensor(state, dtype=torch.float32).to(device)
                    action_v = torch.tensor(actions, dtype=torch.float32).to(device)
                    reward_v = torch.tensor(reward, dtype=torch.float32).to(device)
                    new_state_v = torch.tensor(new_state, dtype=torch.float32).to(device)

                    minibatch.append([state_v, action_v, reward_v, new_state_v])
                    if len(minibatch) >= 1:
                        LearningFunction(self, minibatch, actor, critic, actor_opt, critic_opt,
                                         device)
                        minibatch = []

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
save_path = os.path.join("../", "Models","Joint_percent", "_", "saves_episode", "VeRL0-" + args.name)
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
