# -*-coding: utf-8 -*-
# @Time : 2022/8/3 10:30
# @Author : Chen Haoyang   SEU
# @File : RL_algo_COX.py
# @Software : PyCharm

import os
import random
import time

# from blaze.expr.datetime import dt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from simulator.simulator import *
from objects import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter
from haversine import haversine

import numpy as np
import ptan
import argparse
from sklearn.preprocessing import minmax_scale

GLOBAL_VISION = False
PREFERENCE = True

EPISODES = 10000

PRIO_REPLAY_PHI = 0.6
BETA_START = 0.4
BETA_FRAMES = 108 * 4

HIDDEN_SIZE = 64
MINI_BATCH = 56
REPLAY_PERIOD = 3
BUFFER_SIZE = 108 * 4

KAPPA = 8
GAMMA = 0.95

EPSILON = 0.3
EPSILON_EP = 1



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
    # state.append(pre_idxs[4])

    nClusters = cCluster.Neighbor

    for neighbour in nClusters:
        state.append(self.StayExpect[neighbour.ID] + self.SupplyExpect[neighbour.ID])
        state.append(int(self.DemandExpect[neighbour.ID]))

        # state.append(pre_idxs[Getidx(cluster.ID, neighbour.ID)])

    while len(state) < (KAPPA + 1) * 2 + 1:
        state.append(0)

    if len(state) > (KAPPA + 1) * 2 + 1:
        state = state[:(KAPPA + 1) * 2 + 1]

    return state


def RewardFunction(self, state, vehicle, cluster, pre_idxs):
    """Supply-demand"""
    Omega_i = 0
    Omega_g = 0
    cluster_state = GetStateFunction(self, vehicle.ID, cluster, pre_idxs)
    Omega_i = cluster_state[1] / (cluster_state[2] + 0.001)
    Omega_g = cluster_state[1 + self.Getidx(cluster.ID, vehicle.Cluster.ID) * 2] / (
            cluster_state[2 + self.Getidx(cluster.ID, vehicle.Cluster.ID) * 2] + 0.001)
    if 0 < Omega_i < 1:
        if cluster == vehicle.Cluster:
            return 5
        else:
            return -5
    else:
        if 0 <= Omega_g <= 1:
            return 1 / (Omega_g + 0.001)
        else:
            if cluster == vehicle.Cluster:
                return 0
            else:
                return -Omega_g


def LearningFunction(self, buffer, optimizer, net, tgt_net, frame_idx, device='cpu'):
    """
    for i=1 to b do
        sample transition tuple i~P(i) = p^phi_i / sum_j(p^phi_j)
        Compute importance sampling weight w_i = (N * P(i))^-beta / max_j(w_j)     N is the size of the Memory M
        Compute TD-error delta_i using delta = r_{t-1} + gamma * max_{a_star}(Q(s_t,a_star)-Q(s_{t-1},a_{t-1})
        Update transition priority p_i = abs(delta_i)
        Accumulate weight-change Delta := Delta + w_i * delta_i * Q(s_i,a_i)
    Update Q network theta := theta + eta * Delta, reset Delta = 0
    Set theta' = theta after replay period of 144 steps
    """
    beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)
    optimizer.zero_grad()
    batch, batch_indices, batch_weights = buffer.sample(MINI_BATCH, beta)
    loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                       GAMMA, device=device)
    loss_v.backward()
    optimizer.step()
    buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())
    return


def unpack_batch(batch):
    states, actions, rewards, last_states = [], [], [], []
    for exp in batch:
        state = np.array(exp[0], copy=False)
        states.append(state)
        actions.append(exp[1])
        rewards.append(exp[2])
        if exp[3] is None:
            last_states.append(state)
        else:
            last_states.append(np.array(exp[3], copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(last_states,
                                                                                                          copy=False)


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device='cpu'):
    states, actions, rewards, next_states = unpack_batch(batch)
    states_v = torch.tensor(states, dtype=torch.float32).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    # print('states_v shape: ', states_v.shape)
    # print('states_v.unsqueeze(dim=0) shape: ', states_v.unsqueeze(dim=0).shape)
    # print('net(): ', net(states_v))
    # print('net() shape: ', net(states_v).shape)

    # print(net(states_v))
    # print(net(states_v).shape)
    # print(net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1))
    # print(net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1).shape)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), losses_v + 1e-5


def train(self):
    M = []

    Q_net = DQN().to(device)
    tgt_net = ptan.agent.TargetNet(Q_net)
    opt_Q = optim.Adam(Q_net.parameters())
    writer = SummaryWriter(comment='-VeRL_' + args.name)
    best_TOV = 0

    for episode in range(EPISODES):
        if episode % 10 == 0:
            tgt_net.sync()
        print('Now running episide: ', episode)
        self.Reset()
        EpisodeStartTime = dt.datetime.now()
        self.RealExpTime = self.Orders[0].ReleasTime
        self.NowOrder = self.Orders[0]

        epsilon = EPSILON - episode * (EPSILON / EPSILON_EP)

        total_reward = 0
        self.step = 0
        short = 0
        step=0
        reject = 0

        EndTime = self.Orders[-1].ReleasTime

        pair_per_cluster = []

        while self.RealExpTime <= EndTime:

            self.UpdateFunction()

            self.MatchFunction()

            if (self.step >= REPLAY_PERIOD) and (episode >= 2):
                M = M[-BUFFER_SIZE:]
                buffer = PrioReplayBuffer(M, BUFFER_SIZE)
                buffer.populate(len(M))
                # print('learning************************')
                frame_idx = self.step - REPLAY_PERIOD

                StepLearningStartTime = dt.datetime.now()
                LearningFunction(self, buffer, opt_Q, Q_net, tgt_net, frame_idx, device)
                self.TotallyLearningTime += dt.datetime.now() - StepLearningStartTime

            pair_per_cluster = []

            ##############################################
            self.SupplyExpectFunction()
            self.DemandPredictFunction(step)
            self.IdleTimeCounterFunction()
            ##############################################

            cluster_counter = 0
            step_reward = 0

            # print('step', step)
            for cluster in self.Clusters:

                cluster_counter += 1
                vehicle_counter = 0
                # print('ClusterCounter:', cluster_counter)

                for vehicle in cluster.IdleVehicles:
                    vehicle_counter += 1

                    PreList = self.PreList[vehicle.ID]
                    pre_idxs = [PreList[5], PreList[6], PreList[7], PreList[4], PreList[8], PreList[0],
                                PreList[3], PreList[2], PreList[1]]

                    # pre_idxs = [round(x, 3) for x in pre_idxs]
                    pre_idxs = np.array(pre_idxs)

                    idxs = pre_idxs.argsort()
                    s = 0
                    for i in idxs:
                        pre_idxs[i] = s
                        s += 1

                    idxs = idxs[::-1]
                    pre_idxs = minmax_scale(pre_idxs, (0, 1))

                    state = GetStateFunction(self, vehicle.ID, cluster, pre_idxs)
                    action = DispatchFunction(self, state, Q_net, episode, epsilon)

                    #
                    # state_v = torch.tensor(state, dtype=torch.float32)
                    # writer.add_graph(Alter_DQN(), (state_v.unsqueeze(dim=0)))

                    self.move(vehicle, action, cluster, pre_idxs, idxs)

                    new_state = GetStateFunction(self, vehicle.ID, cluster, pre_idxs)

                    reward = RewardFunction(self, state, vehicle, cluster, pre_idxs)
                    # print('reward:', reward)

                    total_reward += reward
                    step_reward += reward

                    M.append([state, action, reward, new_state])

                pair_per_cluster.append(vehicle_counter)

            if episode % 50 == 0:
                writer.add_scalar('Episode_' + str(episode) + '_reward_per_step', step_reward, step)

            step += 1
            self.RealExpTime += self.TimePeriods

        if episode % 10 == 0:
            name = 'episode_' + str(episode)
            fname = os.path.join(save_path, name)
            torch.save(Q_net.state_dict(), fname)

        EpisodeEndTime = dt.datetime.now()
        SumOrderValue = 0
        OrderValueNum = 0
        for i in self.Orders:
            if i.ArriveInfo != "Reject":
                SumOrderValue += i.OrderValue
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
                print('Best reward updated: %.3f -> %.3f' % (best_TOV, avg_reward))
                name = 'best_%+.3f_%d.dat' % (SumOrderValue, episode)
                fname = os.path.join(save_path, name)
                torch.save(Q_net.state_dict(), fname)
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

        print("Number of Dispatch: " + str(self.DispatchNum - self.StayNum))
        print('Number of Stay; ' + str(self.StayNum))
        print('Short Dispatch: ' + str(short))

        writer.add_scalar("Number of Dispatch: ", self.DispatchNum - self.StayNum, episode)

        if self.DispatchNum != 0:
            print("Average Dispatch Cost: " + str(self.TotallyDispatchCost / self.DispatchNum))
        if (len(self.Orders) - self.RejectNum) != 0:
            print("Average waiting time: " + str(self.TotallyWaitTime / (len(self.Orders) - self.RejectNum)))

        writer.add_scalar("Average Dispatch Cost: ", self.TotallyDispatchCost / self.DispatchNum, episode)
        writer.add_scalar("Average wait time: ", self.TotallyWaitTime / (len(self.Orders) - self.RejectNum), episode)

        print("Total Order value: " + str(SumOrderValue))

        writer.add_scalar("Total Order value: ", SumOrderValue, episode)

        # print("Total Update Time : " + str(self.TotallyUpdateTime))
        # print("Total NextState Time : " + str(self.TotallyNextStateTime))
        print("Total Learning Time : " + str(self.TotallyLearningTime))
        # print("Total Demand Predict Time : " + str(self.TotallyDemandPredictTime))
        # print("Total Dispatch Time : " + str(self.TotallyDispatchTime))
        # print("Total Simulation Time : " + str(self.TotallyMatchTime))
        print("Episode Run time : " + str(EpisodeEndTime - EpisodeStartTime))

    return

def DispatchFunction(self, state, net, ep, epsilon):
    self.DispatchNum += 1
    state_v = torch.tensor(state, dtype=torch.float32).to(device)
    output = net(state_v.unsqueeze(dim=0)).reshape(1, KAPPA + 1)
    idx = torch.max(output, 1)[1].numpy()
    return int(idx)


DispatchMode = "Simulation"
DemandPredictionMode = "None"
ClusterMode = "Grid"

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
parser.add_argument("-n", "--name", default='test00', help="Name of the run")
args = parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_path = os.path.join("../", 'COX', "_", "saves_episode", "VeRL0-" + args.name)  ###模型名称
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
