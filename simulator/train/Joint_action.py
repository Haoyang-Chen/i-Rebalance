# -*-coding: utf-8 -*-
# @Time : 2022/5/6 19:50 下午
# @Author : Chen Haoyang   SEU
# @File : RL_algo.py
# @Software : PyCharm

import os

# sys.path.append(os.path.dirname(sys.path[0]))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from simulator.simulator import *

import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from preprocessing.readfiles import *

import numpy as np
import argparse
from sklearn.preprocessing import minmax_scale

EPISODES = 5000

HIDDEN_SIZE = 64

ENTROPY_BETA = 0.01

KAPPA = 8
GAMMA = 0.98

EPSILON = 0.5
EPSILON_EP = 10

BATCH_SIZE = 10

LEARNING_RATE = 0.001


def LearningFunction(self, minibatch, actor, critic, actor_opt, critic_opt, device='cpu'):
    states = torch.stack([x[0] for x in minibatch])
    actions = torch.stack([x[1] for x in minibatch])
    rewards = torch.stack([x[2] for x in minibatch])
    next_states = torch.stack([x[3] for x in minibatch])
    policy_output = actor(states).view(len(states), BATCH_SIZE, 9)
    action_probs = nn.functional.softmax(policy_output, dim=2)

    value_output = critic(states)
    value_output = value_output.squeeze()

    next_value_output = critic(next_states)
    next_value_output = next_value_output.squeeze()

    advantage_mask = (actions != -1)
    masked_advantage = rewards + GAMMA * next_value_output - value_output
    masked_advantage = masked_advantage.view(len(masked_advantage), -1)
    masked_advantage = masked_advantage * advantage_mask.float()

    critic_loss = torch.mean(masked_advantage ** 2)

    policy_dist = torch.distributions.Categorical(action_probs)
    entropy = policy_dist.entropy()

    valid_advantage = masked_advantage * advantage_mask
    valid_actions = actions * advantage_mask
    valid_actions = valid_actions.long()
    selected_probs = torch.zeros(len(states), BATCH_SIZE).to(device)
    for i in range(len(states)):
        for j in range(BATCH_SIZE):
            selected_probs[i][j] += action_probs[i][j][valid_actions[i][j]]

    actor_loss = - torch.mean(
        torch.log(selected_probs) * valid_advantage - ENTROPY_BETA * entropy)

    actor_opt.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_opt.step()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    return


def DispatchFunction(self, state, actor, ep, epsilon):
    state_v = torch.tensor(state, dtype=torch.float32).to(device)
    output = actor(state_v).view(BATCH_SIZE, -1)

    action_probs = F.softmax(output, dim=1)
    action = torch.multinomial(action_probs, 1).tolist()
    action_list = [item[0] for item in action]

    if ep < EPSILON_EP and random.randrange(0, 10000) / 10000 < epsilon:
        action_list = []
        for i in range(BATCH_SIZE):
            action_list.append(random.randrange(0, 9))
    return action_list


class Actor(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * 2 + 1 + BATCH_SIZE * 9, output_size=BATCH_SIZE * 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 2 * HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_size=(KAPPA + 1) * 2 + 1 + BATCH_SIZE * 9, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 2 * HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, output_size)
        )

    def forward(self, x):
        return self.net(x)


def GetStateFunction(self, cluster, pre_idxs_batch):
    DemandExpect = np.array(self.DemandExpect)
    DemandExpect = np.array(np.random.normal(DemandExpect, DemandExpect * 0.2) + 0.5, dtype=int)
    DemandExpect = np.maximum(DemandExpect, 0)

    state = np.zeros((KAPPA + 1) * 2 + 1)
    neighbour = cluster.Neighbor
    state[0] = cluster.ID
    state[9] = int(self.SupplyExpect[cluster.ID] + len(cluster.IdleVehicles))
    state[10] = int(DemandExpect[cluster.ID])
    for nc in neighbour:
        id = nc.ID
        idx = self.Getidx(cluster.ID, id)
        state[idx * 2 + 1] = int(self.SupplyExpect[id] + len(nc.IdleVehicles))
        state[idx * 2 + 2] = int(DemandExpect[id])
    pre_part = []
    for pre_idx in pre_idxs_batch:
        pre_part += list(pre_idx)
    pre_part = np.array(pre_part)
    state = np.concatenate([state, pre_part], axis=0)
    if len(state) < (KAPPA + 1) * 2 + 1 + BATCH_SIZE * 9:
        sup_len = (KAPPA + 1) * 2 + 1 + BATCH_SIZE * 9 - len(state)
        sup = np.zeros(sup_len)
        state = np.concatenate([state, sup])
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

        epsilon = 0

        total_reward = 0
        step = 0
        self.reject = 0

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

                v_num = len(cluster.IdleVehicles)

                batch_num = v_num // BATCH_SIZE
                last_batch_size = v_num % BATCH_SIZE
                if last_batch_size:
                    batch_num += 1
                if batch_num and last_batch_size == 0:
                    last_batch_size = BATCH_SIZE

                for i_batch in range(batch_num):
                    mask = 0
                    if i_batch == batch_num - 1:
                        mask = last_batch_size

                    pre_idxs_batch = []
                    idxs_batch = []
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

                    state = GetStateFunction(self, cluster, pre_idxs_batch)
                    joint_action = DispatchFunction(self, state, actor, episode, epsilon)
                    if i_batch == batch_num - 1:
                        for h in range(last_batch_size, len(joint_action)):
                            joint_action[h] = -1
                    pre_reward = 0
                    dest = []
                    count = 0
                    for v_idx in range(BATCH_SIZE):
                        if mask and v_idx == mask:
                            break
                        self.DispatchNum += 1
                        count += 1
                        act = joint_action[v_idx]
                        vehicle = cluster.IdleVehicles[v_idx]
                        self.move(vehicle, act, cluster, pre_idxs_batch[v_idx], idxs_batch[v_idx])
                        dest.append(vehicle.Cluster.ID)
                        pre_reward += pre_idxs_batch[v_idx][act]

                    pre_reward /= count
                    neighbors = cluster.Neighbor
                    n_after = []
                    for neighbor in neighbors:
                        id = neighbor.ID
                        n_after.append(self.SupplyExpect[id] - self.DemandExpect[id])

                    after_var = torch.tensor(n_after, dtype=torch.float32).var()
                    before = torch.zeros(72)
                    n_before = []
                    for d_cluster in dest:
                        before[d_cluster] -= 1

                    for neighbor in neighbors:
                        before[neighbor.ID] += self.SupplyExpect[neighbor.ID] - self.DemandExpect[neighbor.ID]
                        n_before.append(before[neighbor.ID])

                    before_var = torch.tensor(n_before, dtype=torch.float32).var()
                    reward = (float(before_var - after_var)) + pre_reward
                    total_reward += reward
                    step_reward += reward
                    new_state = GetStateFunction(self, cluster, pre_idxs_batch)
                    if self.RealExpTime != EndTime:
                        state_v = torch.tensor(state, dtype=torch.float32).to(device)
                        joint_action_v = torch.tensor(joint_action, dtype=torch.float32).to(device)
                        reward_v = torch.tensor(reward, dtype=torch.float32).to(device)
                        new_state_v = torch.tensor(new_state, dtype=torch.float32).to(device)
                        minibatch.append([state_v, joint_action_v, reward_v, new_state_v])
                        if len(minibatch) == 10:
                            LearningFunction(self, minibatch, actor, critic, actor_opt,
                                             critic_opt,
                                             device)
                            minibatch = []
                    cluster.IdleVehicles = cluster.IdleVehicles[BATCH_SIZE::]

            if episode % 50 == 0:
                writer.add_scalar('Episode_' + str(episode) + '_reward_per_step', step_reward, step)

            step += 1
            self.RealExpTime += self.TimePeriods

        if episode % 10 == 0:
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
        print('reject rate: ', self.reject / self.DispatchNum)

        writer.add_scalar('PRE_reject', self.reject / self.DispatchNum, episode)
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
save_path = os.path.join("../", "Models","A2C_joint_diff_7", "_", "saves_episode", "VeRL0-" + args.name)
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
