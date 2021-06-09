""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""

import os

import torch
import torch.nn.functional as F

path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random
# from model import BaseModel, FRAPModel

import os
from collections import deque
from configs import *
import numpy as np


# contains all of the intersections
class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=(3), padding=(1))
        self.flatten = nn.Flatten()
        # self.embedding = nn.Embedding(50, 64)
        self.linear1 = nn.Linear(input_dim, 1024)
        # self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1024)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(1024, output_dim)

    def forward(self, ob):
        # idx = torch.where(ob >= 50)
        # ob[idx] = 49
        #
        # assert (ob < 50).all()
        # assert (ob >= 0).all()
        # ob: (bsz, 72)
        # x = self.embedding(ob) # x: (bsz, 72, 64)
        # x = self.relu(x)
        x = self.linear1(ob)  # (bsz, 72, 32)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear2(x)
        x = self.linear3(self.relu(x))

        return x

class TestAgent():
    def __init__(self):

        # DQN parameters

        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        self.memory = deque(maxlen=2000)
        self.learning_start = 0
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 32
        self.ob_length = 24

        self.action_space = 8

        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        path = os.path.split(os.path.realpath(__file__))[0]
        self.load_model(path, 105)
        self.target_model = self._build_model()
        self.update_target_network()

        self.last_observations = None


    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id
    def load_roadnet(self,intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
    ################################

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        if MODEL_NAME == 'MLP':
            actions = {}
            for agent_id in self.agent_list:
                action = self.get_action(np.concatenate([
                    observations_for_agent[agent_id]['lane_vehicle_num'],
                    observations_for_agent[agent_id]['lane_speed']
                ]))
                actions[agent_id] = action
                self.last_change_step[agent_id] = action
        else:
            actions = {}
            for agent_id in self.agent_list:
                # lane_vehicle_num = observations_for_agent[agent_id]['lane_vehicle_num']
                # pressures = []
                # for i in range(8):
                #     pressure_i = lane_vehicle_num[FRAP_intersections[i][1]] - lane_vehicle_num[FRAP_intersections[i][0]]
                #     pressures.append([pressure_i, Phase_to_FRAP_Phase[self.last_change_step[agent_id]][i]])

                action = self.get_action(observations_for_agent[agent_id])
                if isinstance(action, int): self.last_change_step[agent_id] = action
                else: self.last_change_step[agent_id] = action.item()
                actions[agent_id] = action

        return actions


    def get_phase_pressures(self, lane_vehicle_num):
        pressures = []
        for i in range(8):
            in_lanes = self.phase_lane_map_in[i]
            out_lanes = self.phase_lane_map_out[i]
            pressure = 0
            for in_lane in in_lanes:
                pressure += lane_vehicle_num[in_lane] * 3
            for out_lane in out_lanes:
                pressure -= lane_vehicle_num[out_lane]
            pressures.append(pressure)
        # # print("pressures: ", pressures)
        return pressures

    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                phase_id += 1
                if lane in in_lanes and phase_id not in unavailable_phases:
                    unavailable_phases.append(phase_id)

        return unavailable_phases

    def get_action_pressure(self, lane_vehicle_num):
        pressures = self.get_phase_pressures(lane_vehicle_num)
        unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)

        max_pressure_id = np.argmax(pressures) + 1
        while (max_pressure_id in unavailable_phases):
            pressures[max_pressure_id - 1] -= 999999
            max_pressure_id = np.argmax(pressures) + 1
        # # print(max_pressure_id)
        return max_pressure_id


    def act(self, obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]


        if self.last_observations is None:
            for agent in self.agent_list:
                # select the now_step
                for k, v in observations_for_agent[agent].items():
                    now_step = v[0]
                    break
                lane_vehicle_num = observations_for_agent[agent]["lane_vehicle_num"]
                # print("agent id: ", agent)
                # print("lane vehicle: ", lane_vehicle_num)

                action = self.get_action_pressure(lane_vehicle_num)

                step_diff = now_step - self.last_change_step[agent]
                if (step_diff >= self.green_sec):
                    self.now_phase[agent] = action
                    self.last_change_step[agent] = now_step

                actions[agent] = self.now_phase[agent]

            self.last_observations = observations

        else:
            for agent in self.agent_list:
                self.last_observations


        # Get actions
        # if MODEL_NAME == 'MLP':
        #     for agent in self.agent_list:
        #         self.epsilon = 0
        #         actions[agent] = self.get_action(np.concatenate([
        #             observations_for_agent[agent]['lane_vehicle_num'],
        #             observations_for_agent[agent]['lane_speed']
        #         ]))
        # else:
        #     for agent_id in self.agent_list:
        #         lane_vehicle_num = observations_for_agent[agent_id]['lane_vehicle_num']
        #         pressures = []
        #         for i in range(8):
        #             pressure_i = lane_vehicle_num[FRAP_intersections[i][1]] - lane_vehicle_num[FRAP_intersections[i][0]]
        #             pressures.append([pressure_i, Phase_to_FRAP_Phase[self.last_change_step[agent_id]][i]])
        #
        #         action = self.get_action(pressures) + 1
        #         actions[agent_id] = action

        return actions

    def get_action(self, ob):

        # The epsilon-greedy action selector.

        if np.random.rand() <= self.epsilon:
            return self.sample()
        ob = torch.tensor(ob, dtype=torch.float32)
        if MODEL_NAME == 'MLP':
            act_values = self.model(ob.reshape(1, -1).cuda())
        else:
            act_values = self.model(ob.reshape(1, -1).cuda())
        # ob = self._reshape_ob(ob)
        # act_values = self.model.predict([ob])
        return torch.argmax(act_values[0]).item()

    def sample(self):
        # Random samples
        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        if MODEL_NAME == 'MLP':
            return BaseModel(input_dim=72 + 8 + 8, output_dim=self.action_space).cuda()
        else:
            return FRAPModel(relations=relations)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        # Update the Q network from the memory buffer.

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        obs, actions, rewards, next_obs, = [np.stack(x) for x in np.array(minibatch).T]
        output = self.target_model(torch.tensor(next_obs, dtype=torch.float32).cuda())
        target = rewards + self.gamma * np.amax(output.detach().cpu().numpy(), axis=1)
        target_f = self.model(torch.tensor(obs, dtype=torch.float32).cuda()).detach()

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]

        pred_target = self.model(torch.tensor(obs, dtype=torch.float32).cuda())
        loss = F.mse_loss(pred_target, target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.model.fit([obs], target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", step=0):
        name = f"../supervise_model/model_{step}_48_MLP.ckpt"
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.ckpt".format(step)
        model_name = os.path.join(dir, name)
        torch.save(self.model.state_dict(), model_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

