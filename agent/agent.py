import os

import torch
import torch.nn.functional as F

path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random
from model import BaseModel, Base_QR_DQN_Model

import os
from collections import deque
import numpy as np
from configs import *

N_QUANT = 200
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]
QUANTS_TARGET = (np.linspace(0.0, 1.0, N_QUANT + 1)[:-1] + QUANTS) / 2


FRAP_intersections = [[11, 17],
                      [4, 19],
                      [2, 20],
                      [7, 22],
                      [5, 23],
                      [10, 13],
                      [8, 14],
                      [1, 18]]
Phase_to_FRAP_Phase = {
    0: [0, 0, 0, 1, 0, 0, 0, 1],
    1: [0, 0, 1, 0, 0, 0, 1, 0],
    2: [0, 1, 0, 0, 0, 1, 0, 0],
    3: [1, 0, 0, 0, 1, 0, 0, 0],
    4: [0, 0, 1, 0, 0, 0, 0, 1],
    5: [0, 1, 0, 0, 1, 0, 0, 0],
    6: [0, 0, 0, 1, 0, 0, 1, 0],
    7: [1, 0, 0, 0, 0, 1, 0, 0]
}

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
        self.learning_start = 200
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

        self.rotate_matrix = torch.zeros([8, 8])
        self.inverse_clockwise_mapping = {
            1: 3,
            2: 4,
            3: 1,
            4: 2,
            5: 8,
            6: 5,
            7: 6,
            8: 7
        }

        self.clockwise_mapping = {
            y: x for x,y in self.inverse_clockwise_mapping.items()
        }

        for i in range(8):
            self.rotate_matrix[i][self.inverse_clockwise_mapping[i+1]-1] = 1

        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        path = os.path.split(os.path.realpath(__file__))[0]
        self.load_model(path, 99)
        self.target_model = self._build_model()
        self.update_target_network()



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
                action = self.get_action(observations_for_agent[agent_id]['lane_vehicle_num'])
                actions[agent_id] = action
            return actions
        else:
            actions = {}
            for agent_id in self.agent_list:
                lane_vehicle_num = observations_for_agent[agent_id]['lane_vehicle_num']
                pressures = []
                for i in range(8):
                    pressure_i = lane_vehicle_num[FRAP_intersections[i][1]] - lane_vehicle_num[FRAP_intersections[i][0]]
                    pressures.append([pressure_i, Phase_to_FRAP_Phase[self.last_change_step[agent_id]][i]])

                action = self.get_action(pressures)

                actions[agent_id] = action

            return actions

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

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            if with_Speed:
                action = self.get_action(np.concatenate(observations_for_agent[agent]['lane_vehicle_num'],
                                                        observations_for_agent[agent]['lane_speed']))
            else:
                action = self.get_action(observations_for_agent[agent]['lane_vehicle_num'])
            actions[agent] = action + 1
        return actions

    def get_action(self, ob):

        # The epsilon-greedy action selector.
        if np.random.rand() <= self.epsilon:
            return self.sample()

        ob = torch.tensor(ob, dtype=torch.float32)

        if MODEL_NAME == 'MLP':
            act_values =  self.model(ob.reshape(1, -1).cuda()).mean(dim=2)
        else:
            act_values = self.model(ob.cuda())
        # ob = self._reshape_ob(ob)
        # act_values = self.model.predict([ob])
        return torch.argmax(act_values[0]).item()

    def sample(self):
        # Random samples
        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        # return BaseModel(input_dim=self.ob_length, output_dim=self.action_space)
        if with_Speed:
            return Base_QR_DQN_Model(input_dim=self.ob_length * 2, output_dim=self.action_space, n_quant=N_QUANT).cuda()
        else:
            return Base_QR_DQN_Model(input_dim=self.ob_length, output_dim=self.action_space, n_quant=N_QUANT).cuda()


    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def obs_clock_wise_rotate(self, obs):
        new_obs = obs.clone()
        new_obs[3: 12] = obs[0: 9]
        new_obs[0: 3] = obs[9: 12]
        new_obs[15: 24] = obs[12: 21]
        new_obs[12: 15] = obs[21: 24]
        return new_obs

    def action_clock_wise_rotate(self, action):
        # action.shape: [bsz, n_actions, k] k = N_QUANT for QR-DQN, and k = 1 for DQN
        return torch.einsum('ij,bjn->bin', (self.rotate_matrix, action.cpu())).cuda()

    def QR_DQN_loss(self, obs, actions, rewards, next_obs, b_w, b_idxes):
        q_eval = self.model(obs)  # (m, N_ACTIONS, N_QUANT)
        bsz = q_eval.shape[0]
        q_eval = torch.stack([q_eval[i].index_select(0, actions[i]) for i in range(bsz)]).squeeze(1)
        # (m, N_QUANT)
        q_eval = q_eval.unsqueeze(2)  # (m, N_QUANT, 1)
        # note that dim 1 is for present quantile, dim 2 is for next quantile

        # get next state value
        q_next = self.target_model(next_obs).detach()  # (m, N_ACTIONS, N_QUANT)
        best_actions = q_next.mean(dim=2).argmax(dim=1)  # (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(bsz)]).squeeze(1)
        # (m, N_QUANT)
        q_target = rewards.unsqueeze(1) + self.gamma * q_next
        # (m, N_QUANT)
        q_target = q_target.unsqueeze(1)  # (m , 1, N_QUANT)

        # quantile Huber loss
        u = q_target.detach() - q_eval  # (m, N_QUANT, N_QUANT)
        tau = torch.FloatTensor(QUANTS_TARGET).view(1, -1, 1).cuda()  # (1, N_QUANT, 1)
        # note that tau is for present quantile
        weight = torch.abs(tau - u.le(0.).float())  # (m, N_QUANT, N_QUANT)
        loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
        # (m, N_QUANT, N_QUANT)
        loss = torch.mean(weight * loss, dim=1).mean(dim=1)

        # calc importance weighted loss
        b_w = torch.Tensor(b_w).cuda()
        loss = torch.mean(b_w * loss)
        return loss

    def replay(self):
        # Update the Q network from the memory buffer.

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, = [np.stack(x) for x in np.array(minibatch).T]
        b_w, b_idxes = np.ones_like(rewards), None
        obs, actions, rewards, next_obs = torch.FloatTensor(obs).cuda(), torch.LongTensor(actions).cuda(), \
                                          torch.FloatTensor(rewards).cuda(), torch.FloatTensor(next_obs).cuda()
        loss = self.QR_DQN_loss(obs, actions, rewards, next_obs, b_w, b_idxes)
        actions_1 = torch.LongTensor([self.clockwise_mapping[x.item() + 1] - 1 for x in actions.flatten()]).reshape(actions.shape).cuda()
        actions_2 = torch.LongTensor([self.clockwise_mapping[x.item() + 1] - 1 for x in actions_1.flatten()]).reshape(actions.shape).cuda()
        actions_3 = torch.LongTensor([self.clockwise_mapping[x.item() + 1] - 1 for x in actions_2.flatten()]).reshape(actions.shape).cuda()

        obs_1 = self.obs_clock_wise_rotate(obs).detach()
        obs_2 = self.obs_clock_wise_rotate(obs_1).detach()
        obs_3 = self.obs_clock_wise_rotate(obs_2).detach()

        next_obs_1 = self.obs_clock_wise_rotate(next_obs).detach()
        next_obs_2 = self.obs_clock_wise_rotate(next_obs_1).detach()
        next_obs_3 = self.obs_clock_wise_rotate(next_obs_2).detach()

        loss += self.QR_DQN_loss(obs_1, actions_1, rewards, next_obs_1, b_w, b_idxes)
        loss += self.QR_DQN_loss(obs_2, actions_2, rewards, next_obs_2, b_w, b_idxes)
        loss += self.QR_DQN_loss(obs_3, actions_3, rewards, next_obs_3, b_w, b_idxes)

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.model.fit([obs], target_f, epochs=1, verbose=0)
        # unsupervised learning
        # q_eval = self.model(obs).detach()
        # q_eval_1_target = self.action_clock_wise_rotate(q_eval)
        # q_eval_2_target = self.action_clock_wise_rotate(q_eval_1_target)
        # q_eval_3_target = self.action_clock_wise_rotate(q_eval_2_target)
        #
        # obs_1 = self.obs_clock_wise_rotate(obs)
        # obs_2 = self.obs_clock_wise_rotate(obs_1)
        # obs_3 = self.obs_clock_wise_rotate(obs_2)
        #
        # q_eval_1 = self.model(obs_1)
        # q_eval_2 = self.model(obs_2)
        # q_eval_3 = self.model(obs_3)
        #
        # loss += (F.mse_loss(q_eval_1, q_eval_1_target)
        #     + F.mse_loss(q_eval_2, q_eval_2_target)
        #     + F.mse_loss(q_eval_3, q_eval_3_target)) * 0.05

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", step=0):
        name = "qr_dqn_agent_{}.ckpt".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/dqn", step=0):
        name = "qr_dqn_agent_{}.ckpt".format(step)
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

