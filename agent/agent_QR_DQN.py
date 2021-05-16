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

N_QUANT = 200
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]
QUANTS_TARGET = (np.linspace(0.0, 1.0, N_QUANT + 1)[:-1] + QUANTS) / 2

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
        self.learning_start = 2000
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
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
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

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id]['lane_vehicle_num'])
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
            actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

        return actions

    def get_action(self, ob):

        # The epsilon-greedy action selector.

        if np.random.rand() <= self.epsilon:
            return self.sample()
        ob = torch.tensor(ob, dtype=torch.float32)
        act_values =  self.model(ob.reshape(1, -1)).mean(dim=2)
        # ob = self._reshape_ob(ob)
        # act_values = self.model.predict([ob])
        return torch.argmax(act_values[0]).item()

    def sample(self):
        # Random samples
        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        # return BaseModel(input_dim=self.ob_length, output_dim=self.action_space)
        return Base_QR_DQN_Model(input_dim=self.ob_length, output_dim=self.action_space, n_quant=N_QUANT)

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
        b_w, b_idxes = np.ones_like(rewards), None

        obs, actions, rewards, next_obs = torch.FloatTensor(obs), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.FloatTensor(next_obs)
        q_eval = self.model(obs)  # (m, N_ACTIONS, N_QUANT)
        bsz = q_eval.shape[0]
        q_eval = torch.stack([q_eval[i].index_select(0, actions[i]) for i in range(bsz)]).squeeze(1)
        # (m, N_QUANT)
        q_eval = q_eval.unsqueeze(2) # (m, N_QUANT, 1)
        # note that dim 1 is for present quantile, dim 2 is for next quantile

        # get next state value
        q_next = self.target_model(next_obs).detach() # (m, N_ACTIONS, N_QUANT)
        best_actions = q_next.mean(dim=2).argmax(dim=1) # (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(bsz)]).squeeze(1)
        # (m, N_QUANT)
        q_target = rewards.unsqueeze(1) + self.gamma * q_next
        # (m, N_QUANT)
        q_target = q_target.unsqueeze(1)  # (m , 1, N_QUANT)

        # quantile Huber loss
        u = q_target.detach() - q_eval # (m, N_QUANT, N_QUANT)
        tau = torch.FloatTensor(QUANTS_TARGET).view(1, -1, 1) # (1, N_QUANT, 1)
        # note that tau is for present quantile
        weight = torch.abs(tau - u.le(0.).float()) # (m, N_QUANT, N_QUANT)
        loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
        # (m, N_QUANT, N_QUANT)
        loss = torch.mean(weight * loss, dim=1).mean(dim=1)

        # calc importance weighted loss
        b_w = torch.Tensor(b_w)
        loss = torch.mean(b_w * loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self.model.fit([obs], target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.ckpt".format(step)
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

