import pickle
import gym

from pathlib import Path
import pickle
import gym
import torch


from utils import *



# how to import or load local files

import os
import sys
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
with open(path + "/gym_cfg.py", "r") as f:
    pass

class TestAgent():
    def __init__(self):
        self.now_phase = {}
        self.green_sec = 40
        self.max_phase = 8
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        self.intersections = {}
        self.roads = {}
        self.agents = {}

        self.obs_length = 24
        self.n_action = 8

        self.actor= Actor(self.obs_length,self.n_action)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(self.obs_length)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=3e-4)



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
        probs = {}

        #print("this is list",self.agent_list)
        for agent_id in self.agent_list:
            action,prob = self.get_action(observations_for_agent[agent_id])
            actions[agent_id] = action
            probs[agent_id] = prob
        return actions,probs


    def get_action(self,obs):
        # here the input should be the number of vechile in each lane
        #state_dim = len(obs)
        # here the action_number is pre-defined
        #n_actions = 8
        #actor = Actor(state_dim, n_actions, activation=Mish)
        #critic = Critic(state_dim, activation=Mish)

        #adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
        obs = list_to_tensor(obs)
        #obs = obs[1:]
        probs = self.actor(obs)
        action = torch.argmax(probs.detach())

        return action,probs[action]

    # def act(self, obs):
    #     """ !!! MUST BE OVERRIDED !!!
    #     """
    #     # here obs contains all of the observations and infos
    #
    #     # observations is returned 'observation' of env.step()
    #     # info is returned 'info' of env.step()
    #     observations = obs['observations']
    #     info = obs['info']
    #     actions = {}
    #
    #     # a simple fixtime agent
    #
    #     # preprocess observations
    #     observations_for_agent = {}
    #     for key,val in observations.items():
    #         observations_agent_id = int(key.split('_')[0])
    #         observations_feature = key[key.find('_')+1:]
    #         if(observations_agent_id not in observations_for_agent.keys()):
    #             observations_for_agent[observations_agent_id] = {}
    #         observations_for_agent[observations_agent_id][observations_feature] = val
    #
    #     # get actions
    #     for agent in self.agent_list:
    #         # select the now_step
    #         for k,v in observations_for_agent[agent].items():
    #             now_step = v[0]
    #             break
    #         step_diff = now_step - self.last_change_step[agent]
    #         if(step_diff >= self.green_sec):
    #             self.now_phase[agent] = self.now_phase[agent] % self.max_phase + 1
    #             self.last_change_step[agent] = now_step
    #
    #
    #         actions[agent] = self.now_phase[agent]
    #     # print(self.intersections,self.roads,self.agents)
    #     return actions

    def act_replay(self,act_loss):
        tmp_loss = 0
        for agent_id in self.agent_list:
            tmp_loss = act_loss[agent_id] + tmp_loss
        self.actor_optimizer.zero_grad()
        tmp_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def cri_replay(self,critic_loss):
        tmp_loss_c = 0
        for agent_id in self.agent_list:
            tmp_loss_c = critic_loss[agent_id] + tmp_loss_c
        self.critic_optimizer.zero_grad()
        tmp_loss_c.backward()
        self.critic_optimizer.step()



    def load_model(self, dir="model/ppo", step=0):
        name = "ppo_agent_actor_{}.ckpt".format(step)
        actor_name = os.path.join(dir, name)
        print("load from " + actor_name)
        #self.model.load_weights(model_name)
        self.actor.load_state_dict(torch.load(actor_name))

        name = "ppo_agent_critic_{}.ckpt".format(step)
        critic_name = os.path.join(dir,name)
        print("load from" + critic_name)
        self.critic.load_state_dict(torch.load(critic_name))

    def save_model(self, dir="model/ppo", step=0):
        act_name = "ppo_agent_actor_{}.ckpt".format(step)
        actor_name = os.path.join(dir, act_name)
        #self.model.save_weights(model_name)
        torch.save(self.actor.state_dict(),actor_name)

        cri_name = "ppo_agent_critic_{}.ckpt".format(step)
        critic_name = os.path.join(dir,cri_name)
        torch.save(self.critic.state_dict(),critic_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()

