import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs import *


class BaseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, output_dim)

    def forward(self, ob):
        x = self.relu(self.linear1(ob))
        return self.linear2(x)

class CoLightModel(nn.Module):
    def __init__(self, input_dim, output_dim, agent_list=None):
        super(CoLightModel, self).__init__()
        self.observation_proj = nn.Linear(input_dim, 32)
        self.obs_inner_agent_proj = nn.Linear(32, 16)
        self.obs_outer_agent_proj = nn.Linear(32, 16)
        self.cooperation_proj = nn.Linear(16, 16)
        self.final_proj = nn.Linear(16, output_dim)

    def forward(self, obs):
        '''
        :param obs:
        "ob_embedding": embedding for current intersection observation, torch.tensor
        "lane_vehicle_num": no need to explain
        "lane_vehicle_speed": no need to explain
        "adjacency": the ob_embedding of adjacency intersections, torch.tensor
        :return: action
        '''

        print("obs:", obs)

        inner_ob_embedding = self.obs_inner_agent_proj(obs['ob_embedding'])

        interaction_scores = []

        for adj_ob_embedding in obs['adjacency']:
            interaction_score = torch.dot(inner_ob_embedding.reshape(-1), self.obs_outer_agent_proj(adj_ob_embedding).reshape(-1))
            interaction_scores.append(interaction_score)
        interaction_scores = torch.stack(interaction_scores)
        interaction_scores = F.softmax(interaction_scores, dim=0)

        adj_ob_embeddings = torch.stack(obs['adjacency'])
        cooperation_adj = torch.sum(self.cooperation_proj(adj_ob_embeddings) * interaction_scores.unsqueeze(1), dim=0)
        self.final_proj(cooperation_adj.unsqueeze(0))




        # act, attention = self.action_att_predict([obs])
        # return act[0], attention[0]



    # def action_att_predict(self, state, total_features=None, total_adjs=None, bar=False):
    #     batch_size = len(state)
    #     if total_features is None and total_adjs is None:
    #         total_features, total_adjs = [], []
    #         for i in range(batch_size):
    #             feature, adj = [], []
    #             for agent in range(self.num_agents):
    #                 observation = [self.last_change_step[agent]]
    #                 for name, value in state[i].items():
    #                     if "vehicle_num" in name:
    #                         observation.extend(value[1:])
    #                 feature.append(observation)
    #                 adj.append(state[i][agent]['adjacency_matrix'])
    #             total_features.append(feature)
    #             total_adjs.append(adj)
    #         total_features = np.reshape(np.array(total_features), [batch_size, self.num_agents, -1])
    #         total_adjs = self.adjacency_index2matrix(np.array(total_adjs))
    #     if bar:
    #         all_output = self.q_network_bar.predict([total_features, total_adjs])
    #     else:
    #         all_output = self.q_network([total_features, total_adjs])
    #     action, attention = all_output[0], all_output[1]
    #
    #     if len(action) > 1:
    #         return total_features, total_adjs, action, attention
    #
    #     max_action = np.extend_dims(np.argmax(action, acis=-1), axis=-1)
    #     random_action = np.reshape(np.random.randint(self.num_actions, size=1 * self.num_agents),
    #                                (1, self.num_agents, 1))
    #     # [batch,agent,2]
    #     possible_action = np.concatenate([max_action, random_action], axis=-1)
    #     selection = np.random.choice(
    #         [0, 1],
    #         size=batch_size * self.num_agents,
    #         p=[1 - dic_agent_conf["EPSILON"], dic_agent_conf["EPSILON"]])
    #     act = possible_action.reshape((batch_size * self.num_agents, 2))[
    #         np.arange(batch_size * self.num_agents), selection]
    #     act = np.reshape(act, (batch_size, self.num_agents))
    #     return act, attention



