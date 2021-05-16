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

class Base_QR_DQN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, n_quant):
        super(Base_QR_DQN_Model, self).__init__()
        self.n_quant = n_quant
        self.n_action = output_dim
        self.linear1 = nn.Linear(input_dim, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, output_dim * n_quant)

    def forward(self, ob):
        x = self.relu(self.linear1(ob))
        return self.linear2(x).view(-1, self.n_action, self.n_quant)


# class FRAPModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FRAPModel, self).__init__()
#
#     def forward(self, ob, last_phase):
#
#         return ob


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


