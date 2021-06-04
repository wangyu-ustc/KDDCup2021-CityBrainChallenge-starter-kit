import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs import *
from lib import dqn_model, common


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
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(32, output_dim * n_quant)

    def forward(self, ob):
        x = self.relu(self.linear1(ob))
        x=  torch.tanh(self.linear2(x))
        return self.linear3(x).view(-1, self.n_action, self.n_quant)

class Noisy_QR_DQN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, n_quant):
        super(Noisy_QR_DQN_Model, self).__init__()
        self.n_quant = n_quant
        self.n_action = output_dim
        self.linear1 = dqn_model.NoisyLinear(input_dim, 64)
        self.linear2 = dqn_model.NoisyLinear(64, 32)
        self.relu = nn.ReLU()
        self.linear3 = dqn_model.NoisyLinear(32, output_dim * n_quant)

    def forward(self, ob):
        x = self.relu(self.linear1(ob))
        x=  torch.tanh(self.linear2(x))
        return self.linear3(x).view(-1, self.n_action, self.n_quant)



class RainbowDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RainbowDQN, self).__init__()
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(input_dim, 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, N_ATOMS)
        )
        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(input_dim, 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, output_dim * N_ATOMS)
        )
        self.register_buffer("supports", torch.arange(Vmin, Vmax + DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        val_out = self.fc_val(x).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(x).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


class FRAPModel(nn.Module):
    def __init__(self, relations):
        super(FRAPModel, self).__init__()
        self.demand_embedding = nn.Linear(2, 4)
        self.conv2d1 = nn.Conv2d(in_channels=8, out_channels=20, kernel_size=(1,1),stride=(1,1))
        self.relations = torch.tensor(relations)  # (1, 8, 7)
        self.relation_embedding = nn.Embedding(2, 4)
        self.relation_conv2d = nn.Conv2d(in_channels=4, out_channels=20, kernel_size=(1,1), stride=(1,1))
        self.conv2d2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(1,1), stride=(1,1))
        self.conv2d3 = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=(1,1), stride=(1,1))

    def forward(self, ob):
        '''
        :param ob: [bsz, 8, 2]
        :return: distribution of actions
        '''
        bsz = ob.shape[0]
        demands = self.demand_embedding(ob.reshape(-1, 2)).reshape(bsz, 8, 4)
        # demand: [bsz, 8, 4]
        pair_representation = torch.zeros([bsz, 8, 7, 8])
        for i, demand in enumerate(demands):
            # demand: [8, 4]
            for s in range(8):
                count = 0
                for k in range(8):
                    if k == s: continue
                    pair_representation[i][s][count] = torch.cat([demand[s], demand[k]])
                    count += 1

        # pair_representation: (bsz, 8, 7, 8) -> (bsz, 8, 8, 7) --> x: (bsz, 20, 8, 7)
        x = self.conv2d1(pair_representation.permute(0, 3, 1, 2))

        # Phase competition mask
        phase_cometition_mask = self.relation_embedding(self.relations).permute(0, 3, 1, 2) # (1, 4, 8, 7)
        phase_cometition_mask = self.relation_conv2d(phase_cometition_mask) # (1, 20, 8, 7)

        x = x * phase_cometition_mask # (bsz, 20, 8, 7)
        x = self.conv2d2(x)
        x = self.conv2d3(x) # (bsz, 1, 8, 7)
        x = x.squeeze() # (bsz, 8, 7)
        x = torch.sum(x, dim=-1)
        return x


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


