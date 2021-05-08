import time
import torch
import torch.nn as nn
import torch.optim as optim
from agent import TestAgent
import numpy as np
from configs import *
import os


class RepeatVector3d(nn.Module):
    def __init__(self, times):
        super(RepeatVector3d, self).__init__()
        self.times = times

    def forward(self, inputs):
        return inputs.unsqueeze(1).repeat(1, self.times, 1, 1)



def MLP(In_0, layers=[128, 128]):
    linears = nn.ModuleList()
    for layer_index, layer_size in enumerate(layers):
        if layer_index == 0:
            linears.append(nn.Linear(In_0, layer_size))
        else:
            linears.append(nn.Linear(layers[layer_index-1], layer_size))
    return linears

class MultiHeadsAttModel(nn.Module):
    def __init__(self, num_agents, num_neighbors, dv, nv, d, dout):
        super(MultiHeadsAttModel, self).__init__()
        self.d = d
        self.dv = dv
        self.nv = nv
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.RepeatVector3D = RepeatVector3d(num_agents)
        self.agent_proj = nn.Linear(d, dv*nv)
        self.neighbor_proj = nn.Linear(d, dv*nv)
        self.embedding = nn.Linear(self.num_neighbors, dv*nv)
        self.out_proj = nn.Linear(dv, dout)

    def forward(self, In_agent, In_neighbor):
        agent_repr = In_agent.reshape(-1, self.num_agents, 1, self.d)
        neighbor_repr = self.RepeatVector3D(In_agent)
        neighbor_repr = torch.bmm(In_neighbor, neighbor_repr)
        agent_repr_head = self.agent_proj(agent_repr).reshape(-1, self.num_agents, 1, dv, nv)
        agent_repr_head = agent_repr_head.permute(0, 1, 4, 2, 3)
        neighbor_repr_head = self.neighbor_proj(neighbor_repr)

        att = torch.softmax(torch.bmm(agent_repr_head, neighbor_repr_head), -1)
        att_record = att.reshape(self.num_agents, self.nv, self.num_neighbors)

        # self embedding again
        neighbor_hidden_repr_head = self.embedding(neighbor_repr).reshape(self.num_agents, self.num_neighbors, self.dv, self.nv)
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.permute(0, 1, 4, 2, 3)
        out = torch.bmm(att, neighbor_hidden_repr_head).reshape(self.num_agents. self.dv)
        out = self.out_proj(out)

        return out, att_record

class Q_network(nn.Module):
    def __init__(self, num_agents, len_feature, MLP_layers, CNN_layers, CNN_heads, num_actions):
        super(Q_network).__init__()
        self.MLP = MLP(len_feature, MLP_layers)
        self.num_agents = num_agents
        self.CNN_layers = CNN_layers
        self.MultiHeadsAttModel = nn.ModuleList()
        for CNN_layer_index, CNN_layer_size in enumerate(CNN_layers):
            print("CNN_heads[CNN_layer_index]:", CNN_heads[CNN_layer_index])
            self.MultiHeadsAttModel.append(MultiHeadsAttModel(
                num_agents = num_agents,
                num_neighbors = self.num_neighbors,
                d = MLP_layers[-1],
                dv = CNN_layer_size[0],
                dout = CNN_layer_size[1],
                nv = CNN_heads(CNN_layer_index),
            ))

        self.out_proj = nn.Linear(MLP_layers[-1], num_actions)

    def forward(self, In):
        att_record_all_layers = []
        feature = self.MLP(In[0])
        for i, layer in enumerate(self.MultiHeadsAttModel):
            if i == 0:
                h, att_record = layer(feature, In[1])
            else:
                h, att_record = layer(h, In[1])

            att_record_all_layers.append(att_record)

        if len(self.CNN_layers) > 1:
            att_record_all_layers=torch.cat(att_record_all_layers, dim=1)
        else:
            att_record_all_layers = att_record_all_layers[0]

        att_record_all_layers=att_record_all_layers.reshape(-1, len(self.CNN_layers), self.num_agents)
        out = self.out_proj(h)
        return out, att_record_all_layers





class ColightAgent(TestAgent):
    def __init__(self, num_agents, cnt_round=None, best_round=None, bar_round=None,
                 intersection_id='0'):
        super(ColightAgent, self).__init__()

        self.CNN_layers = [[32, 32]]
        self.num_agents = num_agents
        self.num_neighbors = min(dic_traffic_env_conf['TOP_K_ADJACENCY'], self.num_agents)
        self.intersection_id = intersection_id
        self.vec = np.zeros((1, self.num_neighbors))
        self.vec[0][0] = 1

        self.num_actions = len(dic_traffic_env_conf['PHASE'][dic_traffic_env_conf['SIMULATOR_TYPE']])
        self.num_lanes = np.sum(np.array(list(dic_traffic_env_conf['LANE_NUM'].values())))
        self.len_feature = self.compute_len_feature()
        self.memory = self.build_memory()

        if cnt_round == 0:
            self.q_network = self.build_network()
            if os.listdir(dic_path["PATH_TO_MODEL"]):
                self.q_network.load_dict(torch.load(os.path.join(dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.ckpt")))
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                if best_round:
                    self.load_network("round_{0}_inter_{1}".format(best_round,self.intersection_id))

                    if bar_round and bar_round != best_round and cnt_round > 10:
                        self.load_network_bar("round_{0}_inter_{1}".format(bar_round, self.intersection_id))
                    else:
                        if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                            if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                                self.load_network_bar("round_{0}".format(
                                    max((best_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                        "UPDATE_Q_BAR_FREQ"], 0),
                                    self.intersection_id))
                            else:
                                self.load_network_bar("round_{0}_inter_{1}".format(
                                    max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                    self.intersection_id))
                        else:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max(best_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))

                else:
                    # not use model pool
                    # TODO how to load network for multiple intersections?
                    # print('init q load')
                    self.load_network("round_{0}_inter_{1}".format(cnt_round - 1, self.intersection_id))
                    # print('init q_bar load')
                    if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                        if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max((cnt_round - 1) // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"] * self.dic_agent_conf[
                                    "UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                        else:
                            self.load_network_bar("round_{0}_inter_{1}".format(
                                max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0),
                                self.intersection_id))
                    else:
                        self.load_network_bar("round_{0}_inter_{1}".format(
                            max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0), self.intersection_id))

            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        # decay the epsilon
        """
            "EPSILON": 0.8,
            "EPSILON_DECAY": 0.95,
            "MIN_EPSILON": 0.2,
        """
        if os.path.exists(
                os.path.join(
                    dic_path["PATH_TO_MODEL"],
                    "round_-1_inter_{0}.ckpt".format(intersection_id))):
            # the 0-th model is pretrained model
            dic_agent_conf["EPSILON"] = dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f' % (cnt_round, dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = dic_agent_conf["EPSILON"] * pow(dic_agent_conf["EPSILON_DECAY"], cnt_round)
            dic_agent_conf["EPSILON"] = max(decayed_epsilon, dic_agent_conf["MIN_EPSILON"])

    def compute_len_feature(self):
        from functools import reduce
        len_feature = tuple()
        for feature_name in dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if "adjacency" in feature_name:
                continue
            elif "phase" in feature_name:
                len_feature += dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()]
            elif feature_name == "lane_num_vehicle":
                len_feature += (
                dic_traffic_env_conf["DIC_FEATURE_DIM"]["D_" + feature_name.upper()][0] * self.num_lanes,)
        return sum(len_feature)


    def build_q_network(self, MLP_layers=None, Output_layers=None):
        if MLP_layers is None:
            MLP_layers = [32, 32]
        if Output_layers is None:
            Output_layers = []
        CNN_layers = self.CNN_layers
        CNN_heads = [1] * len(CNN_layers)

        start_time = time.time()
        assert len(CNN_layers)==len(CNN_heads)

        return Q_network(num_agents=self.num_agents,
                         len_feature=self.len_feature,
                         MLP_layers=MLP_layers,
                         CNN_layers=CNN_layers,
                         CNN_heads=CNN_heads,
                         num_actions=8)


    def action_att_predict(self, state, total_features=None, total_adjs=None, bar=False):
        bsz = len(state)
        if total_features is None and total_adjs is None:
            total_features, total_adjs = [], []
            for i in range(bsz):
                feature, adj = [], []
                for agent in range(self.num_agents):
                    observation = [self.last_change_step[agent]]
                    for name, value in state[i].items():
                        if "vehicle_num" in name:
                            observation.extend(value[1:])
                    feature.append(observation)
                    adj.append(state[i][agent]['adjacency_matrix'])
                total_features.append(feature)
                total_adjs.append(adj)
            total_features = np.reshape(np.array(total_features), [batch_size, self.num_agents, -1])
            total_adjs = self.adjacency_index2matrix(np.array(total_adjs))
        if bar:
            all_output = self.q_network_bar.predict([total_features, total_adjs])
        else:
            all_output = self.q_network([total_features, total_adjs])
        action, attention = all_output[0], all_output[1]
        
        if len(action) > 1:
            return total_features, total_adjs, action, attention
        
        max_action = np.extend_dims(np.argmax(action, acis=-1), axis=-1)
        random_action = np.reshape(np.random.randint(self.num_actions, size=1 * self.num_agents),
                                   (1, self.num_agents, 1))
        # [batch,agent,2]
        possible_action = np.concatenate([max_action, random_action], axis=-1)
        selection = np.random.choice(
            [0, 1],
            size=bsz * self.num_agents,
            p=[1 - dic_agent_conf["EPSILON"], dic_agent_conf["EPSILON"]])
        act = possible_action.reshape((bsz * self.num_agents, 2))[
            np.arange(bsz * self.num_agents), selection]
        act = np.reshape(act, (bsz, self.num_agents))
        return act, attention

    def act(self, obs):
        act, attention = self.action_att_predict([obs])
        return act[0], attention[0]



