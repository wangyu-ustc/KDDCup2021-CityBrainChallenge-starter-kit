import os
import sys
import torch
import pickle
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler


# from agent.model import BaseModel, FRAPModel
# from agent.configs import *

phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                           [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                           [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                           [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]

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


class BaseModel2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaseModel2, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=(3), padding=(1))
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(50, 64)
        # self.linear1 = nn.Linear(72, 512)
        self.linear1 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32 * 72, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(512, output_dim)

    def forward(self, ob):
        idx = torch.where(ob >= 50)
        ob[idx] = 49

        assert (ob < 50).all()
        assert (ob >= 0).all()
        # ob: (bsz, 72)
        x = self.embedding(ob)  # x: (bsz, 72, 64)

        x = self.relu(x)
        x = self.linear1(x)  # (bsz, 72, 32)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear4(x)
        x = self.linear2(x)
        x = self.linear3(self.relu(x))

        return x


inverse_clockwise_mapping = {
    1: 3,
    2: 4,
    3: 1,
    4: 2,
    5: 8,
    6: 5,
    7: 6,
    8: 7
}

clockwise_mapping = {
    y: x for x, y in inverse_clockwise_mapping.items()
}


def clock_wise_rotate(obs):
    new_obs = torch.zeros_like(obs)
    if len(obs) == 25:
        buffer = 1
    else:
        buffer = 0
    new_obs[:, buffer + 3: buffer + 12] = obs[:, buffer + 0: buffer + 9]
    new_obs[:, buffer + 0: buffer + 3] = obs[:, buffer + 9: buffer + 12]
    new_obs[:, buffer + 15: buffer + 24] = obs[:, buffer + 12: buffer + 21]
    new_obs[:, buffer + 12: buffer + 15] = obs[:, buffer + 21: buffer + 24]
    return new_obs


def validate(data_loader, model):
    loss = 0
    for batch_idx, (last_obs, last_obs_speed, obs, obs_speed, new_obs, new_obs_speed, action, last_action) in enumerate(
            data_loader):
        action = action - 1
        action = action.cuda().long()
        if model_name == 'FRAP':
            pressures = []
            for ob_i, ob in enumerate(obs):
                lane_vehicle_num = ob.numpy()
                pressure = []
                for j in range(8):
                    pressure_j = 0
                    for out_lane in IN_OUT_LANE[FRAP_intersections[j]]:
                        pressure_j += lane_vehicle_num[FRAP_intersections[j] - 1] - lane_vehicle_num[out_lane - 1]

                    pressure.append([pressure_j, Phase_to_FRAP_Phase[last_action[ob_i] - 1][j]])
                pressures.append(pressure)
            pressures = torch.tensor(pressures).cuda()

            action_pred = model(pressures.float())
            loss = F.cross_entropy(action_pred, action)

        else:

            obs, obs_speed = obs[:, 1:].cuda(), obs_speed[:, 1:].cuda()
            last_obs, last_obs_speed = last_obs[:, 1:].cuda(), last_obs_speed[:, 1:].cuda()
            new_obs, new_obs_speed = new_obs[:, 1:].cuda(), new_obs_speed[:, 1:].cuda()

            clean_turn_right(last_obs)
            clean_turn_right(last_obs_speed)
            clean_turn_right(obs)
            clean_turn_right(new_obs)
            clean_turn_right(obs_speed)
            clean_turn_right(new_obs_speed)

            last_obs *= (1 - (last_obs_speed > 5.5).to(torch.int))
            obs *= (1 - (obs_speed > 5.5).to(torch.int))
            diff_obs = obs - last_obs
            new_obs *= (1 - (new_obs_speed > 5.5).to(torch.int))

            last_pressure = cal_pressure(last_obs)
            pressure = cal_pressure(obs)

            action_pred = model(torch.cat([last_obs, obs, diff_obs, last_pressure, pressure], dim=1).float())
            # action_pred = model(torch.cat([last_obs+1, obs+1, diff_obs - torch.min(diff_obs)], dim=1).long())
            loss += F.cross_entropy(action_pred, action).item()
            # obs_1 = clock_wise_rotate(obs)
            # obs_2 = clock_wise_rotate(obs_1)
            # obs_3 = clock_wise_rotate(obs_2)
            # obs_speed_1 = clock_wise_rotate(obs_speed)
            # obs_speed_2 = clock_wise_rotate(obs_speed_1)
            # obs_speed_3 = clock_wise_rotate(obs_speed_2)

            # actions_1 = torch.LongTensor([clockwise_mapping[x.item() + 1] - 1 for x in action.flatten()]).reshape(action.shape).cuda()
            # actions_2 = torch.LongTensor([clockwise_mapping[x.item() + 1] - 1 for x in actions_1.flatten()]).reshape(action.shape).cuda()
            # actions_3 = torch.LongTensor([clockwise_mapping[x.item() + 1] - 1 for x in actions_2.flatten()]).reshape(action.shape).cuda()

            # new_obs_1 = clock_wise_rotate(new_obs).detach()
            # new_obs_2 = clock_wise_rotate(new_obs_1).detach()
            # new_obs_3 = clock_wise_rotate(new_obs_2).detach()

            # loss = cal_loss(obs, obs_speed, model) + cal_loss(obs_1, obs_speed_1, model) + cal_loss(obs_2, obs_speed_2, model)
            # loss = cal_loss(obs, obs_speed, model)

    return loss / len(data_loader)


def clean_turn_right(obs):
    obs[:, 2] = 0
    obs[:, 5] = 0
    obs[:, 8] = 0
    obs[:, 11] = 0


def cal_loss(obs, obs_speed, model):
    if with_speed:
        action_pred = model(torch.cat([obs, obs_speed], dim=1))
    else:
        action_pred = model(obs.float())

    loss = F.cross_entropy(action_pred, action)
    return loss


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

def cal_pressure(obs):
    all_pressures = []
    for idx, ob in enumerate(obs):
        ob = ob.cpu().numpy()
        pressures = []
        for i in range(8):
            in_lanes = phase_lane_map_in[i]
            out_lanes = phase_lane_map_out[i]
            pressure = 0
            for in_lane in in_lanes:
                pressure += ob[in_lane - 1] * 3
            for out_lane in out_lanes:
                pressure -= ob[out_lane - 1]
            pressures.append(pressure)
        all_pressures.append(pressures)
    return torch.tensor(all_pressures).cuda()




if __name__ == '__main__':
    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 1000

    # model_name = 'FRAP'
    # with_speed = False

    with_speed = True
    model_name = 'MLP'

    input_dim = 72 + 8 + 8
    if model_name == 'MLP':
        model = BaseModel(input_dim=72 + 8 + 8, output_dim=8).cuda()
    else:
        model = FRAPModel(relations=relations).cuda()

    round = 131

    try:
        model.load_state_dict(torch.load(f"./supervise_model/model_{round}_{input_dim}_{model_name}_1024.ckpt"))
        print("model loaded")
    except:
        print("model load failed.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data = None

    with open("./data/train_data_4.pkl", "rb") as file:
        tmp_data = pickle.load(file)
        if train_data is None:
            train_data = tmp_data
        else:
            for k, v in train_data.items():
                train_data[k].extend(v)

    # for file in os.listdir("./data"):
    #     if 'train_data_new' in file:
    #         with open("./data/" + file, "rb") as file:
    #             tmp_data = pickle.load(file)
    #             if train_data is None:
    #                 train_data = tmp_data
    #             else:
    #                 for k, v in train_data.items():
    #                     train_data[k].extend(v)

    print("length:", len(train_data['obs']))
    print(train_data.keys())
    print("obs sampled:")

    for ob in train_data['obs']:
        if ob[0] > 2000:
            print(ob)
        if np.sum(ob[1:]) > 0:
            print(ob)
        break

    dataset = data.TensorDataset(torch.tensor(train_data['last_obs']), torch.tensor(train_data['last_obs_speed']),
                                 torch.tensor(train_data['obs']), torch.tensor(train_data['obs_speed']),
                                 torch.tensor(train_data['new_obs']), torch.tensor(train_data['new_obs_speed']),
                                 torch.tensor(train_data['action']), torch.tensor(train_data['last_action']))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = data.DataLoader(dataset, batch_size=2048, sampler=train_sampler)
    val_dataloader = data.DataLoader(dataset, batch_size=2048, sampler=valid_sampler)

    for i in range(100):
        for batch_idx, (
        last_obs, last_obs_speed, obs, obs_speed, new_obs, new_obs_speed, action, last_action) in enumerate(
                train_dataloader):
            action = action - 1
            action = action.cuda().long()
            if model_name == 'FRAP':
                pressures = []
                for ob_i, ob in enumerate(obs):
                    lane_vehicle_num = ob.numpy()
                    pressure = []
                    for j in range(8):
                        pressure_j = 0
                        for out_lane in IN_OUT_LANE[FRAP_intersections[j]]:
                            pressure_j += lane_vehicle_num[FRAP_intersections[j] - 1] - lane_vehicle_num[out_lane - 1]

                        pressure.append([pressure_j, Phase_to_FRAP_Phase[last_action[ob_i] - 1][j]])
                    pressures.append(pressure)
                pressures = torch.tensor(pressures).cuda()

                action_pred = model(pressures.float())
                loss = F.cross_entropy(action_pred, action)

            else:
                obs, obs_speed = obs[:, 1:].cuda(), obs_speed[:, 1:].cuda()
                last_obs, last_obs_speed = last_obs[:, 1:].cuda(), last_obs_speed[:, 1:].cuda()
                new_obs, new_obs_speed = new_obs[:, 1:].cuda(), new_obs_speed[:, 1:].cuda()

                clean_turn_right(last_obs)
                clean_turn_right(last_obs_speed)
                clean_turn_right(obs)
                clean_turn_right(new_obs)
                clean_turn_right(obs_speed)
                clean_turn_right(new_obs_speed)

                # last_obs *= (1 - (last_obs_speed > 5.5).to(torch.int))
                # obs *= (1 - (obs_speed > 5.5).to(torch.int))
                diff_obs = obs - last_obs
                # new_obs *= (1 - (new_obs_speed > 5.5).to(torch.int))

                last_pressure = cal_pressure(last_obs)
                pressure = cal_pressure(obs)

                # action_pred = model(torch.cat([last_obs+1, obs+1, diff_obs - torch.min(diff_obs)], dim=1).long())
                action_pred = model(torch.cat([last_obs, obs, diff_obs, last_pressure, pressure], dim=1).float())
                loss = F.cross_entropy(action_pred, action)
                # obs_1 = clock_wise_rotate(obs)
                # obs_2 = clock_wise_rotate(obs_1)
                # obs_3 = clock_wise_rotate(obs_2)
                # obs_speed_1 = clock_wise_rotate(obs_speed)
                # obs_speed_2 = clock_wise_rotate(obs_speed_1)
                # obs_speed_3 = clock_wise_rotate(obs_speed_2)

                # actions_1 = torch.LongTensor([clockwise_mapping[x.item() + 1] - 1 for x in action.flatten()]).reshape(action.shape).cuda()
                # actions_2 = torch.LongTensor([clockwise_mapping[x.item() + 1] - 1 for x in actions_1.flatten()]).reshape(action.shape).cuda()
                # actions_3 = torch.LongTensor([clockwise_mapping[x.item() + 1] - 1 for x in actions_2.flatten()]).reshape(action.shape).cuda()

                # new_obs_1 = clock_wise_rotate(new_obs).detach()
                # new_obs_2 = clock_wise_rotate(new_obs_1).detach()
                # new_obs_3 = clock_wise_rotate(new_obs_2).detach()

                # loss = cal_loss(obs, obs_speed, model) + cal_loss(obs_1, obs_speed_1, model) + cal_loss(obs_2, obs_speed_2, model)
                # loss = cal_loss(obs, obs_speed, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 500 == 0:
                val_loss = validate(val_dataloader, model)
                print(f"Epoch {i}, batch_idx {batch_idx}, loss = {loss.item()}, val_loss = {val_loss}")
                sys.stdout.flush()

            torch.save(model.state_dict(), f"./supervise_model/model_{round + i}_{input_dim}_{model_name}_1024.ckpt")

