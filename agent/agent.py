import matplotlib
import pickle
from types import CodeType
# from typing import AwaitableGenerator
import gym
# from matplotlib.pyplot import step
# from numpy.lib.function_base import angle
import CBEngine
import json
import traceback
import argparse
import logging
import os
import sys
import time
import re
import numpy as np
from pathlib import Path
from pdb import set_trace as bp


# how to import or load local files
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
import numpy as np

with open(path + "/gym_cfg.py", "r") as f:
    pass

simulator_cfg_file = "./simulator.cfg"
gym_cfg_instance = gym_cfg.gym_cfg()
last_obs = {}
last_info = {}
road_wait = {}

FEATURES = ['lane_speed', 'lane_vehicle_num']

INF = -float('inf')
PHASE_NO_NEED = set([3, 6, 9, 12])  # 与信号灯无关的三条lane
# 信号灯的phase与lane通行的关系
PHASE_LANE = {
    1: [1, 7],
    2: [2, 8],
    3: [4, 10],
    4: [5, 11],
    5: [2, 1],
    6: [5, 4],
    7: [7, 8],
    8: [10, 11]
}

PHASE_IN_ROAD = {
    1: [0, 2],
    2: [0, 2],
    3: [1, 3],
    4: [1, 3],
    5: [0],
    6: [1],
    7: [2],
    8: [3]
}

PHASE_OUT_ROAD = {
    1: [1, 3],
    2: [0, 2],
    3: [0, 2],
    4: [1, 3],
    5: [1, 2],
    6: [3, 2],
    7: [3, 0],
    8: [0, 1]
}

IN_OUT_LANE = {
    1: [16, 17, 18],
    2: [19, 20, 21],
    4: [19, 20, 21],
    5: [22, 23, 24],
    7: [22, 23, 24],
    8: [13, 14, 15],
    10: [13, 14, 15],
    11: [16, 17, 18]
}

LANE_ROAD = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
    10: 3,
    11: 3,
    12: 3
}


class TestAgent():
    def __init__(self, params=None):

        if params is not None:
            self.params = params
        else:
            # self.params = {
            #     "speed": 5.5,
            #     "wait_up": 0.6,
            #     "wait_down": 0.3,
            #     "num_threshold": 10,
            #     "max_wait": 15,
            # }

            self.params = {
                "speed": 5,
                "wait_up": 0.6,
                "wait_down": 0.4,
                "num_threshold": 15,
                "max_wait": 15,
                "length": 5,
                "mean_tff": 800,
                "out_weight": 0.5,
                "speed_ratio": 1,
                "in_weight": 1,
                "rounds": 1,
                "mp_step": 2500,
            }

        print(self.params)

        self.green_sec = 17
        self.max_phase = 8
        self.last_change_step = {}  # 存储上一步的时间 key是agent_id
        self.agent_list = []
        self.intersections = {}
        self.roads = {}
        self.agents = {}

        self.last_phase = {}  # 当前的phase key是agent_id
        self.last_max_waiting = {}
        self.last_max_pressure = {}
        self.waiting_pressure = {}
        self.pressure = {}
        self.phase = {}
        self.waiting_times = {}
        self.pressure_times = {}

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.last_change_step = dict.fromkeys(self.agent_list, 0)
        self.last_phase = dict.fromkeys(self.agent_list, 1)
        self.last_max_waiting = dict.fromkeys(self.agent_list, 0)
        self.last_max_pressure = dict.fromkeys(self.agent_list, 0)

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
    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

        # print("agents_num ",len(self.agents))
        # print("intersection_num ",len(self.intersections))
        # print("------------------------------------------------------------------------------------------------------------")
        ################################
        # agent_list = list(self.agents.keys())

        # env_roads_length = {}
        # for road_name in env.roads.keys():
        #     env_roads_length[road_name] = env.roads[road_name]['length']

        agent_line_length = {}
        self.agent_line_capacity = {}
        self.roadlength = {}
        self.road_speed_limit = {}
        for agent_name in self.agent_list:
            agent_lane = []
            agent_roads = self.agents[agent_name]  # inroad和outroad,共8个，inroad是0123
            data_length = []
            data_speed_limit = []
            for agent_road in agent_roads:
                if agent_road == -1:
                    agent_lane.extend([-1] * 3)
                    data_length.append(0)
                    data_speed_limit.extend([0] * 3)
                else:
                    agent_lane.extend([self.roads[agent_road]['length']] * 3)  # 每个lane的限速
                    data_length.append(self.roads[agent_road]['length'])
                    data_speed_limit.extend([self.roads[agent_road]['speed_limit']] * 3)
            self.roadlength[agent_name] = data_length
            self.road_speed_limit[agent_name] = data_speed_limit

            agent_line_length[agent_name] = agent_lane
            temp = np.array(agent_lane)
            self.agent_line_capacity[agent_name] = (temp // self.params['length']).tolist()

    def max_pressure_phase(self, obs, key, lane_car, car_speed, infos):
        # aim to calculate max pressure for every single phase
        # here,obs means observation for this agent
        phase_pressure = {}
        for phase in PHASE_LANE.keys():
            inroad_lanes = PHASE_LANE[phase]
            for in_lane in inroad_lanes:
                real_inroad_num = obs[str(key) + '_lane_vehicle_num'][in_lane]
                step = obs[str(key) + '_lane_vehicle_num'][0]
                if real_inroad_num == -1:
                    continue

                if step < self.params['mp_step']:
                    inroad_num = real_inroad_num
                else:
                    inlane_id = self.agents[key][(in_lane - 1) // 3] * 100 + ((in_lane - 1) % 3)
                    if inlane_id not in self.agents_lane_smoothed_vehicle_num[key]:
                        inroad_num = 0
                        assert real_inroad_num == 0
                    else:
                        inroad_num = self.agents_lane_smoothed_vehicle_num[key][inlane_id]

                # 此处加入下游车道承载讷能力 capacity
                out_road_lanes = IN_OUT_LANE[in_lane]
                rest_capacity = []
                for out_lane in out_road_lanes:
                    rest_capacity_temp = self.agent_line_capacity[key][out_lane - 1]
                    full_capacity = rest_capacity_temp

                    if self.agents[key][(out_lane - 1)//3] == -1:
                        continue

                    speed_limit = self.roads[self.agents[key][(out_lane - 1)//3]]['speed_limit']
                    out_lane_id = self.agents[key][(out_lane - 1)//3] * 100 + ((out_lane - 1) % 3)
                    if out_lane_id not in lane_car:
                        lane_car[out_lane_id] = {}
                    for car in lane_car[out_lane_id]:
                        speed = car_speed[car]
                        car_num = max(1 - speed / speed_limit * self.params['speed_ratio'], 0)
                        # car_num /= (infos[car]["t_ff"][0] * speed_limit) * self.params['mean_tff']
                        rest_capacity_temp -= car_num
                    rest_capacity.append(rest_capacity_temp)
                total_rest_capacity = 0
                total_outroad_num = full_capacity * len(rest_capacity) - np.sum(rest_capacity)
                for i in range(len(rest_capacity)):
                    total_rest_capacity += (full_capacity - rest_capacity[i]) / total_outroad_num * rest_capacity[i]

                rest_capacity = total_rest_capacity


                if phase not in phase_pressure:
                    phase_pressure[phase] = min(inroad_num, rest_capacity)
                elif phase in phase_pressure:
                    phase_pressure[phase] += min(inroad_num, rest_capacity)

        # 选取最大pressure的phase
        max_phase = max(phase_pressure, key=phase_pressure.get)
        max_pressure = phase_pressure[max_phase]

        return max_phase, max_pressure, phase_pressure

    def phase_time(self, phase, key):
        if (phase == 1 or phase == 2):
            time = self.cal_green_time(float(self.roadlength[key][0] + self.roadlength[key][2]) / 2)
        elif (phase == 3 or phase == 4):
            time = self.cal_green_time(float(self.roadlength[key][1] + self.roadlength[key][3]) / 2)
        elif (phase == 5):
            time = self.cal_green_time(self.roadlength[key][0])
        elif (phase == 6):
            time = self.cal_green_time(self.roadlength[key][1])
        elif (phase == 7):
            time = self.cal_green_time(self.roadlength[key][2])
        elif (phase == 8):
            time = self.cal_green_time(self.roadlength[key][3])
        return time

    def cal_green_time(self, length):

        green_time = 0
        green_time = 25 / 1000 * np.mean(length) + 15

        return green_time

    def cal_waiting(self, lane_car, car_speed, inroads, outroads, infos, agent_id, step):

        inroad_lane = {}  # key是inroad_id
        re_inroad_lane = {}  # key是1-12 inroad_id
        reverse_lane_dict = {}
        for idx, inroad_id in enumerate(inroads):  # 获取每一个inroad的3条lane_id
            if inroad_id == -1:
                continue
            lane_dict = self.roads[inroad_id]['lanes']
            for k, v in lane_dict.items():
                if inroad_id not in inroad_lane:
                    inroad_lane[inroad_id] = [k]
                else:
                    inroad_lane[inroad_id].append(k)

                pos = 0
                for i in range(len(v)):
                    if v[i] == 1:
                        pos = i
                        break
                pos += 1
                re_inroad_lane[idx * 3 + pos] = k
                reverse_lane_dict[k] = idx * 3 + pos


        # print("-" * 40)
        # print("re_inroad_lane:", re_inroad_lane)
        # print("-" * 40)

        outroad_lane = {}  # key是inroad_id
        re_outroad_lane = {}  # key是1-12 inroad_id
        for idx, outroad_id in enumerate(outroads):  # 获取每一个inroad的3条lane_id
            if outroad_id == -1:
                continue
            lane_dict = self.roads[outroad_id]['lanes']
            # 这里lane_dict是lane的id(由road得到)
            for k, v in lane_dict.items():
                if outroad_id not in outroad_lane:
                    outroad_lane[outroad_id] = [k]
                else:
                    outroad_lane[outroad_id].append(k)

                pos = 0
                for i in range(len(v)):
                    if v[i] == 1:
                        pos = i
                        break
                pos += 1
                re_outroad_lane[idx * 3 + pos] = k
                reverse_lane_dict[k] = idx * 3 + pos

        lane_waiting_num = {}
        # 计算堵车数量
        for idx, inroad_id in enumerate(inroads):
            if inroad_id == -1:
                continue

            speed_limit = self.roads[inroad_id]['speed_limit']
            road_length = self.roads[inroad_id]['length']
            inroad_capacity = road_length / self.params['length']
            lanes = inroad_lane[inroad_id]
            for lane in lanes:
                if reverse_lane_dict[lane] in PHASE_NO_NEED:
                    continue
                if lane not in lane_car:
                    out_rest_capacity = self.agent_line_capacity[agent_id][IN_OUT_LANE[reverse_lane_dict[lane]][0] - 1]
                    self.agents_out_rest_capacity[agent_id][lane] = out_rest_capacity
                    continue
                all_cars = lane_car[lane]
                for car in all_cars:
                    speed = car_speed[car]

                    car_num = max(1 - speed / speed_limit * self.params['speed_ratio'], 0)
                    # car_num /= road_length
                    car_num = car_num / (infos[car]["t_ff"][0] * speed_limit) * self.params['mean_tff']

                    if lane not in lane_waiting_num:
                        lane_waiting_num[lane] = car_num
                    else:
                        lane_waiting_num[lane] += car_num

                self.agents_lane_smoothed_vehicle_num[agent_id][lane] = lane_waiting_num[lane]

                out_rest_capacity = self.agent_line_capacity[agent_id][IN_OUT_LANE[reverse_lane_dict[lane]][0] - 1]
                outroad_capacity = out_rest_capacity
                out_speed_limit = self.road_speed_limit[agent_id][IN_OUT_LANE[reverse_lane_dict[lane]][0] - 1]
                # out_rest_capacity = [out_rest_capacity] * 3
                total_lane_num = 0
                out_lane_road_num_list = []
                for i, out_lane in enumerate(IN_OUT_LANE[reverse_lane_dict[lane]]):
                    out_lane_road_num = 0
                    if out_lane - 12 not in re_outroad_lane:
                        continue
                    out_origin = re_outroad_lane[out_lane - 12]
                    if out_origin not in lane_car:
                        continue
                    all_out_cars = lane_car[out_origin]

                    for car in all_out_cars:
                        speed = car_speed[car]

                        # car_num = max((out_speed_limit - speed) / out_speed_limit, 0)
                        # car_num /= road_length
                        # car_num = car_num / (infos[car]["t_ff"][0] * out_speed_limit) * 610
                        # out_rest_capacity -= car_num

                        if speed < self.params['speed']:
                        # if speed < speed_limit * 0.3:
                            # out_rest_capacity[i] -= 1
                            # out_rest_capacity -= 1
                            out_lane_road_num += 1
                        total_lane_num += 1
                    out_lane_road_num_list.append(out_lane_road_num)
                for i in range(len(out_lane_road_num_list)):
                    out_rest_capacity -= (out_lane_road_num_list[i] / (total_lane_num + 1e-3)) * (out_lane_road_num_list[i])

                self.agents_out_rest_capacity[agent_id][lane] = out_rest_capacity
                # lane_waiting_num[lane] = min(lane_waiting_num[lane], out_rest_capacity // 3)
                if step > 20000:
                    lane_waiting_num[lane] = min(lane_waiting_num[lane] / inroad_capacity, out_rest_capacity / outroad_capacity)
                else:
                    lane_waiting_num[lane] = min(lane_waiting_num[lane], out_rest_capacity) * 3
                # lane_waiting_num[lane] = min(lane_waiting_num[lane], out_rest_capacity) * 3




        return re_inroad_lane, lane_waiting_num

    def waiting_phase_selection(self, agent_id, lane_car, car_speed, infos):

        re_inroad_lane = self.agents_lane_waiting_num[agent_id]['re_inroad_lane']
        lane_waiting_num = self.agents_lane_waiting_num[agent_id]['lane_waiting_num']

        # 计算哪个phase对应的lane排队数最多
        phase_waiting_nums = dict.fromkeys(PHASE_LANE, 0)
        for phase, lanes in PHASE_LANE.items():
            for lane in lanes:
                if lane not in re_inroad_lane:
                    continue
                # print(re_inroad_lane)
                ori_lane = re_inroad_lane[lane]
                # print(lane,ori_lane)
                if ori_lane in lane_waiting_num:
                    phase_waiting_nums[phase] += lane_waiting_num[ori_lane]
        # print(phase_waiting_nums)
        max_phase = max(phase_waiting_nums, key=phase_waiting_nums.get)
        # print(max_phase)
        max_waiting = phase_waiting_nums[max_phase]

        return max_phase, max_waiting, phase_waiting_nums, lane_waiting_num

    def second_waiting_phase_selection(self, agent_id, lane_car, car_speed, phase_list, info):
        re_inroad_lane = self.agents_lane_waiting_num[agent_id]['re_inroad_lane']
        lane_waiting_num = self.agents_lane_waiting_num[agent_id]['lane_waiting_num']

        # 计算哪个phase对应的lane排队数最多
        phase_waiting_nums = dict.fromkeys(PHASE_LANE, 0)
        for phase, lanes in PHASE_LANE.items():
            for lane in lanes:
                if lane not in re_inroad_lane:
                    continue
                # print(re_inroad_lane)
                ori_lane = re_inroad_lane[lane]
                # print(lane,ori_lane)
                if ori_lane in lane_waiting_num:
                    phase_waiting_nums[phase] += lane_waiting_num[ori_lane]
        # print(phase_waiting_nums)
        max_phase = max(phase_waiting_nums, key=phase_waiting_nums.get)
        # print(max_phase)
        phase_waiting_nums.pop(phase_list[0])
        phase_waiting_nums.pop(phase_list[1])
        print(phase_waiting_nums)
        second_max_phase = max(phase_waiting_nums, key=phase_waiting_nums.get)

        return second_max_phase

    def visualize(self, ob, action):
        vis = np.zeros([8, 8])
        vis[0, 1] = ob[3]
        vis[0, 2] = ob[2]
        vis[0, 3] = ob[1]
        vis[0, 4] = ob[13]
        vis[0, 5] = ob[14]
        vis[0, 6] = ob[15]
        vis[1, 0] = ob[24]
        vis[2, 0] = ob[23]
        vis[3, 0] = ob[22]
        vis[4, 0] = ob[10]
        vis[5, 0] = ob[11]
        vis[6, 0] = ob[12]
        vis[1, 7] = ob[6]
        vis[2, 7] = ob[5]
        vis[3, 7] = ob[4]
        vis[4, 7] = ob[16]
        vis[5, 7] = ob[17]
        vis[6, 7] = ob[18]
        vis[7, 1] = ob[21]
        vis[7, 2] = ob[20]
        vis[7, 3] = ob[19]
        vis[7, 4] = ob[7]
        vis[7, 5] = ob[8]
        vis[7, 6] = ob[9]
        vis[3,3] = vis[3,4] = vis[4,3] = vis[4,4] = action
        print(vis)


    def get_actions(self, new_obs, observations, actions, lane_car, car_speed, info):
        for agent_id in self.agent_list:
            # self.visualize(new_obs[agent_id]['lane_vehicle_num'], 0)
            agent_obs = new_obs[agent_id]
            # select the cur_second
            for k, v in agent_obs.items():
                cur_second = v[0]
                break

            if cur_second == 0:
                max_pressure_phase, max_pressure, pressure_dict = self.max_pressure_phase(
                    obs=observations,
                    key=agent_id,
                    lane_car=lane_car,
                    car_speed=car_speed,
                    infos=info
                )
                self.last_phase[agent_id] = max_pressure_phase
                self.last_max_pressure[agent_id] = max_pressure

                actions[agent_id] = max_pressure_phase
                continue

            last_phase = self.last_phase[agent_id]
            max_waiting_phase, max_waiting, waiting_dict, lane_waiting = self.waiting_phase_selection(
                agent_id=agent_id,
                lane_car=lane_car,
                car_speed=car_speed,
                infos=info
            )
            last_max_waiting = self.last_max_waiting[agent_id]
            last_now_waiting = waiting_dict[last_phase]

            max_pressure_phase, max_pressure, pressure_dict = self.max_pressure_phase(
                obs=observations,
                key=agent_id,
                lane_car=lane_car,
                car_speed=car_speed,
                infos=info
            )
            last_max_pressure = self.last_max_pressure[agent_id]
            last_now_pressure = pressure_dict[last_phase]

            step_diff = cur_second - self.last_change_step[agent_id]
            green_time = self.phase_time(last_phase, agent_id)

            # select phase
            select_phase = None
            phase_no_effect1 = None
            phase_no_effect2 = None
            phase_list = list(range(1, 9))
            # threshold = 5
            threshold = self.params['num_threshold']
            if max_waiting != 0:

                if cur_second not in self.waiting_times:
                    self.waiting_times[cur_second] = 1
                elif cur_second in self.waiting_times:
                    self.waiting_times[cur_second] += 1

                if (max_waiting_phase == last_phase):  # phase doesn't change

                    if last_max_waiting == max_waiting:  # 上游堵到一定程度时，如果phase维持不变，上游堵车依然不能缓解时，不如避开相关的phase，去尝试其他的phase

                        '''
                        self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0]] 
                        intersection是路口，结构为{agent_id:{'xxx':abc, 'lanes':[inlane_id1, inlane_id2, ..., inlane_id12, outlane_id1, ..., outlane_id12]}}  
                        PHASE_LANE[last_phase][0]是上一个phase对应的第一条lane的序号，例如对于phase1来说，即lane1或者lane7（不是lane_id，不是378000这种）
                        '''

                        if self.intersections[agent_id]['lanes'][
                            PHASE_LANE[last_phase][0] - 1] in lane_waiting.keys() and \
                                self.intersections[agent_id]['lanes'][
                                    PHASE_LANE[last_phase][1] - 1] in lane_waiting.keys():

                            if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][0]]] / (lane_waiting[
                                self.intersections[agent_id]['lanes'][
                                    PHASE_LANE[last_phase][0] - 1]] + 0.01)) < threshold and float(
                                    self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][1]]] / lane_waiting[
                                        self.intersections[agent_id]['lanes'][
                                            PHASE_LANE[last_phase][1] - 1]]) < threshold:

                                if lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0] - 1]] > \
                                        lane_waiting[self.intersections[agent_id]['lanes'][
                                            PHASE_LANE[last_phase][1] - 1]]:  # 看该phase中哪条路堵的厉害
                                    phase_no_effect1 = last_phase
                                    phase_no_effect2 = [k for k, v in PHASE_LANE.items() if
                                                        PHASE_LANE[last_phase][0] in v]

                                    select_phase = self.second_waiting_phase_selection(agent_id=agent_id,
                                                                                       lane_car=lane_car,
                                                                                       car_speed=car_speed,
                                                                                       phase_list=phase_no_effect2, info=info)
                                    # print(select_phase)
                                    actions[agent_id] = select_phase
                                else:
                                    phase_no_effect1 = last_phase
                                    phase_no_effect2 = [k for k, v in PHASE_LANE.items() if
                                                        PHASE_LANE[last_phase][1] in v]
                                    # 去掉最堵的路对应的两个phase，在剩下6个中做选择

                                    # 随机
                                    select_phase = self.second_waiting_phase_selection(agent_id=agent_id,
                                                                                       lane_car=lane_car,
                                                                                       car_speed=car_speed,
                                                                                       phase_list=phase_no_effect2, info=info)
                                    # print(select_phase)
                                    actions[agent_id] = select_phase

                        elif self.intersections[agent_id]['lanes'][
                            PHASE_LANE[last_phase][0] - 1] in lane_waiting.keys() and \
                                self.intersections[agent_id]['lanes'][
                                    PHASE_LANE[last_phase][1] - 1] not in lane_waiting.keys():
                            # print('\n0\n')
                            # print('\n', lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0]]])
                            if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][0]]] / lane_waiting[
                                self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0] - 1]]) < threshold:
                                phase_no_effect1 = last_phase
                                phase_no_effect2 = [k for k, v in PHASE_LANE.items() if PHASE_LANE[last_phase][0] in v]

                                select_phase = self.second_waiting_phase_selection(agent_id=agent_id, lane_car=lane_car,
                                                                                   car_speed=car_speed,
                                                                                   phase_list=phase_no_effect2, info=info)
                                # print(select_phase)
                                actions[agent_id] = select_phase
                        elif self.intersections[agent_id]['lanes'][
                            PHASE_LANE[last_phase][0] - 1] not in lane_waiting.keys() and \
                                self.intersections[agent_id]['lanes'][
                                    PHASE_LANE[last_phase][1] - 1] in lane_waiting.keys():
                            # print('\n1\n')
                            # print('\n', lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][1]]])
                            if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][1]]] / (lane_waiting[
                                self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][1] - 1]] + 0.01)) < threshold:
                                phase_no_effect1 = last_phase
                                phase_no_effect2 = [k for k, v in PHASE_LANE.items() if PHASE_LANE[last_phase][1] in v]

                                select_phase = self.second_waiting_phase_selection(agent_id=agent_id, lane_car=lane_car,
                                                                                   car_speed=car_speed,
                                                                                   phase_list=phase_no_effect2, info=info)
                                print(select_phase)
                                actions[agent_id] = select_phase
                        else:  # phase doesn't change
                            select_phase = last_phase
                            actions[agent_id] = last_phase

                    else:  # phase doesn't change
                        select_phase = last_phase
                        actions[agent_id] = last_phase

                    # select_phase = last_phase
                    # actions[agent_id] = last_phase
                else:  # phase perhaps changes
                    wait_up = self.params['wait_up']
                    wait_down = self.params['wait_down']
                    max_wait = self.params['max_wait']

                    if (last_now_waiting == 0):
                        select_phase = max_waiting_phase
                    # elif (max_waiting <= 5):
                    #     if (pressure_dict[max_waiting_phase] <= last_now_pressure):
                    #         select_phase = last_phase
                    #     else:
                    #         select_phase = max_waiting_phase
                    # elif (last_now_waiting >= 1 and max_waiting <= 10):
                    #     select_phase = last_phase

                    # elif (last_now_waiting / max_waiting < 0.60 and last_now_waiting / max_waiting > 0.30 and max_waiting <= 8 ):
                    #     select_phase = last_phase
                    # elif (last_now_waiting / max_waiting >= 0.60):
                    #     select_phase = last_phase
                    # else:
                    #     select_phase = max_waiting_phase

                    elif (
                            last_now_waiting / max_waiting < wait_up and last_now_waiting / max_waiting > wait_down and max_waiting <= max_wait):
                        select_phase = last_phase
                    elif (last_now_waiting / max_waiting >= wait_up):
                        select_phase = last_phase
                    else:
                        select_phase = max_waiting_phase

                    self.last_phase[agent_id] = select_phase
                    actions[agent_id] = select_phase
                    # if (cur_second >= 1800):
                    #     self.last_phase[agent_id] = select_phase
                    #     actions[agent_id] = select_phase
                    # else:
                    #     if (select_phase == max_waiting_phase):
                    #         if (step_diff >= green_time):
                    #             self.last_change_step[agent_id] = select_phase
                    #             self.last_phase[agent_id] = select_phase
                    #         actions[agent_id] = self.last_phase[agent_id]
                    #     else:
                    #         self.last_phase[agent_id] = select_phase
                    #         actions[agent_id] = select_phase
            else:  # 如果排队数为0 就采用lane_vehicle_num

                if cur_second not in self.pressure_times:
                    self.pressure_times[cur_second] = 1
                elif cur_second in self.pressure_times:
                    self.pressure_times[cur_second] += 1

                if (max_pressure == 0):
                    select_phase = last_phase
                    actions[agent_id] = select_phase
                else:
                    if (max_pressure_phase == last_phase):
                        select_phase = last_phase
                        actions[agent_id] = last_phase
                    else:
                        if (last_now_pressure == 0):
                            select_phase = max_pressure_phase
                        elif (
                                last_now_pressure / max_pressure <= 0.3 and last_now_pressure / max_pressure >= 0.1 and max_pressure <= 10):
                            select_phase = last_phase
                        elif (last_now_pressure / max_pressure >= 0.7):
                            select_phase = last_phase
                        else:
                            select_phase = max_pressure_phase

                        self.last_phase[agent_id] = select_phase
                        actions[agent_id] = select_phase

                        # if (cur_second >= 1800):
                        #     self.last_phase[agent_id] = select_phase
                        #     actions[agent_id] = select_phase
                        # else:
                        #     if (select_phase == max_pressure_phase):
                        #         if (step_diff >= green_time):
                        #             self.last_change_step[agent_id] = select_phase
                        #             self.last_phase[agent_id] = select_phase
                        #         actions[agent_id] = self.last_phase[agent_id]
                        #     else:
                        #         self.last_phase[agent_id] = select_phase
                        #         actions[agent_id] = select_phase

            # 检查是否需要进行select phase
            # if (step_diff >= green_time):
            #     if agent_id not in actions or actions[agent_id] == last_phase:
            #         pass
            #     else:
            #         self.last_change_step[agent_id] = step
            # else:
            #     actions[agent_id] = last_phase
            #     self.last_phase[agent_id] = last_phase

            self.last_max_waiting[agent_id] = max_waiting
            self.last_max_pressure[agent_id] = max_pressure


    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos

        # observations is returned 'observation' of env.step()
        # info is returned 'info' of env.step()
        observations = obs['observations']
        info = obs['info']
        actions = {}


        # preprocess observations
        new_obs = {}  # 嵌套字典 第一级key为agent_id 第二级key为feature
        for key, val in observations.items():
            agent_id = int(key.split('_')[0])  # 获取agent_id
            feature = key[key.find('_') + 1:]  # feature为lane_num或lane_speed
            if (agent_id not in new_obs.keys()):
                new_obs[agent_id] = {}
            new_obs[agent_id][feature] = val
            step = val[0]

        lane_car = {}  # key为lane_id val为car_id是一个list
        car_speed = {}  # key为car_id val为速度
        for car_id, car_info in info.items():
            cur_lane = car_info['drivable'][0]
            cur_speed = car_info['speed'][0]
            if cur_lane not in lane_car:
                lane_car[cur_lane] = [car_id]
            elif cur_lane in lane_car:
                lane_car[cur_lane].append(car_id)
            car_speed[car_id] = cur_speed


        # if step < 200:
        #     self.params['mean_tff'] = 1000
        #     # elif step>=100 and step<200:
        #     #     self.params[]
        # elif step >= 200:
        #     self.params['mean_tff'] = 500


        # if step < 1000:
        #     self.params['rounds'] = 0
        # else:
        #     self.params['rounds'] = 1


        self.agents_lane_waiting_num = {}
        self.agents_lane_smoothed_vehicle_num = {}
        self.agents_out_rest_capacity = {}
        for agent_id in self.agent_list:
            agent_roads = self.agents[agent_id]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
            inroads_id = []
            for road_id in agent_roads[0:4]:
                if road_id != -1:
                    inroads_id.append(road_id)
                else:
                    inroads_id.append(-1)

            outroads_id = []
            for road_id in agent_roads[4:8]:
                if road_id != -1:
                    outroads_id.append(road_id)
                else:
                    outroads_id.append(-1)

            self.agents_lane_smoothed_vehicle_num[agent_id] = {}
            self.agents_out_rest_capacity[agent_id] = {}

            re_inroad_lane, lane_waiting_num = self.cal_waiting(
                lane_car=lane_car,
                car_speed=car_speed,
                inroads=inroads_id,
                outroads=outroads_id,
                infos=info,
                agent_id=agent_id,
                step=step,
            )  # waiting_num的key是lane_id

            self.agents_lane_waiting_num[agent_id] = {}
            self.agents_lane_waiting_num[agent_id]['re_inroad_lane'] = re_inroad_lane
            self.agents_lane_waiting_num[agent_id]['lane_waiting_num'] = lane_waiting_num

        # get actions
        self.get_actions(new_obs, observations, actions, lane_car, car_speed, info)

        for _ in range(self.params['rounds']):

            self.agents_adj = {}
            for agent_id in self.agent_list:
                agent_adj = []
                for in_road_id in self.agents[agent_id][:4]:
                    if in_road_id == -1:
                        agent_adj.append(-1)
                    else:
                        agent_adj.append(self.roads[in_road_id]['start_inter'])
                self.agents_adj[agent_id] = agent_adj

            # refine actions
            for agent_id in self.agent_list:

                agent_adj = self.agents_adj[agent_id]

                re_inroad_lane = self.agents_lane_waiting_num[agent_id]['re_inroad_lane']
                lane_waiting_num = self.agents_lane_waiting_num[agent_id]['lane_waiting_num']

                for i in range(1, 13):
                    if i not in re_inroad_lane: continue
                    lane = re_inroad_lane[i]

                    if lane not in lane_waiting_num:
                        continue
                        # 该车道无车辆排队, 其实也可能需要上游车道传入参数，暂时先不考虑

                    out_agent_capacity = 0

                    # 获得出路到达的agent的id
                    out_road_idx = (IN_OUT_LANE[i][0] - 1 - 12) // 3
                    out_agent_id = agent_adj[out_road_idx]

                    # 获得出路的id
                    out_road_id = self.agents[agent_id][4:][out_road_idx]

                    # 检查该out_agent_id是否在agent_list里面.
                    if out_road_id == -1:
                        # TODO: 说明这辆车停在这里是为了直行而非左转
                        continue
                    out_rest_capacity_each_lane = self.roads[out_road_id]['length'] / self.params['length']
                    if out_agent_id not in self.agent_list:
                        # 认为它的capacity是无穷
                        out_rest_capacity = [10000] * 3
                    else:
                        out_rest_capacity = []
                        for j in range(2):
                            rest_capacity = out_rest_capacity_each_lane

                            out_agent_action = self.last_phase[out_agent_id] if out_agent_id not in actions else actions[out_agent_id]

                            # 先将排队的车辆数减掉, TODO: 这里可以assert一些东西来确保正确性, 也可以尝试减掉smoothed的车辆数
                            if out_road_id * 100 + j in lane_car:
                                rest_capacity -= len(lane_car[out_road_id * 100 + j])

                            # 检查该出路在out_agent那里是入路的第几个, 并检查是否在action的phase之内
                            out_in_road_idx = self.agents[out_agent_id][:4].index(out_road_id)
                            if out_in_road_idx in PHASE_IN_ROAD[out_agent_action]:
                                # TODO: 把这一个车道的下游的capacity加到现在的capacity上来
                                # TODO: 这里并未考虑到下游的capacity会被两个路口共享，可能是小概率事件，暂不考虑

                                # 当前车道在下游agent的下游三车道的Capacity
                                # print(self.agents_out_rest_capacity[out_agent_id])
                                rest_capacity += self.agents_out_rest_capacity[out_agent_id][out_road_id * 100 + j]

                            out_rest_capacity.append(rest_capacity)
                        out_rest_capacity.append(10000)

                    # 至此, 我们获得了三条出路的所有 out_rest_capacity
                    # 接下来, 我们需要获得这条lane中的车的信息, 以及即将汇入这个车道的车的信息

                    full_pressure = 0
                    ending_car = 0

                    # 先对于当前车道中的车的信息进行处理
                    if out_agent_id not in self.agent_list:
                        full_pressure += len(lane_car[lane])
                    else:
                        out_agent_roads = self.agents[out_agent_id]
                        for car in lane_car[lane]:
                            # TODO: 计算多少辆车可以在10s的间隙内通过到下一个路口
                            # TODO: 现在就假设所有车都可以通过该路口
                            if len(info[car]['route']) < 3:
                                ending_car += 1
                                continue
                            next_road = info[car]['route'][2]
                            # print("info[car]:", info[car]['route'])
                            # print("next_road:", next_road)
                            # print("out_road_id:", out_road_id)
                            # print("out_agent_roads:", out_agent_roads)
                            # print("self.agents:", self.agents[agent_id])
                            # bp()

                            # if next_road not in out_agent_roads: # TODO: 为什么报错?
                            if next_road not in out_agent_roads[4:]:
                                # 左转车道上出现了直行车
                                continue

                            diff = out_agent_roads[4:].index(next_road) - out_agent_roads[:4].index(out_road_id)
                            if diff == 1:
                                if out_rest_capacity[0] > 0:
                                    full_pressure += 1
                                    out_rest_capacity[0] -= 1
                            if diff == 2 or diff == -2:
                                if out_rest_capacity[1] > 0:
                                    full_pressure += 1
                                    out_rest_capacity[1] -= 1
                            if diff == -1:
                                full_pressure += 1

                    # 首先拿到当前lane的出发agent_id, 便于拿到汇入这个车道的所有车的信息
                    in_road_id = lane // 100
                    in_agent_id = self.roads[in_road_id]['start_inter']
                    if in_agent_id not in self.agent_list:
                        pass
                        # # TODO: 将上一个路口中可能过来的车全部搜出来, 作为inroad_num
                        # in_road_idx_for_in_agent = self.intersections[in_agent_id]['start_roads'].index(in_road_id)
                        #
                        # for i in range(1, 4):
                        #     # print("end_roads:", self.intersections[in_agent_id]['end_roads'])
                        #     # print("lanes:", self.intersections[in_agent_id]['lanes'])
                        #     if len(self.intersections[in_agent_id]['end_roads']) < 4: continue
                        #
                        #     cur_road_id = self.intersections[in_agent_id]['end_roads'][(in_road_idx_for_in_agent + i) % 4]
                        #     # i == 1: for turn right road
                        #     # i == 2: for go straight road
                        #     # i == 3: for turn left road
                        #
                        #     if out_agent_id not in self.agent_list:
                        #         full_pressure += len(lane_car[cur_road_id*100 + (3-i)])
                        #         continue
                        #
                        #     out_agent_roads = self.agents[out_agent_id]
                        #
                        #     for car in lane_car[cur_road_id*100 + (3 - i)]:
                        #         # TODO: 计算多少辆车可以在10s的间隙内通过到这里.
                        #         # TODO: 暂时先就假设全部都可以通过来
                        #
                        #         if len(info[car]['route']) < 4:
                        #             # 说明这辆车即将到达终点, 没有下一个可言, 那么exception += 1
                        #             ending_car += 1
                        #             continue
                        #         next_road = info[car]['route'][3]
                        #
                        #         # 观察next_road 和 in_road_id的关系
                        #         diff = out_agent_roads.index(next_road) - out_agent_roads.index(out_road_id)
                        #         # diff == -1: turn right
                        #         # diff == 1: turn left
                        #         # diff == 2 or -2: go straight
                        #         if diff == 1:
                        #             if out_rest_capacity[0] > 0:
                        #                 full_pressure += 1
                        #                 out_rest_capacity[0] -= 1
                        #
                        #         if diff == 2 or diff == -2:
                        #             if out_rest_capacity[1] > 0:
                        #                 full_pressure += 1
                        #                 out_rest_capacity[1] -= 1
                        #
                        #         if diff == -1:
                        #             full_pressure += 1

                    else:
                        # TODO: 检查上一个路口的action,观察有没有可能将车输入过来
                        in_agent_action = self.last_phase[in_agent_id] if in_agent_id not in actions else actions[in_agent_id]

                        # 先看该入路在in_agent那里是出路的第几个, 并检查是否在action的phase之内
                        in_out_road_idx = self.agents[in_agent_id][4:].index(in_road_id)
                        if in_out_road_idx in PHASE_OUT_ROAD[in_agent_action]:

                            for in_lane in PHASE_LANE[in_agent_action]:
                                if (in_lane - 1) // 3 == in_out_road_idx:
                                    in_lane_id = self.agents_lane_waiting_num[in_agent_id]['re_inroad_lane'][in_lane]

                                    if out_agent_id not in self.agent_list:
                                        full_pressure += len(lane_car[in_lane_id])
                                        continue

                                    out_agent_roads = self.agents[out_agent_id]

                                    for car in lane_car[in_lane_id]:
                                        if len(info[car]['route']) < 4:
                                            # 说明这辆车即将到达终点, 没有下一个可言, 那么exception += 1
                                            ending_car += 1
                                            continue
                                        next_road = info[car]['route'][3]

                                        # 观察next_road 和 in_road_id的关系
                                        if next_road not in out_agent_roads:
                                            continue
                                        diff = out_agent_roads.index(next_road) - out_agent_roads.index(out_road_id)
                                        # diff == -1: turn right
                                        # diff == 1: turn left
                                        # diff == 2 or -2: go straight
                                        if diff == 1:
                                            if out_rest_capacity[0] > 0:
                                                full_pressure += 1
                                                out_rest_capacity[0] -= 1

                                        if diff == 2 or diff == -2:
                                            if out_rest_capacity[1] > 0:
                                                full_pressure += 1
                                                out_rest_capacity[1] -= 1

                                        if diff == -1:
                                            full_pressure += 1


                    rest_capacity = out_rest_capacity[0] + out_rest_capacity[1]
                    full_pressure += min(rest_capacity, ending_car)

                    self.agents_lane_waiting_num[agent_id]['lane_waiting_num'][lane] = full_pressure

            self.get_actions(new_obs, observations, actions, lane_car, car_speed, info)


        return actions


        # for _ in range(self.params['rounds']):
        #
        #     for agent_id in self.agent_list:
        #         agent_adj = []
        #         for in_road_id in self.agents[agent_id][:4]:
        #             if in_road_id == -1:
        #                 agent_adj.append(-1)
        #             else:
        #                 agent_adj.append(self.roads[in_road_id]['start_inter'])
        #
        #
        #         re_inroad_lane = self.agents_lane_waiting_num[agent_id]['re_inroad_lane']
        #         lane_waiting_num = self.agents_lane_waiting_num[agent_id]['lane_waiting_num']
        #
        #         for i in range(1, 13):
        #
        #             if i in re_inroad_lane:
        #
        #                 lane = re_inroad_lane[i]
        #
        #                 if lane not in lane_waiting_num:
        #                     continue
        #
        #                 # waiting_num = lane_waiting_num[lane]
        #                 out_agent_pressure = 0
        #
        #                 out_road_idx = (IN_OUT_LANE[i][0] - 1 - 12) // 3
        #                 out_agent_id = agent_adj[out_road_idx]
        #
        #                 out_in_road_id = self.agents[agent_id][4:][out_road_idx]
        #
        #                 # 检查该out_agent_id 是否在agent_list里面
        #                 out_agent_pressure += self.agents_out_rest_capacity[agent_id][lane]
        #
        #                 if out_agent_id in self.agents_lane_waiting_num.keys():
        #
        #                     for j in range(3):
        #
        #                         if (out_in_road_id * 100 + j) not in self.agents_lane_waiting_num[out_agent_id]['lane_waiting_num']:
        #                             # this out agent has no vehicle in the road
        #                             # print("out_agent:")
        #                             # print("road id:", self.agents[agent_id][out_road_idx])
        #                             # print("road information:", self.roads[self.agents[agent_id][out_road_idx]])
        #                             # print("agents[outroad_id]", self.agents[out_agent_id])
        #                             # self.visualize(new_obs[out_agent_id]['lane_vehicle_num'], 0)
        #                             continue
        #
        #                         out_agent_pressure += self.params['out_weight'] * self.agents_lane_waiting_num[out_agent_id]['lane_waiting_num'][(out_in_road_id * 100 + j)] \
        #                         # print("waiting_num += ", self.params['out_weight'] * self.agents_lane_waiting_num[out_agent_id]['lane_waiting_num'][(out_in_road_id * 100 + j)])
        #                 in_road_id = lane // 100
        #
        #                 last_agent = self.roads[lane // 100]['start_inter']
        #
        #                 in_outroad_pressure = 0
        #                 # 检查该out_agent_id 是否在agent_list里面
        #                 if last_agent in self.agent_list:
        #                     in_road_idx_for_last_agent = self.agents[last_agent][4:].index(in_road_id)
        #
        #                     straight_lane = (in_road_idx_for_last_agent + 2) % 4 * 100 + 1
        #                     straight_lane_num = 0 if straight_lane not in lane_car else len(lane_car[straight_lane])
        #                     # straight_lane_num = 0 if straight_lane not in self.agents_lane_waiting_num[last_agent].keys() else self.agents_lane_waiting_num[last_agent][straight_lane]
        #                     turn_left_lane = (in_road_idx_for_last_agent + 3) % 4 * 100 + 0
        #                     turn_left_lane_num = 0 if turn_left_lane not in lane_car else len(lane_car[turn_left_lane])
        #                     # turn_left_lane_num = 0 if turn_left_lane not in self.agents_lane_waiting_num[last_agent].keys() else self.agents_lane_waiting_num[last_agent][turn_left_lane]
        #                     turn_right_lane = (in_road_idx_for_last_agent + 1) % 4 * 100 + 2
        #                     turn_right_lane_num = 0 if turn_right_lane not in lane_car else len(lane_car[turn_right_lane])
        #
        #                     # 右转的暂时不考虑
        #                     in_outroad_pressure = (straight_lane_num + turn_left_lane_num + turn_right_lane_num) / 3
        #
        #
        #                 # lane_waiting_num[lane] = min(out_agent_pressure, self.agents_lane_smoothed_vehicle_num[agent_id][lane])
        #                 # print("before GNN: lane_waiting_num[lane]", lane_waiting_num[lane])
        #                 # print("left: ", out_agent_pressure, "right:", in_outroad_pressure + lane_waiting_num[lane])
        #                 # print(min(out_agent_pressure, in_outroad_pressure + lane_waiting_num[lane]))
        #
        #                 if not out_agent_pressure >= lane_waiting_num[lane]:
        #                     print(out_agent_pressure)
        #                     print(lane_waiting_num[lane])
        #                 assert in_outroad_pressure + lane_waiting_num[lane] >= lane_waiting_num[lane]
        #
        #                 lane_waiting_num[lane] = min(out_agent_pressure, in_outroad_pressure * self.params['in_weight'] + lane_waiting_num[lane])
        #                 # lane_waiting_num[lane] = min(out_agent_pressure, in_outroad_pressure + self.agents_lane_waiting_num[agent_id][lane])
        #
        #                 # if out_agent_id in self.agents_lane_waiting_num.keys():
        #                 #     out_re_inroad_lane = self.agents_lane_waiting_num[out_agent_id]['re_inroad_lane']
        #                 #     out_lane_waiting_num = self.agents_lane_waiting_num[out_agent_id]['lane_waiting_num']
        #                 #
        #                 #     out_in_road_idx = ((out_road_idx) + 2) % 4 # \in (0, 1, 2, 3)
        #                 #
        #                 #     out_inroad_lane_idx = [(out_in_road_idx) * 3 + 1, (out_in_road_idx) * 3 + 2, (out_in_road_idx) * 3 + 3]
        #                 #
        #                 #     for idx in out_inroad_lane_idx:
        #                 #
        #                 #         if idx not in out_re_inroad_lane:
        #                 #             print("-" * 40)
        #                 #             print("out_road_idx:", out_road_idx)
        #                 #             print("out_in_road_idx: ", out_in_road_idx)
        #                 #             print("out_inroad_lane_idx: ", out_inroad_lane_idx)
        #                 #             print("out_re_inroad_lane:", out_re_inroad_lane)
        #                 #             print("agent_adj:", agent_adj)
        #                 #             print("agents[agent_id]: ", self.agents[agent_id])
        #                 #             print("out_agent_id:", out_agent_id)
        #                 #             print("road id:", self.agents[agent_id][out_road_idx])
        #                 #             print("road information:", self.roads[self.agents[agent_id][out_road_idx]])
        #                 #             print("agents[outroad_id]", self.agents[out_agent_id])
        #                 #             self.visualize(new_obs[agent_id]['lane_vehicle_num'], 0)
        #                 #             self.visualize(new_obs[out_agent_id]['lane_vehicle_num'], 0)
        #                 #             print("-" * 40)
        #                 #             continue
        #                 #
        #                 #         if out_re_inroad_lane[idx] not in out_lane_waiting_num:
        #                 #             continue
        #                 #         out_waiting_num = out_lane_waiting_num[out_re_inroad_lane[idx]]
        #                 #         waiting_num += self.params['out_weight'] * out_waiting_num
        #
        #
        #         self.agents_lane_waiting_num[agent_id]['lane_waiting_num'] = lane_waiting_num


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
