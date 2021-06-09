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
import copy
import time
import re
import numpy as np
from pathlib import Path

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
# ob_space_dim = ob_space_dim
# action_space_dim = action_space_dim
# env = gym.make(
#             'CBEngine-v0',
#             simulator_cfg_file=simulator_cfg_file,
#             thread_num=5,
#             gym_dict=gym_cfg_instance.cfg,
#             metric_period=200)
# env.set_log(1)
# env.set_warning(0)
# env.set_ui(0)
# env.set_info(1)


# roadlength = {}
# for line in open(path + "/inter_road_length.txt","r"):
#     a = str.split(line)
#     data=[]
#     key=int(a[0])
#     for i in range(1,5):
#         c=float(a[i])
#         data.append(c)
#     roadlength[key]=data

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

PHASE_LANE_3INTER = {
    2: [2, 8],
    # 3: [10, 4],
    5: [2, 1],
    6: [5, 4],
    # 6: [4, 4],
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
    def __init__(self):

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

        self.rotate_matrix = np.zeros([8, 8])
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
            y: x for x, y in self.inverse_clockwise_mapping.items()
        }

        for i in range(8):
            self.rotate_matrix[i][self.inverse_clockwise_mapping[i + 1] - 1] = 1

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

    def clock_wise_rotate(self, obs):


        # new_obs = copy.deepcopy(obs)
        new_obs = np.zeros(len(obs)).tolist()
        buffer = 1
        new_obs[0] = obs[0]
        new_obs[buffer + 3: buffer + 12] = obs[buffer + 0: buffer + 9]
        new_obs[buffer + 0: buffer + 3] = obs[buffer + 9: buffer + 12]
        new_obs[buffer + 15: buffer + 24] = obs[buffer + 12: buffer + 21]
        new_obs[buffer + 12: buffer + 15] = obs[buffer + 21: buffer + 24]
        return new_obs

    def obs_clock_wise_rotate(self, new_obs, obs, agent_id):
        # new_obs_clone = new_obs
        # obs_clone = obs
        # new_obs_clone = copy.deepcopy(new_obs)
        # obs_clone = copy.deepcopy(obs)

        new_obs[agent_id]['lane_vehicle_num'] = self.clock_wise_rotate(new_obs[agent_id]['lane_vehicle_num'])
        new_obs[agent_id]['lane_speed'] = self.clock_wise_rotate(new_obs[agent_id]['lane_speed'])
        obs[str(agent_id) + '_lane_vehicle_num'] = self.clock_wise_rotate(obs[str(agent_id) + '_lane_vehicle_num'])
        obs[str(agent_id) + '_lane_speed'] = self.clock_wise_rotate(obs[str(agent_id) + '_lane_speed'])
        return new_obs, obs

    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
        ################################
        # agent_list = list(self.agents.keys())

        # env_roads_length = {}
        # for road_name in env.roads.keys():
        #     env_roads_length[road_name] = env.roads[road_name]['length']

        agent_line_length = {}
        self.agent_line_capacity = {}
        self.roadlength = {}
        for agent_name in self.agent_list:
            agent_lane = []
            agent_roads = self.agents[agent_name]  # inroad和outroad,共8个，inroad是0123
            data = []
            for agent_road in agent_roads:
                if agent_road == -1:
                    agent_lane.extend([-1] * 3)
                    data.append(0)
                else:
                    agent_lane.extend([self.roads[agent_road]['length']] * 3)  # 每个lane的限速
                    data.append(self.roads[agent_road]['length'])
            self.roadlength[agent_name] = data
            agent_line_length[agent_name] = agent_lane
            temp = np.array(agent_lane)
            self.agent_line_capacity[agent_name] = (temp // 0.17).tolist()

    def max_pressure_phase(self, obs, key):
        # aim to calculate max pressure for every single phase
        # here,obs means observation for this agent
        phase_pressure = {}
        for phase in PHASE_LANE.keys():
            inroad_lanes = PHASE_LANE[phase]
            for in_lane in inroad_lanes:
                inroad_num = obs[str(key) + '_lane_vehicle_num'][in_lane]

                # 若该条路不存在
                if inroad_num == -1:
                    continue

                # 此处加入下游车道承载讷能力 capacity
                outroad_lanes = IN_OUT_LANE[in_lane]
                outroad_num_sum = 0
                rest_capacity = 0
                for out_lane in outroad_lanes:

                    if obs[str(key) + '_lane_vehicle_num'][out_lane] != -1:
                        outroad_num = obs[str(key) + '_lane_vehicle_num'][out_lane]

                        rest_capacity_temp = self.agent_line_capacity[key][out_lane - 1] - outroad_num
                        rest_capacity += rest_capacity_temp
                        outroad_num_sum += outroad_num

                if phase not in phase_pressure:
                    phase_pressure[phase] = inroad_num - rest_capacity
                elif phase in phase_pressure:
                    phase_pressure[phase] += (inroad_num - rest_capacity)

                # if phase not in phase_pressure:
                #     phase_pressure[phase] = inroad_num - outroad_num_sum
                # elif phase in phase_pressure:
                #     phase_pressure[phase] += (inroad_num - outroad_num_sum)


        # 选取最大pressure的phase
        max_phase = max(phase_pressure, key=phase_pressure.get)
        max_pressure = phase_pressure[max_phase]

        return max_phase, max_pressure, phase_pressure

    def max_pressure_phase_3inter(self, obs, key):
        # aim to calculate max pressure for every single phase
        # here,obs means observation for this agent
        phase_pressure = {}
        for phase in PHASE_LANE_3INTER.keys():
            inroad_lanes = PHASE_LANE_3INTER[phase]
            for in_lane in inroad_lanes:
                inroad_num = obs[str(key) + '_lane_vehicle_num'][in_lane]

                # 若该条路不存在
                if inroad_num == -1:
                    continue

                # 此处加入下游车道承载讷能力 capacity
                outroad_lanes = IN_OUT_LANE[in_lane]
                outroad_num = 0
                rest_capacity = 0
                for out_lane in outroad_lanes:

                    if obs[str(key) + '_lane_vehicle_num'][out_lane] != -1:
                        outroad_num = obs[str(key) + '_lane_vehicle_num'][out_lane]

                        rest_capacity_temp = self.agent_line_capacity[key][out_lane - 1] - outroad_num
                        rest_capacity += rest_capacity_temp

                if phase not in phase_pressure:
                    phase_pressure[phase] = inroad_num - rest_capacity
                elif phase in phase_pressure:
                    phase_pressure[phase] += (inroad_num - rest_capacity)

        # 选取最大pressure的phase
        max_phase = max(phase_pressure, key=phase_pressure.get)
        max_pressure = phase_pressure[max_phase]

        return max_phase, max_pressure, phase_pressure

    def phase_time(self, phase, key):
        if (phase == 1 or phase == 2):
            time = self.cal_green_time((self.roadlength[key][0] + self.roadlength[key][2]) / 2)
        elif (phase == 3 or phase == 4):
            time = self.cal_green_time((self.roadlength[key][1] + self.roadlength[key][3]) / 2)
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

    def cal_waiting(self, lane_car, car_speed, inroads):
        lane_waiting_num = {}

        inroad_lane = {}  # key是inroad_id
        re_inroad_lane = {}  # key是1-12 inroad_id
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

        # 计算堵车数量
        for idx, inroad_id in enumerate(inroads):
            if inroad_id == -1:
                continue
            lanes = inroad_lane[inroad_id]
            for lane in lanes:
                if lane not in lane_car:
                    continue
                all_cars = lane_car[lane]
                for car in all_cars:
                    speed = car_speed[car]
                    if speed < 4.5:
                        if lane not in lane_waiting_num:
                            lane_waiting_num[lane] = 1
                        elif lane in lane_waiting_num:
                            lane_waiting_num[lane] += 1

        return re_inroad_lane, lane_waiting_num

    def waiting_phase_selection(self, agent_id, lane_car, car_speed):
        # road都是按逆时针排序的
        agent_roads = self.agents[agent_id]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
        inroads_id = []
        for road_id in agent_roads[0:4]:
            if road_id != -1:
                inroads_id.append(road_id)
            else:
                inroads_id.append(-1)
        # inroads_id = [road_id for road_id in agent_roads[0:4] if road_id != -1]
        # print(inroads_id)
        # print(self.roads[4555]['lanes'])
        # print(self.roads[4809]['lanes'])
        # print(self.roads[4553]['lanes'])
        # print(self.roads[4807]['lanes'])
        # outroads_id = [road_id for road_id in agent_roads[4:8] if road_id != -1]
        # print(outroads_id)

        re_inroad_lane, lane_waiting_num = self.cal_waiting(
            lane_car=lane_car,
            car_speed=car_speed,
            inroads=inroads_id
        )  # waiting_num的key是lane_id

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

    def waiting_phase_selection_3inter(self, agent_id, lane_car, car_speed):
        # road都是按逆时针排序的
        agent_roads = self.agents[agent_id]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
        inroads_id = []
        for road_id in agent_roads[0:4]:
            if road_id != -1:
                inroads_id.append(road_id)
            else:
                inroads_id.append(-1)
        # inroads_id = [road_id for road_id in agent_roads[0:4] if road_id != -1]
        # print(inroads_id)
        # print(self.roads[4555]['lanes'])
        # print(self.roads[4809]['lanes'])
        # print(self.roads[4553]['lanes'])
        # print(self.roads[4807]['lanes'])
        # outroads_id = [road_id for road_id in agent_roads[4:8] if road_id != -1]
        # print(outroads_id)

        re_inroad_lane, lane_waiting_num = self.cal_waiting(
            lane_car=lane_car,
            car_speed=car_speed,
            inroads=inroads_id
        )  # waiting_num的key是lane_id

        # 计算哪个phase对应的lane排队数最多
        phase_waiting_nums = dict.fromkeys(PHASE_LANE_3INTER, 0)
        for phase, lanes in PHASE_LANE_3INTER.items():
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

    def second_waiting_phase_selection(self, agent_id, lane_car, car_speed, phase_list):
        # road都是按逆时针排序的
        agent_roads = self.agents[agent_id]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
        inroads_id = []
        for road_id in agent_roads[0:4]:
            if road_id != -1:
                inroads_id.append(road_id)
            else:
                inroads_id.append(-1)

        re_inroad_lane, lane_waiting_num = self.cal_waiting(
            lane_car=lane_car,
            car_speed=car_speed,
            inroads=inroads_id
        )  # waiting_num的key是lane_id

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

    def second_waiting_phase_selection_3inter(self, agent_id, lane_car, car_speed, phase_list, last_phase):
        # road都是按逆时针排序的
        agent_roads = self.agents[agent_id]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
        inroads_id = []
        for road_id in agent_roads[0:4]:
            if road_id != -1:
                inroads_id.append(road_id)
            else:
                inroads_id.append(-1)

        re_inroad_lane, lane_waiting_num = self.cal_waiting(
            lane_car=lane_car,
            car_speed=car_speed,
            inroads=inroads_id
        )  # waiting_num的key是lane_id

        # 计算哪个phase对应的lane排队数最多
        phase_waiting_nums = dict.fromkeys(PHASE_LANE_3INTER, 0)
        for phase, lanes in PHASE_LANE_3INTER.items():
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
        if len(phase_list) > 1:
            phase_waiting_nums.pop(phase_list[1])
        # print(phase_waiting_nums)
        second_max_phase = max(phase_waiting_nums, key=phase_waiting_nums.get)

        return second_max_phase

    def get_action(self, agent_id, new_obs, observations, actions, lane_car, car_speed):
        agent_obs = new_obs[agent_id]
        # select the cur_second
        for k, v in agent_obs.items():
            cur_second = v[0]
            break

        if cur_second == 0:
            max_pressure_phase, max_pressure, pressure_dict = self.max_pressure_phase(
                obs=observations,
                key=agent_id
            )
            self.last_phase[agent_id] = max_pressure_phase
            self.last_max_pressure[agent_id] = max_pressure

            actions[agent_id] = max_pressure_phase
            return

        last_phase = self.last_phase[agent_id]
        max_waiting_phase, max_waiting, waiting_dict, lane_waiting = self.waiting_phase_selection(
            agent_id=agent_id,
            lane_car=lane_car,
            car_speed=car_speed
        )
        last_max_waiting = self.last_max_waiting[agent_id]
        last_now_waiting = waiting_dict[last_phase]

        max_pressure_phase, max_pressure, pressure_dict = self.max_pressure_phase(
            obs=observations,
            key=agent_id
        )

        # print("max_pressure_phase:", max_pressure_phase)
        # print("max_pressure:", max_pressure)
        # print("pressure_dict:", pressure_dict)
        #
        # print("max_waiting_phase:", max_waiting_phase)
        # print("max_waiting:", max_waiting)
        # print("waiting_dict:", waiting_dict)
        # print("lane waiting:", lane_waiting)
        #
        # print()

        last_max_pressure = self.last_max_pressure[agent_id]
        last_now_pressure = pressure_dict[last_phase]

        step_diff = cur_second - self.last_change_step[agent_id]
        green_time = self.phase_time(last_phase, agent_id)

        # select phase
        select_phase = None
        phase_no_effect1 = None
        phase_no_effect2 = None
        phase_list = list(range(1, 9))
        threshold = 5
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

                    if self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0] - 1] in lane_waiting.keys() and \
                            self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][1] - 1] in lane_waiting.keys():

                        if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][0]]] / lane_waiting[
                            self.intersections[agent_id]['lanes'][
                                PHASE_LANE[last_phase][0] - 1]]) < threshold and float(
                            self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][1]]] / lane_waiting[
                                self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][1] - 1]]) < threshold:

                            if lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0] - 1]] > \
                                    lane_waiting[self.intersections[agent_id]['lanes'][
                                        PHASE_LANE[last_phase][1] - 1]]:  # 看该phase中哪条路堵的厉害
                                phase_no_effect1 = last_phase
                                phase_no_effect2 = [k for k, v in PHASE_LANE.items() if PHASE_LANE[last_phase][0] in v]

                                print("phase_no_effect2:", phase_no_effect2)

                                select_phase = self.second_waiting_phase_selection(agent_id=agent_id, lane_car=lane_car,
                                                                                   car_speed=car_speed,
                                                                                   phase_list=phase_no_effect2)
                                print(select_phase)
                                actions[agent_id] = select_phase
                            else:
                                phase_no_effect1 = last_phase
                                phase_no_effect2 = [k for k, v in PHASE_LANE.items() if PHASE_LANE[last_phase][1] in v]
                                # 去掉最堵的路对应的两个phase，在剩下6个中做选择

                                print("phase_no_effect2:", phase_no_effect2)

                                # 随机
                                select_phase = self.second_waiting_phase_selection(agent_id=agent_id, lane_car=lane_car,
                                                                                   car_speed=car_speed,
                                                                                   phase_list=phase_no_effect2)
                                print(select_phase)
                                actions[agent_id] = select_phase

                    elif self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0] - 1] in lane_waiting.keys() and \
                            self.intersections[agent_id]['lanes'][
                                PHASE_LANE[last_phase][1] - 1] not in lane_waiting.keys():
                        # print('\n0\n')
                        # print('\n', lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0]]])
                        if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][0]]] / lane_waiting[
                            self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][0] - 1]]) < threshold:
                            phase_no_effect1 = last_phase
                            phase_no_effect2 = [k for k, v in PHASE_LANE.items() if PHASE_LANE[last_phase][0] in v]

                            print("phase_no_effect2:", phase_no_effect2)

                            select_phase = self.second_waiting_phase_selection(agent_id=agent_id, lane_car=lane_car,
                                                                               car_speed=car_speed,
                                                                               phase_list=phase_no_effect2)
                            # print(select_phase)
                            actions[agent_id] = select_phase
                    elif self.intersections[agent_id]['lanes'][
                        PHASE_LANE[last_phase][0] - 1] not in lane_waiting.keys() and \
                            self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][1] - 1] in lane_waiting.keys():
                        # print('\n1\n')
                        # print('\n', lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][1]]])
                        if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE[last_phase][1]]] / lane_waiting[
                            self.intersections[agent_id]['lanes'][PHASE_LANE[last_phase][1] - 1]]) < threshold:
                            phase_no_effect1 = last_phase
                            phase_no_effect2 = [k for k, v in PHASE_LANE.items() if PHASE_LANE[last_phase][1] in v]

                            print("phase_no_effect2:", phase_no_effect2)

                            select_phase = self.second_waiting_phase_selection(agent_id=agent_id, lane_car=lane_car,
                                                                               car_speed=car_speed,
                                                                               phase_list=phase_no_effect2)
                            # print(select_phase)
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
                if (last_now_waiting == 0):
                    select_phase = max_waiting_phase
                # elif (max_waiting <= 5):
                #     if (pressure_dict[max_waiting_phase] <= last_now_pressure):
                #         select_phase = last_phase
                #     else:
                #         select_phase = max_waiting_phase
                # elif (last_now_waiting >= 1 and max_waiting <= 10):
                #     select_phase = last_phase
                elif (
                        last_now_waiting / max_waiting < 0.60 and last_now_waiting / max_waiting > 0.30 and max_waiting <= 8):
                    select_phase = last_phase
                elif (last_now_waiting / max_waiting >= 0.60):
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

        # print("action:", actions[agent_id])
        if agent_id not in actions.keys():
            if not self.last_phase[agent_id] == actions[agent_id]:
                print("4 intersections")
                print("last:", self.last_phase[agent_id])
                print("now:", actions[agent_id])
        self.last_max_waiting[agent_id] = max_waiting
        self.last_max_pressure[agent_id] = max_pressure



    def get_action_3inter_pressure(self, agent_id, new_obs, observations, actions, lane_car, car_speed):
        pressures = []
        for phase, in_lanes  in PHASE_LANE_3INTER.items():
            pressure = 0
            for in_lane in in_lanes:
                pressure += new_obs[agent_id]['lane_vehicle_num'][in_lane] * 3
                for out_lane in IN_OUT_LANE[in_lane]:
                    pressure -= new_obs[agent_id]['lane_vehicle_num'][out_lane]
            pressures.append(pressure)
        max_pressure_id = np.argmax(pressures)
        max_pressure_id = list(PHASE_LANE_3INTER.keys())[max_pressure_id]
        self.last_phase = max_pressure_id
        actions[agent_id] = max_pressure_id

    def get_action_3inter(self, agent_id, new_obs, observations, actions, lane_car, car_speed):
        agent_obs = new_obs[agent_id]
        # select the cur_second
        for k, v in agent_obs.items():
            cur_second = v[0]
            break

        if cur_second == 0:
            max_pressure_phase, max_pressure, pressure_dict = self.max_pressure_phase_3inter(
                obs=observations,
                key=agent_id
            )

            self.last_phase[agent_id] = max_pressure_phase
            self.last_max_pressure[agent_id] = max_pressure

            actions[agent_id] = max_pressure_phase
            return

        last_phase = self.last_phase[agent_id]
        max_waiting_phase, max_waiting, waiting_dict, lane_waiting = self.waiting_phase_selection_3inter(
            agent_id=agent_id,
            lane_car=lane_car,
            car_speed=car_speed
        )
        last_max_waiting = self.last_max_waiting[agent_id]
        last_now_waiting = waiting_dict[last_phase]


        # print("max_waiting_phase:", max_waiting_phase)
        # print("max_waiting:", max_waiting)
        # print("waiting_dict:", waiting_dict)
        # print("lane waiting:", lane_waiting)


        max_pressure_phase, max_pressure, pressure_dict = self.max_pressure_phase_3inter(
            obs=observations,
            key=agent_id
        )

        # print("max_pressure_phase:", max_pressure_phase)
        # print("max_pressure:", max_pressure)
        # print("pressure_dict:", pressure_dict)



        last_max_pressure = self.last_max_pressure[agent_id]
        last_now_pressure = pressure_dict[last_phase]

        step_diff = cur_second - self.last_change_step[agent_id]
        green_time = self.phase_time(last_phase, agent_id)

        # select phase
        select_phase = None
        phase_no_effect1 = None
        phase_no_effect2 = None
        phase_list = list(range(1, 9))
        threshold = 5
        if max_waiting != 0:

            if cur_second not in self.waiting_times:
                self.waiting_times[cur_second] = 1
            elif cur_second in self.waiting_times:
                self.waiting_times[cur_second] += 1

            if (max_waiting_phase == last_phase):  # phase doesn't change

                if last_max_waiting == max_waiting:  # 上游堵到一定程度时，如果phase维持不变，上游堵车依然不能缓解时，不如避开相关的phase，去尝试其他的phase

                    '''
                    self.intersections[agent_id]['lanes'][PHASE_LANE_3INTER[last_phase][0]] 
                    intersection是路口，结构为{agent_id:{'xxx':abc, 'lanes':[inlane_id1, inlane_id2, ..., inlane_id12, outlane_id1, ..., outlane_id12]}}  
                    PHASE_LANE_3INTER[last_phase][0]是上一个phase对应的第一条lane的序号，例如对于phase1来说，即lane1或者lane7（不是lane_id，不是378000这种）
                    '''

                    if self.intersections[agent_id]['lanes'][
                        PHASE_LANE_3INTER[last_phase][0] - 1] in lane_waiting.keys() and \
                            self.intersections[agent_id]['lanes'][
                                PHASE_LANE_3INTER[last_phase][1] - 1] in lane_waiting.keys():

                        if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE_3INTER[last_phase][0]]] / lane_waiting[
                            self.intersections[agent_id]['lanes'][
                                PHASE_LANE_3INTER[last_phase][0] - 1]]) < threshold and float(
                            self.roadlength[agent_id][LANE_ROAD[PHASE_LANE_3INTER[last_phase][1]]] / lane_waiting[
                                self.intersections[agent_id]['lanes'][
                                    PHASE_LANE_3INTER[last_phase][1] - 1]]) < threshold:

                            if lane_waiting[
                                self.intersections[agent_id]['lanes'][PHASE_LANE_3INTER[last_phase][0] - 1]] > \
                                    lane_waiting[self.intersections[agent_id]['lanes'][
                                        PHASE_LANE_3INTER[last_phase][1] - 1]]:  # 看该phase中哪条路堵的厉害
                                phase_no_effect1 = last_phase
                                phase_no_effect2 = [k for k, v in PHASE_LANE_3INTER.items() if
                                                    PHASE_LANE_3INTER[last_phase][0] in v]

                                select_phase = self.second_waiting_phase_selection_3inter(agent_id=agent_id, lane_car=lane_car,
                                                                                   car_speed=car_speed,
                                                                                   phase_list=phase_no_effect2, last_phase=last_phase)
                                # print(select_phase)
                                actions[agent_id] = select_phase
                            else:
                                phase_no_effect1 = last_phase
                                phase_no_effect2 = [k for k, v in PHASE_LANE_3INTER.items() if
                                                    PHASE_LANE_3INTER[last_phase][1] in v]
                                # 去掉最堵的路对应的两个phase，在剩下6个中做选择

                                # 随机
                                select_phase = self.second_waiting_phase_selection_3inter(agent_id=agent_id, lane_car=lane_car,
                                                                                   car_speed=car_speed,
                                                                                   phase_list=phase_no_effect2, last_phase=last_phase)
                                # print(select_phase)
                                actions[agent_id] = select_phase

                    elif self.intersections[agent_id]['lanes'][
                        PHASE_LANE_3INTER[last_phase][0] - 1] in lane_waiting.keys() and \
                            self.intersections[agent_id]['lanes'][
                                PHASE_LANE_3INTER[last_phase][1] - 1] not in lane_waiting.keys():
                        # print('\n0\n')
                        # print('\n', lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE_3INTER[last_phase][0]]])
                        if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE_3INTER[last_phase][0]]] / lane_waiting[
                            self.intersections[agent_id]['lanes'][PHASE_LANE_3INTER[last_phase][0] - 1]]) < threshold:
                            phase_no_effect1 = last_phase
                            phase_no_effect2 = [k for k, v in PHASE_LANE_3INTER.items() if
                                                PHASE_LANE_3INTER[last_phase][0] in v]

                            select_phase = self.second_waiting_phase_selection_3inter(agent_id=agent_id, lane_car=lane_car,
                                                                               car_speed=car_speed,
                                                                               phase_list=phase_no_effect2, last_phase=last_phase)
                            # print(select_phase)
                            actions[agent_id] = select_phase
                    elif self.intersections[agent_id]['lanes'][
                        PHASE_LANE_3INTER[last_phase][0] - 1] not in lane_waiting.keys() and \
                            self.intersections[agent_id]['lanes'][
                                PHASE_LANE_3INTER[last_phase][1] - 1] in lane_waiting.keys():
                        # print('\n1\n')
                        # print('\n', lane_waiting[self.intersections[agent_id]['lanes'][PHASE_LANE_3INTER[last_phase][1]]])
                        if float(self.roadlength[agent_id][LANE_ROAD[PHASE_LANE_3INTER[last_phase][1]]] / lane_waiting[
                            self.intersections[agent_id]['lanes'][PHASE_LANE_3INTER[last_phase][1] - 1]]) < threshold:
                            phase_no_effect1 = last_phase
                            phase_no_effect2 = [k for k, v in PHASE_LANE_3INTER.items() if
                                                PHASE_LANE_3INTER[last_phase][1] in v]

                            select_phase = self.second_waiting_phase_selection_3inter(agent_id=agent_id, lane_car=lane_car,
                                                                               car_speed=car_speed,
                                                                               phase_list=phase_no_effect2, last_phase=last_phase)
                            # print(select_phase)
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
                if (last_now_waiting == 0):
                    select_phase = max_waiting_phase
                # elif (max_waiting <= 5):
                #     if (pressure_dict[max_waiting_phase] <= last_now_pressure):
                #         select_phase = last_phase
                #     else:
                #         select_phase = max_waiting_phase
                # elif (last_now_waiting >= 1 and max_waiting <= 10):
                #     select_phase = last_phase
                elif (
                        last_now_waiting / max_waiting < 0.60 and last_now_waiting / max_waiting > 0.30 and max_waiting <= 8):
                    select_phase = last_phase
                elif (last_now_waiting / max_waiting >= 0.60):
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


        # if actions[agent_id] not in [2,5,6]:
        #     actions[agent_id] = np.random.choice([2, 5, 6])
        #     print("agent_id not in actions")
        #     print("max_waiting:", max_waiting)
        #     print("max_waiting_phase == last_phase: ", max_waiting_phase == last_phase)
        #     print("last_max_waiting == max_waiting:", last_max_waiting == max_waiting)
        #     print("condition_1:", self.intersections[agent_id]['lanes'][
        #                 PHASE_LANE_3INTER[last_phase][0] - 1] in lane_waiting.keys())
        #     print("condition_2:", self.intersections[agent_id]['lanes'][
        #                         PHASE_LANE_3INTER[last_phase][1] - 1] not in lane_waiting.keys())
        #     print("max_pressure:", max_pressure)

        if agent_id in actions.keys():
            self.last_phase[agent_id] = actions[agent_id]
            # if not self.last_phase[agent_id] == actions[agent_id]:
            #     print("last:", self.last_phase[agent_id])
            #     print("now:", actions[agent_id])
        self.last_max_waiting[agent_id] = max_waiting
        self.last_max_pressure[agent_id] = max_pressure

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

        # get actions

        for agent_id in self.agent_list:
            if -1 in new_obs[agent_id]['lane_vehicle_num']:
                # start = time.time()
                idx = self.agents[agent_id][:4].index(-1)

                for _ in range(3 - idx):
                    new_obs, observations = self.obs_clock_wise_rotate(new_obs, observations, agent_id)

                self.get_action_3inter(agent_id, new_obs, observations, actions, lane_car, car_speed)

                if agent_id in actions.keys():
                    # if np.random.random() < 0.0001:
                    #     print("3 intersections")
                    #     self.visualize(new_obs[agent_id]['lane_vehicle_num'], actions[agent_id])
                    for _ in range(3 - idx):
                        actions[agent_id] = self.inverse_clockwise_mapping[actions[agent_id]]

            else:
                self.get_action(agent_id, new_obs, observations, actions, lane_car, car_speed)

                # if np.random.random() < 0.0001:
                #     if agent_id in actions.keys():
                #         print("4 intersections")
                #         self.visualize(new_obs[agent_id]['lane_vehicle_num'], actions[agent_id])


        return actions


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()

