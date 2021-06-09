import pickle
import CBEngine
import json
import traceback
import argparse
import logging
import os
import sys
import time
from pathlib import Path
import re
import gym
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

gym.logger.setLevel(gym.logger.ERROR)


def pretty_files(path):
    contents = os.listdir(path)
    return "[{}]".format(", ".join(contents))


def resolve_dirs(root_path: str, input_dir: str = None, output_dir: str = None):
    root_path = Path(root_path)

    logger.info(f"root_path={pretty_files(root_path)}")

    if input_dir is not None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        submission_dir = input_dir
        scores_dir = output_dir

        logger.info(f"input_dir={pretty_files(input_dir)}")
        logger.info(f"output_dir={pretty_files(output_dir)}")
    else:
        raise ValueError('need input dir')

    if not scores_dir.exists():
        os.makedirs(scores_dir)

    logger.info(f"submission_dir={pretty_files(submission_dir)}")
    logger.info(f"scores_dir={pretty_files(scores_dir)}")

    if not submission_dir.is_dir():
        logger.warning(f"submission_dir={submission_dir} does not exist")

    return submission_dir, scores_dir


def load_agent_submission(submission_dir: Path):
    logger.info(f"files under submission dir:{pretty_files(submission_dir)}")

    # find agent.py
    module_path = None
    cfg_path = None
    for dirpath, dirnames, file_names in os.walk(submission_dir):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "agent.py":
                module_path = dirpath

            if file_name == "gym_cfg.py":
                cfg_path = dirpath
    # error
    assert (
        module_path is not None
    ), "Cannot find file named agent.py, please check your submission zip"
    assert(
        cfg_path is not None
    ), "Cannot find file named gym_cfg.py, please check your submission zip"
    sys.path.append(str(module_path))


    # This will fail w/ an import error of the submissions directory does not exist
    import gym_cfg as gym_cfg_submission
    import agent as agent_submission

    gym_cfg_instance = gym_cfg_submission.gym_cfg()

    return  agent_submission.agent_specs,gym_cfg_instance


def read_config(cfg_file):
    configs = {}
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if(len(line) == 3 and line[0][0] != '#'):
                configs[line[0]] = line[-1]
    return configs


def process_roadnet(roadnet_file):
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

    intersections = {}
    roads = {}
    agents = {}

    agent_num = 0
    road_num = 0
    signal_num = 0
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {
                        'have_signal': int(line[3]),
                        'end_roads': [],
                        'start_roads': [],
                        'lanes':[]
                    }
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4]),
                            'inverse_road': int(line[-1])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5]),
                            'inverse_road': int(line[-2])
                        }
                        intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                        intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                        intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    signal_road_order = list(map(int, line[1:]))
                    now_agent = int(line[0])
                    in_roads = []
                    for road in signal_road_order:
                        if (road != -1):
                            in_roads.append(roads[road]['inverse_road'])
                        else:
                            in_roads.append(-1)
                    in_roads += signal_road_order
                    agents[now_agent] = in_roads
    for agent, agent_roads in agents.items():
        intersections[agent]['lanes'] = []
        for road in agent_roads:
            ## here we treat road -1 have 3 lanes
            if (road == -1):
                for i in range(3):
                    intersections[agent]['lanes'].append(-1)
            else:
                for lane in roads[road]['lanes'].keys():
                    intersections[agent]['lanes'].append(lane)

    return intersections, roads, agents


def process_delay_index(lines, roads, step):
    vehicles = {}

    for i in range(len(lines)):
        line = lines[i]
        if(line[0] == 'for'):
            vehicle_id = int(line[2])
            now_dict = {
                'distance': float(lines[i + 1][2]),
                'drivable': int(float(lines[i + 2][2])),
                'road': int(float(lines[i + 3][2])),
                'route': list(map(int, list(map(float, lines[i + 4][2:])))),
                'speed': float(lines[i + 5][2]),
                'start_time': float(lines[i + 6][2]),
                't_ff': float(lines[i+7][2]),
            ##############
                'step': int(lines[i+8][2])
            }
            step = now_dict['step']
            ##################
            vehicles[vehicle_id] = now_dict
            tt = step - now_dict['start_time']
            tt_ff = now_dict['t_ff']
            tt_f_r = 0.0
            current_road_pos = 0
            for pos in range(len(now_dict['route'])):
                if(now_dict['road'] == now_dict['route'][pos]):
                    current_road_pos = pos
            for pos in range(len(now_dict['route'])):
                road_id = now_dict['route'][pos]
                if(pos == current_road_pos):
                    tt_f_r += (roads[road_id]['length'] -
                               now_dict['distance']) / roads[road_id]['speed_limit']
                elif(pos > current_road_pos):
                    tt_f_r += roads[road_id]['length'] / roads[road_id]['speed_limit']
            vehicles[vehicle_id]['tt_f_r'] = tt_f_r
            vehicles[vehicle_id]['delay_index'] = (tt + tt_f_r) / tt_ff

    vehicle_list = list(vehicles.keys())
    delay_index_list = []
    for vehicle_id, dict in vehicles.items():
        # res = max(res, dict['delay_index'])
        if('delay_index' in dict.keys()):
            delay_index_list.append(dict['delay_index'])

    # 'delay_index_list' contains all vehicles' delayindex at this snapshot.
    # 'vehicle_list' contains the vehicle_id at this snapshot.
    # 'vehicles' is a dict contains vehicle infomation at this snapshot
    return delay_index_list, vehicle_list, vehicles

def process_score(log_path,roads,step,scores_dir):
    result_write = {
        "data": {
            "total_served_vehicles": -1,
            "delay_index": -1
        }
    }

    with open(log_path / "info_step {}.log".format(step)) as log_file:
        lines = log_file.readlines()
        lines = list(map(lambda x: x.rstrip('\n').split(' '), lines))
        # process delay index
        delay_index_list, vehicle_list, vehicles = process_delay_index(lines, roads, step)
        v_len = len(vehicle_list)
        delay_index = np.mean(delay_index_list)

        result_write['data']['total_served_vehicles'] = v_len
        result_write['data']['delay_index'] = delay_index
        with open(scores_dir / 'scores {}.json'.format(step), 'w' ) as f_out:
            json.dump(result_write,f_out,indent= 2)

    return result_write['data']['total_served_vehicles'],result_write['data']['delay_index']


def cal_waiting(lane_car, car_speed, inroads, roads):
    lane_waiting_num = {}

    inroad_lane = {}  # key是inroad_id
    re_inroad_lane = {}  # key是1-12 inroad_id
    for idx, inroad_id in enumerate(inroads):  # 获取每一个inroad的3条lane_id
        if inroad_id == -1:
            continue
        lane_dict = roads[inroad_id]['lanes']
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
                if speed < 5.5:
                    if lane not in lane_waiting_num:
                        lane_waiting_num[lane] = 1
                    elif lane in lane_waiting_num:
                        lane_waiting_num[lane] += 1

    return re_inroad_lane, lane_waiting_num


def get_lane_waiting_num(infos, agent_id, agent):
    lane_car = {}  # key为lane_id val为car_id是一个list
    car_speed = {}  # key为car_id val为速度
    for car_id, car_info in infos.items():
        cur_lane = car_info['drivable'][0]
        cur_speed = car_info['speed'][0]
        if cur_lane not in lane_car:
            lane_car[cur_lane] = [car_id]
        elif cur_lane in lane_car:
            lane_car[cur_lane].append(car_id)
        car_speed[car_id] = cur_speed

    agent_roads = agent.agents[agent_id]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
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



class Approx_Vehicle_Num():
    def __init__(self):
        self.lane_car = None
        self.car_speed = None
        self.inroads_id = None
        self.outroads_id = None
        
        
    def get_approx_vehicle_num(self, lane_vehicle_num, info, agents, roads, agent_id):
        
        if self.lane_car is None:
            self.lane_car = {}  # key为lane_id val为car_id是一个list
            self.car_speed = {}  # key为car_id val为速度
            for car_id, car_info in info.items():
                cur_lane = car_info['drivable'][0]
                cur_speed = car_info['speed'][0]
                if cur_lane not in self.lane_car:
                    self.lane_car[cur_lane] = [car_id]
                elif cur_lane in self.lane_car:
                    self.lane_car[cur_lane].append(car_id)
                self.car_speed[car_id] = cur_speed
    
            agent_roads = agents[agent_id]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
            self.inroads_id = []
            for road_id in agent_roads[0:4]:
                if road_id != -1:
                    self.inroads_id.append(road_id)
                else:
                    self.inroads_id.append(-1)
    
            self.outroads_id = []
            for road_id in agent_roads[4:]:
                if road_id != -1:
                    self.outroads_id.append(road_id)
                else:
                    self.outroads_id.append(-1)

        re_inroad_lane, in_lane_waiting_num = cal_waiting(
            lane_car=self.lane_car,
            car_speed=self.car_speed,
            inroads=self.inroads_id,
            roads=roads
        )  # waiting_num的key是lane_id

        re_outroad_lane, out_lane_waiting_num = cal_waiting(
            lane_car=self.lane_car,
            car_speed=self.car_speed,
            inroads=self.outroads_id,
            roads=roads
        )  # waiting_num的key是lane_id

        lane_vehicle_num_real = lane_vehicle_num
        lane_vehicle_num = [lane_vehicle_num_real[0]]

        for i in range(12):
            if i + 1 not in re_inroad_lane.keys():
                lane_vehicle_num.append(-1)
            else:
                if re_inroad_lane[i + 1] not in in_lane_waiting_num:
                    lane_vehicle_num.append(0)
                else:
                    lane_vehicle_num.append(in_lane_waiting_num[re_inroad_lane[i + 1]])

        for i in range(12):
            if i + 1 not in re_outroad_lane.keys():
                lane_vehicle_num.append(-1)
            else:
                if re_outroad_lane[i + 1] not in out_lane_waiting_num:
                    lane_vehicle_num.append(0)
                else:
                    lane_vehicle_num.append(out_lane_waiting_num[re_outroad_lane[i + 1]])

        return lane_vehicle_num

def add_to_train_data(train_data, last_observations, observations, last_infos ,infos, new_infos, actions, new_observations, agent_id_list, last_actions, agent):

    last_obs = {}
    for key, val in last_observations.items():
        agent_id = int(key.split('_')[0])
        feature = key[key.find('_') + 1:]  # feature为lane_num或lane_speed
        if (agent_id not in last_obs.keys()):
            last_obs[agent_id] = {}
        last_obs[agent_id][feature] = val


    obs = {}  # 嵌套字典 第一级key为agent_id 第二级key为feature
    for key, val in observations.items():
        agent_id = int(key.split('_')[0])  # 获取agent_id
        feature = key[key.find('_') + 1:]  # feature为lane_num或lane_speed
        if (agent_id not in obs.keys()):
            obs[agent_id] = {}
        obs[agent_id][feature] = val

    new_obs = {}
    for key, val in new_observations.items():
        agent_id = int(key.split('_')[0])  # 获取agent_id
        feature = key[key.find('_') + 1:]  # feature为lane_num或lane_speed
        if (agent_id not in new_obs.keys()):
            new_obs[agent_id] = {}
        new_obs[agent_id][feature] = val


    lane_car = {}  # key为lane_id val为car_id是一个list
    car_speed = {}  # key为car_id val为速度
    for car_id, car_info in infos.items():
        cur_lane = car_info['drivable'][0]
        cur_speed = car_info['speed'][0]
        if cur_lane not in lane_car:
            lane_car[cur_lane] = [car_id]
        elif cur_lane in lane_car:
            lane_car[cur_lane].append(car_id)
        car_speed[car_id] = cur_speed

    last_infos_approx = Approx_Vehicle_Num()
    infos_approx = Approx_Vehicle_Num()
    new_infos_approx = Approx_Vehicle_Num()

    for agent_id in agent_id_list:
        train_data['last_obs'].append(last_infos_approx.get_approx_vehicle_num(last_obs[agent_id]['lane_vehicle_num'],
                        last_infos, agent.agents, agent.roads, agent_id))
        train_data['last_obs_speed'].append(last_obs[agent_id]['lane_speed'])
        train_data['obs'].append(infos_approx.get_approx_vehicle_num(obs[agent_id]['lane_vehicle_num'],
                        infos, agent.agents, agent.roads, agent_id))
        train_data['obs_speed'].append(obs[agent_id]['lane_speed'])
        train_data['new_obs'].append(new_infos_approx.get_approx_vehicle_num(new_obs[agent_id]['lane_vehicle_num'],
                        new_infos, agent.agents, agent.roads, agent_id))
        train_data['new_obs_speed'].append(new_obs[agent_id]['lane_speed'])

        if agent_id not in actions.keys():
            train_data['action'].append(last_actions[agent_id])
        else:
            train_data['action'].append(actions[agent_id])

        train_data['last_action'].append(last_actions[agent_id])
        # adj_observations = []
        # for i in range(4):
        #     agent.agents[agent_id]

def run_simulation(agent_spec, simulator_cfg_file, gym_cfg, metric_period, scores_dir, threshold, train_data):
    logger.info("\n")
    logger.info("*" * 40)

    # get gym instance
    gym_configs = gym_cfg.cfg
    simulator_configs = read_config(simulator_cfg_file)
    env = gym.make(
        'CBEngine-v0',
        simulator_cfg_file=simulator_cfg_file,
        thread_num=1,
        gym_dict=gym_configs,
        metric_period=metric_period
    )
    scenario = [
        'test'
    ]

    # read roadnet file, get data
    roadnet_path = Path(simulator_configs['road_file_addr'])
    intersections, roads, agents = process_roadnet(roadnet_path)
    env.set_warning(0)
    env.set_log(0)
    env.set_info(1)
    env.set_ui(1)
    # get agent instance
    observations, infos = env.reset()
    agent_id_list = []
    for k in observations:
        agent_id_list.append(int(k.split('_')[0]))
    agent_id_list = list(set(agent_id_list))
    agent = agent_spec[scenario[0]]
    agent.load_agent_list(agent_id_list)
    agent.load_roadnet(intersections, roads, agents)
    done = False
    # simulation
    step = 0
    log_path = Path(simulator_configs['report_log_addr'])
    sim_start = time.time()

    tot_v = -1
    d_i = -1

    last_actions = {}

    step = 0
    while not done:
        actions = {}
        step += 1

        all_info = {
            'observations': observations,
            'info': infos
        }

        actions = agent.act(all_info)

        t1 = time.time()
        new_observations, rewards, dones, new_infos = env.step(actions)
        t2 = time.time()

        # for i, agent_id in enumerate(agent_id_list):
        #     if i == 0:
        #         print(new_observations[str(agent_id) + '_lane_vehicle_num'])
        #     if i == 10:
        #         print(new_observations[str(agent_id) + '_lane_vehicle_num'])
        #     elif i > 10:
        #         break

        if len(last_actions) > 0:
            add_to_train_data(train_data, last_observations, observations, last_infos, infos, new_infos, actions, new_observations, agent_id_list, last_actions, agent)
            last_actions.update(actions)
        else:
            last_actions = actions

        last_observations, last_infos = observations, infos
        observations, infos = new_observations, new_infos
        t3 = time.time()

        # print("t3 - t2: ", t3 - t2)
        # print("t2 - t1: ", t2 - t1)

        if (step * 10 % metric_period == 0):
            try:
                tot_v, d_i = process_score(log_path, roads, step * 10 - 1, scores_dir)
            except Exception as e:
                print(e)
                print('Error in process_score. Maybe no log')
                continue
        if (d_i > threshold):
            print(d_i)
            break

        if all(dones.values()):
            done = True

    sim_end = time.time()
    logger.info("simulation cost : {}s".format(sim_end - sim_start))

    # read log file

    # result = {}
    # vehicle_last_occur = {}

    # eval_start = time.time()
    # for dirpath, dirnames, file_names in os.walk(log_path):
    #     for file_name in [f for f in file_names if f.endswith(".log") and f.startswith('info_step')]:
    #         with open(log_path / file_name, 'r') as log_file:
    #             pattern = '[0-9]+'
    #             step = list(map(int, re.findall(pattern, file_name)))[0]
    #             if(step >= int(simulator_configs['max_time_epoch'])):
    #                 continue
    #             lines = log_file.readlines()
    #             lines = list(map(lambda x: x.rstrip('\n').split(' '), lines))
    #             result[step] = {}
    #             # result[step]['vehicle_num'] = int(lines[0][0])
    #
    #             # process delay index
    #             delay_index_list, vehicle_list, vehicles = process_delay_index(lines, roads, step)
    #             result[step]['vehicle_list'] = vehicle_list
    #             result[step]['delay_index'] = delay_index_list
    #             result[step]['vehicles'] = vehicles
    #
    #
    # steps = list(result.keys())
    # steps.sort()
    # for step in steps:
    #     for vehicle in result[step]['vehicles'].keys():
    #         vehicle_last_occur[vehicle] = result[step]['vehicles'][vehicle]
    #
    # delay_index_temp = {}
    # for vehicle in vehicle_last_occur.keys():
    #     if('delay_index' in vehicle_last_occur[vehicle]):
    #         res = vehicle_last_occur[vehicle]['delay_index']
    #         delay_index_temp[vehicle] = res
    #
    # # calc
    # vehicle_total_set = set()
    # delay_index = []
    # for k, v in result.items():
    #     vehicle_total_set = vehicle_total_set | set(v['vehicle_list'])
    #     delay_index += delay_index_list
    #
    # if(len(delay_index)>0):
    #     d_i = np.mean(delay_index)
    # else:
    #     d_i = -1
    #
    # last_d_i = np.mean(list(delay_index_temp.values()))
    # eval_end = time.time()
    # logger.info("scoring cost {}s".format(eval_end-eval_start))
    return tot_v, d_i


def format_exception(grep_word):
    exception_list = traceback.format_stack()
    exception_list = exception_list[:-2]
    exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
    exception_list.extend(traceback.format_exception_only(
        sys.exc_info()[0], sys.exc_info()[1]))
    filtered = []
    for m in exception_list:
        if str(grep_word) in m:
            filtered.append(m)

    exception_str = "Traceback (most recent call last):\n"
    exception_str += "".join(filtered)
    # Removing the last \n
    exception_str = exception_str[:-1]

    return exception_str

if __name__ == "__main__":


    # arg parse
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="1"
    )

    parser.add_argument(
        "--input_dir",
        help="The path to the directory containing the reference "
             "data and user submission data.",
        default='agent',
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="The path to the directory where the submission's "
             "scores.txt file will be written to.",
        default='out',
        type=str,
    )

    parser.add_argument(
        "--file_name",
        default='train_data',
        type=str
    )

    parser.add_argument(
        "--sim_cfg",
        help='The path to the simulator cfg',
        default='cfg/simulator.cfg',
        type=str
    )

    parser.add_argument(
        "--metric_period",
        help="period of scoring",
        default=3600,
        type=int
    )
    parser.add_argument(
        "--threshold",
        help="period of scoring",
        default=1.2,
        type=float
    )

    # result to be written in out/result.json
    result = {
        "success": False,
        "error_msg": "",
        "data": {
            "total_served_vehicles": -1,
            "delay_index": -1
        }
    }

    args = parser.parse_args()
    msg = None
    metric_period = args.metric_period
    threshold = args.threshold
    # get input and output directory
    simulator_cfg_file = args.sim_cfg
    try:
        submission_dir, scores_dir = resolve_dirs(
            os.path.dirname(__file__), args.input_dir, args.output_dir
        )
    except Exception as e:
        msg = format_exception(e)
        result['error_msg'] = msg
        json.dump(result,open(scores_dir / "scores.json",'w'),indent=2)
        raise AssertionError()

    # get agent and configuration of gym
    try:
        agent_spec,gym_cfg = load_agent_submission(submission_dir)
    except Exception as e:
        msg = format_exception(e)
        result['error_msg'] = msg
        json.dump(result,open(scores_dir / "scores.json",'w'),indent=2)
        raise AssertionError()

    logger.info(f"Loaded user agent instance={agent_spec}")

    # simulation
    start_time = time.time()

    train_data = {}
    train_data['last_obs'] = []
    train_data['last_obs_speed'] = []
    train_data['obs'] = []
    # train_data['adj_obs'] = []
    train_data['obs_speed'] = []
    train_data['action'] = []
    train_data['last_action'] = []
    train_data['new_obs'] = []
    train_data['new_obs_speed'] = []

    try:
        for i in range(10000):
            scores = run_simulation(agent_spec, simulator_cfg_file, gym_cfg,metric_period,scores_dir,threshold, train_data)
            print(f"Epoch {i} done, total_served_vehicles = {scores[0]}, delay_index = {scores[1]}")
            sys.stdout.flush()
            with open('./data/' + args.file_name + '.pkl', 'wb') as file:
                pickle.dump(train_data, file)

    except Exception as e:
        msg = format_exception(e)
        result['error_msg'] = msg
        json.dump(result,open(scores_dir / "scores.json",'w'),indent=2)
        raise AssertionError()

    # write result
    result['data']['total_served_vehicles'] = scores[0]
    result['data']['delay_index'] = scores[1]
    # result['data']['last_d_i'] = scores[2]
    result['success'] = True

    # cal time
    end_time = time.time()

    logger.info(f"total evaluation cost {end_time-start_time} s")

    # write score
    logger.info("\n\n")
    logger.info("*" * 40)

    json.dump(result, open(scores_dir / "scores.json", 'w'), indent=2)

    logger.info("Evaluation complete")
