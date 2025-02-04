import datetime

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
from agent.configs import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

gym.logger.setLevel(gym.logger.ERROR)

import warnings

warnings.filterwarnings("ignore")

With_Speed = False


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
    assert (
            cfg_path is not None
    ), "Cannot find file named gym_cfg.py, please check your submission zip"
    sys.path.append(str(module_path))

    # This will fail w/ an import error of the submissions directory does not exist
    import gym_cfg as gym_cfg_submission
    import agent_PG as agent_submission
    # import agent_DQN_pt as agent_submission

    gym_cfg_instance = gym_cfg_submission.gym_cfg()

    return agent_submission.agent_specs, gym_cfg_instance


def read_config(cfg_file):
    configs = {}
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if (len(line) == 3 and line[0][0] != '#'):
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
                        'lanes': []
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
        if (line[0] == 'for'):
            vehicle_id = int(line[2])
            now_dict = {
                'distance': float(lines[i + 1][2]),
                'drivable': int(float(lines[i + 2][2])),
                'road': int(float(lines[i + 3][2])),
                'route': list(map(int, list(map(float, lines[i + 4][2:])))),
                'speed': float(lines[i + 5][2]),
                'start_time': float(lines[i + 6][2]),
                't_ff': float(lines[i + 7][2]),
                ##############
                'step': int(lines[i + 8][2])
            }
            step = now_dict['step']
            ##################
            vehicles[vehicle_id] = now_dict
            tt = step - now_dict['start_time']
            tt_ff = now_dict['t_ff']
            tt_f_r = 0.0
            current_road_pos = 0
            for pos in range(len(now_dict['route'])):
                if (now_dict['road'] == now_dict['route'][pos]):
                    current_road_pos = pos
            for pos in range(len(now_dict['route'])):
                road_id = now_dict['route'][pos]
                if (pos == current_road_pos):
                    tt_f_r += (roads[road_id]['length'] -
                               now_dict['distance']) / roads[road_id]['speed_limit']
                elif (pos > current_road_pos):
                    tt_f_r += roads[road_id]['length'] / roads[road_id]['speed_limit']
            vehicles[vehicle_id]['tt_f_r'] = tt_f_r
            vehicles[vehicle_id]['delay_index'] = (tt + tt_f_r) / tt_ff

    vehicle_list = list(vehicles.keys())
    delay_index_list = []
    for vehicle_id, dict in vehicles.items():
        # res = max(res, dict['delay_index'])
        if ('delay_index' in dict.keys()):
            delay_index_list.append(dict['delay_index'])

    # 'delay_index_list' contains all vehicles' delayindex at this snapshot.
    # 'vehicle_list' contains the vehicle_id at this snapshot.
    # 'vehicles' is a dict contains vehicle infomation at this snapshot
    return delay_index_list, vehicle_list, vehicles


def process_score(log_path, roads, step, scores_dir):
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
        with open(scores_dir / 'scores {}.json'.format(step), 'w') as f_out:
            json.dump(result_write, f_out, indent=2)

    return result_write['data']['total_served_vehicles'], result_write['data']['delay_index']


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
        "--sim_cfg",
        help='The path to the simulator cfg',
        default='cfg/simulator.cfg',
        type=str
    )

    parser.add_argument(
        "--metric_period",
        help="period of scoring",
        default=200,
        type=int
    )
    parser.add_argument(
        "--threshold",
        help="period of scoring",
        default=1.6,
        type=float
    )

    parser.add_argument('--thread', type=int, default=8, help='number of threads')
    parser.add_argument('--steps', type=int, default=360, help='number of steps')
    parser.add_argument('--action_interval', type=int, default=2, help='how often agent make decisions')
    parser.add_argument('--episodes', type=int, default=100, help='training episodes')

    parser.add_argument('--save_model', action="store_true", default=False)
    parser.add_argument('--load_model', action="store_true", default=False)
    parser.add_argument("--save_rate", type=int, default=5,
                        help="save model once every time this many episodes are completed")
    parser.add_argument('--save_dir', type=str, default="model/PG",
                        help='directory in which model should be saved')
    parser.add_argument('--log_dir', type=str, default="cmd_log/PG",
                        help='directory in which logs should be saved')

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
        json.dump(result, open(scores_dir / "scores.json", 'w'), indent=2)
        raise AssertionError()

    # get agent and configuration of gym
    try:
        agent_spec, gym_cfg = load_agent_submission(submission_dir)
    except Exception as e:
        msg = format_exception(e)
        result['error_msg'] = msg
        json.dump(result, open(scores_dir / "scores.json", 'w'), indent=2)
        raise AssertionError()

    logger.info(f"Loaded user agent instance={agent_spec}")

    logger.info("\n")
    logger.info("*" * 40)

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

    done = False

    roadnet_path = Path(simulator_configs['road_file_addr'])

    intersections, roads, agents = process_roadnet(roadnet_path)

    for agent_id in agents.keys():
        if -1 in agents[agent_id]:
            # print(f"agent id {agent_id}, roads: {agents[agent_id]}")
            continue

        if np.random.random() < 0.05:
            # check if the in_road and out_road are consistent in lengths
            in_road_ids = agents[agent_id][:4]
            out_road_ids = agents[agent_id][4:]
            print("lengths of in roads:", end=': ')
            for id in in_road_ids:
                print(roads[id]['length'], end='; ')
            print()
            print("lengths of out roads:", end=': ')
            for id in in_road_ids:
                print(roads[id]['length'], end='; ')
            print()

    observations, infos = env.reset()