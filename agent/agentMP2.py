import pickle
import gym

from pathlib import Path
import pickle
import gym
import numpy as np
PHASE_NO_NEED = set([3, 6, 9, 12])
# contains all of the intersections

class TestAgent():
    def __init__(self):
        self.now_phase = {}
        self.green_sec = 20
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                                   [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                                   [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]

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
        # # print("MAX PRESSURE")
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)

    def load_roadnet(self,intersections,roads,agents):
        # in_roads = []
        # for agent, agent_roads in agents:
        #     in_roads = agent_roads[:4]
        #     now_phase = dict.fromkeys(range(9))
        #     now_phase[0] = []
        #     in_roads[0]

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
            self.agent_line_capacity[agent_name] = (temp//0.17).tolist()

    ################################


    def get_phase_pressures(self, lane_vehicle_num, lane_speed):
        pressures = []
        for i in range(8):
            in_lanes = self.phase_lane_map_in[i]
            out_lanes = self.phase_lane_map_out[i]
            pressure = 0
            for in_lane in in_lanes:
                pressure += lane_vehicle_num[in_lane] * 3

            for out_lane in out_lanes:
                out_lane_speed = lane_speed[out_lane]
                # if out_lane_speed > 5.5:
                #     continue
                # else:
                # pressure -= lane_vehicle_num[out_lane]
                pressure +=  - lane_vehicle_num[out_lane]
            pressures.append(pressure)
        # # print("pressures: ", pressures)
        return pressures

    def get_action(self, lane_vehicle_num, lane_speed):
        pressures = self.get_phase_pressures(lane_vehicle_num, lane_speed)
        # unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # for i in range(1, 25):
        #     if lane_vehicle_num[i] < 0:

        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)
        max_pressure_id = np.argmax(pressures) + 1
        # while (max_pressure_id in unavailable_phases):
        #     pressures[max_pressure_id - 1] -= 999999
        #     max_pressure_id = np.argmax(pressures) + 1
        # # print(max_pressure_id)
        return max_pressure_id

    def get_action_3inter(self,lane_vehicle_num, lane_speed):
        pressures = []
        phases = []
        for i in range(8):
            if i not in [1,4,5]:
                continue
            phases.append(i+1)
            in_lanes = self.phase_lane_map_in[i]
            out_lanes = self.phase_lane_map_out[i]
            pressure = 0
            for in_lane in in_lanes:
                pressure += lane_vehicle_num[in_lane] * 3
            for out_lane in out_lanes:
                out_lane_speed = lane_speed[out_lane]
                # if out_lane_speed > 5.5:
                #     continue
                # else:
                pressure -= lane_vehicle_num[out_lane]
            pressures.append(pressure)

        max_pressure_id = np.argmax(pressures)
        return phases[max_pressure_id]

    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                phase_id += 1
                if lane in in_lanes and phase_id not in unavailable_phases:
                    unavailable_phases.append(phase_id)

        return unavailable_phases
        # return [5, 6, 7, 8]

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
                    if speed < 5.5:
                        if lane not in lane_waiting_num:
                            lane_waiting_num[lane] = 1
                        elif lane in lane_waiting_num:
                            lane_waiting_num[lane] += 1

        return re_inroad_lane, lane_waiting_num


    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos
        observations = obs['observations']
        info = obs['info']
        actions = {}


        # preprocess observations
        # a simple fixtime agent
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val

        for agent in self.agent_list:
            # select the now_step
            for k,v in observations_for_agent[agent].items():
                now_step = v[0]
                break

            lane_vehicle_num = observations_for_agent[agent]["lane_vehicle_num"]

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

            agent_roads = self.agents[agent]  # list 长度为8 0~3 inroads 4~7 outroads 值为-1表示没有路
            inroads_id = []
            for road_id in agent_roads[0:4]:
                if road_id != -1:
                    inroads_id.append(road_id)
                else:
                    inroads_id.append(-1)

            outroads_id = []
            for road_id in agent_roads[4:]:
                if road_id != -1:
                    outroads_id.append(road_id)
                else:
                    outroads_id.append(-1)

            re_inroad_lane, in_lane_waiting_num = self.cal_waiting(
                lane_car=lane_car,
                car_speed=car_speed,
                inroads=inroads_id
            )  # waiting_num的key是lane_id

            re_outroad_lane, out_lane_waiting_num = self.cal_waiting(
                lane_car=lane_car,
                car_speed=car_speed,
                inroads=outroads_id
            )  # waiting_num的key是lane_id



            lane_speed = observations_for_agent[agent]['lane_speed']
            #lane_vehicle_num[1:] = lane_vehicle_num[1:]/self.roadlength[agent]
            # for k in range(1,25):
            #     lane_vehicle_num[k] = lane_vehicle_num[k] / (self.roadlength[agent][(k-1)//3]+0.0001) * 100
            #print("this is the size of lane vehicle num ",len(lane_vehicle_num))
            # print("agent id: ", agent)
            # print("lane vehicle: ", lane_vehicle_num)

            lane_vehicle_num_real = lane_vehicle_num
            lane_vehicle_num = [lane_vehicle_num_real[0]]


            for i in range(12):
                if i+1 not in re_inroad_lane.keys():
                    lane_vehicle_num.append(-1)
                else:
                    if re_inroad_lane[i+1] not in in_lane_waiting_num:
                        lane_vehicle_num.append(0)
                    else:
                        lane_vehicle_num.append(in_lane_waiting_num[re_inroad_lane[i+1]])

            for i in range(12):
                if i+1 not in re_outroad_lane.keys():
                    lane_vehicle_num.append(-1)
                else:
                    if re_outroad_lane[i + 1] not in out_lane_waiting_num:
                        lane_vehicle_num.append(0)
                    else:
                        lane_vehicle_num.append(out_lane_waiting_num[re_outroad_lane[i + 1]])

            if np.random.random() < 0.0001:
                print("num:", lane_vehicle_num)
                print("real num:", lane_vehicle_num_real)

            # for k in range(13, 25):
            #     lane_vehicle_num[k] -= self.roadlength[agent][(k-1)//3] // 3

            if -1 in lane_vehicle_num_real:

                idx = self.agents[agent][:4].index(-1)
                for _ in range(3 - idx):
                    lane_vehicle_num = self.clock_wise_rotate(lane_vehicle_num)

                action = self.get_action_3inter(lane_vehicle_num, lane_speed)

                for _ in range(3 - idx):
                    action = self.inverse_clockwise_mapping[action]

            else:

                action = self.get_action(lane_vehicle_num, lane_speed)

            step_diff = now_step - self.last_change_step[agent]
            if (step_diff >= self.green_sec):
                self.now_phase[agent] = action
                self.last_change_step[agent] = now_step

            actions[agent] = self.now_phase[agent]
            # print("phase: ", actions[agent])
            # print("phase available lane: ", self.phase_lane_map_in[actions[agent]-1])
            # print("________")

        return actions

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`
