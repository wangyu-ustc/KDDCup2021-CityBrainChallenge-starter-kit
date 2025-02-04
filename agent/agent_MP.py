import pickle
import gym

from pathlib import Path
import pickle
import gym
import numpy as np

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


    def get_action(self, lane_vehicle_num):
        pressures = self.get_phase_pressures(lane_vehicle_num)
        unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        # if len(unavailable_phases) > 0:
        #     # print("unavailable_phases: ", unavailable_phases)

        max_pressure_id = np.argmax(pressures) + 1
        while (max_pressure_id in unavailable_phases):
            pressures[max_pressure_id - 1] -= 999999
            max_pressure_id = np.argmax(pressures) + 1
        # # print(max_pressure_id)
        return max_pressure_id



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
            # print("agent id: ", agent)
            # print("lane vehicle: ", lane_vehicle_num)

            if -1 in lane_vehicle_num:

                idx = self.agents[agent][:4].index(-1)
                for _ in range(3 - idx):
                    lane_vehicle_num = self.clock_wise_rotate(lane_vehicle_num)

                action = self.get_action_3inter(lane_vehicle_num)

                for _ in range(3 - idx):
                    action = self.inverse_clockwise_mapping[action]

            else:
                action = self.get_action(lane_vehicle_num)

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