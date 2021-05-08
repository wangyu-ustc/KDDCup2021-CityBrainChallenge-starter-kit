dic_agent_conf={
        'att_regularization': False,
        'rularization_rate': 0.03,
        'LEARNING_RATE': 0.001,
        'SAMPLE_SIZE': 1000,
        'BATCH_SIZE': 20,
        'EPOCHS': 100,
        'UPDATE_Q_BAR_FREQ': 5,
        'UPDATE_Q_BAR_EVERY_C_ROUND': False,
        'GAMMA': 0.8,
        'MAX_MEMORY_LEN': 10000,
        'PATIENCE': 10,
        'D_DENSE': 20,
        'N_LAYER': 2,
        'EPSILON': 0.8,
        'EPSILON_DECAY': 0.95,
        'MIN_EPSILON': 0.2,
        'LOSS_FUNCTION': 'mean_squared_error',
        'SEPARATE_MEMORY': False,
        'NORMAL_FACTOR': 20,
        'TRAFFIC_FILE': 'sumo_1_3_300_connect_all.xml'}



dic_traffic_env_conf={
    'ACTION_PATTERN': 'set',
    'NUM_INTERSECTIONS': 1000,
    'TOP_K_ADJACENCY': 1000,
    'MIN_ACTION_TIME': 10,
    'YELLOW_TIME': 5,
    'ALL_RED_TIME': 0,
    'NUM_PHASES': 2,
    'NUM_LANES': 1,
    'ACTION_DIM': 2,
    'MEASURE_TIME': 10,
    'IF_GUI': False,
    'DEBUG': False,
    'INTERVAL': 1,
    'THREADNUM': 8,
    'SAVEREPLAY': True,
    'RLTRAFFICLIGHT': True,
    'DIC_FEATURE_DIM': {'D_LANE_QUEUE_LENGTH': (4,), 'D_LANE_NUM_VEHICLE': (4,), 'D_COMING_VEHICLE': (4,), 'D_LEAVING_VEHICLE': (4,), 'D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1': (4,), 'D_CUR_PHASE': (8,), 'D_NEXT_PHASE': (8,), 'D_TIME_THIS_PHASE': (1,), 'D_TERMINAL': (1,), 'D_LANE_SUM_WAITING_TIME': (4,), 'D_VEHICLE_POSITION_IMG': (4, 60), 'D_VEHICLE_SPEED_IMG': (4, 60), 'D_VEHICLE_WAITING_TIME_IMG': (4, 60), 'D_PRESSURE': (1,), 'D_ADJACENCY_MATRIX': (3,)},
    'LIST_STATE_FEATURE': ['cur_phase', 'lane_num_vehicle', 'adjacency_matrix'],
    'DIC_REWARD_INFO': {'flickering': 0, 'sum_lane_queue_length': 0, 'sum_lane_wait_time': 0, 'sum_lane_num_vehicle_left': 0, 'sum_duration_vehicle_left': 0, 'sum_num_vehicle_been_stopped_thres01': 0, 'sum_num_vehicle_been_stopped_thres1': 0, 'pressure': -0.25},
    'LANE_NUM': {'LEFT': 1, 'RIGHT': 1, 'STRAIGHT': 1},
    'PHASE': {'sumo': {0: [0, 1, 0, 1, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 1, 0, 1]}, 'anon': {1: [0, 1, 0, 1, 0, 0, 0, 0], 2: [0, 0,0, 0, 0, 1, 0, 1], 3: [1, 0, 1, 0, 0, 0, 0, 0], 4: [0, 0, 0, 0, 1, 0, 1, 0]}},
    'ONE_MODEL': False,
    'NUM_AGENTS': 1,
    'SIMULATOR_TYPE': 'sumo',
    'BINARY_PHASE_EXPANSION': True,
    'NUM_ROW': 3,
    'NUM_COL': 1,
    'TRAFFIC_FILE': 'sumo_1_3_300_connect_all.xml',
    'ROADNET_FILE': 'roadnet_1_3.json'}


dic_path = {
    "PATH_TO_MODEL": "./"
}