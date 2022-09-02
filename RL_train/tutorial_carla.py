from elegantrl.run_tutorial import *
from elegantrl.config import Arguments
from elegantrl.agent import AgentDQN, AgentSAC, AgentDoubleDQN, AgentDuelingDQN, AgentTD3

from elegantrl.env.carla_env import CarlaEnv
import gym

gym.logger.set_level(40)  # Block warning

env_params = {
    'number_of_vehicles': 6,
    'display_size': 256,  # 图像大小
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [1.0, 0.5, 0.0, -0.5, -1.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'town': 'Town05',  # 地图 Town03
    'max_time_episode': 300,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints  # 识别最远路径点数
    'obs_range': 60,  # 观测的范围 (meter)  越大挺好的
    'd_behind': 10,  # ego车后方空白距离
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 10,  # desired speed (m/s)  15
    'min_speed': 5,  # min speed (m/s)
    'max_ego_spawn_times': 300,  # maximum times to spawn ego vehicle
    'display_route': False,  # whether to render the desired route
    'frame_skip': 2,  # whether to render the desired route
}

agent = AgentDoubleDQN()
# agent = AgentSAC()  # 'discrete': False
env = gym.make('carla-v1', params=env_params)  # 输入环境参数
args = Arguments(env, agent)

args.eval_times1 = 2 ** 3
args.eval_times2 = 2 ** 5
args.break_step = 50000  # 最大步
args.max_memo = 60000
args.batch_size = 128
# args.action_dim = 5
args.gamma = 0.98  # 奖励折扣率 0.99 0.95,0.98
args.target_return = 38  # 目标奖励
args.target_step = args.env.max_time_episode
args.gpu_id = 1

train_and_evaluate(args)
