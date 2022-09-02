import gym
import cv2
import os
import numpy as np
import torch
import datetime
import matplotlib
import matplotlib.pyplot as plt
from elegantrl.agent import AgentDQN, AgentSAC, AgentDoubleDQN, AgentDuelingDQN, AgentTD3
from elegantrl.env.carla_env import CarlaEnv
# 测试100episode的得分
gym.logger.set_level(40)  # Block warning

env_params = {
    'number_of_vehicles': 12,
    'display_size': 256,  # 图像大小
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
    'discrete_acc': [1.0, 0.5, 0.0, -0.5, -1.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'town': 'Town05',  # 地图 Town03
    'task_mode': 'roundabout',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 300,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints  # 识别最远路径点数
    'obs_range': 60,  # 观测的范围 (meter)  越大挺好的
    'd_behind': 10,  # ego车后方空白距离
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 10,  # desired speed (m/s)  15
    'min_speed': 5,  # min speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': False,  # whether to render the desired route
    'frame_skip': 2,  # whether to render the desired route
}

agent = AgentDuelingDQN()

gamma = 0.98  # 奖励折扣率 0.99 0.95,0.98
reward_scale = 2 ** 0
learning_rate = 2 ** -15
gpu_id = 1


'''init: Agent'''
action_dim = 5
n_state = 29  # 统计状态数 训练的模型改不车数
a = 2
text_step = 1000

# agent_name = agent.__class__.__name__
# data_time = datetime.datetime.now().strftime("%m%d-%H%M")
# cwd = f'./test_data/{agent_name}_{data_time}'
cwd = "/home/wn/PycharmProjects/pythonProject/elegantrl/train_data/AgentDuelingDQN_0303-2311"

np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(8)  # cpu设置
torch.set_default_dtype(torch.float32)

agent.init(action_dim=action_dim, n_state=n_state, init_dim=a, hide_dim=32, gamma=gamma,
           reward_scale=reward_scale, learning_rate=learning_rate, gpu_id=gpu_id,)

agent.save_or_load_agent(cwd, if_save=False)  # 加载模型


def image_process(screen1,screen2, device):
    img1 = cv2.threshold(screen1, 1, 255, cv2.THRESH_BINARY_INV)[1]  # 60  1输出彩色
    img2 = cv2.threshold(screen2, 1, 255, cv2.THRESH_BINARY_INV)[1]  # 60  1输出彩色 120*120*3
    img1_hb = np.mean(img1, axis=-1)/255  # 100*100成功吧彩色变黑白 /255为了归一化
    img2_hb = np.mean(img2, axis=-1)/255
    new_state = np.concatenate((img1_hb[..., np.newaxis], img2_hb[..., np.newaxis]), axis=2)  # 黑白组合 120*120*2
    batch_input = np.swapaxes(new_state, 0, 2)  # 2*120*120
    batch_input = np.expand_dims(batch_input, 0)  # 1*2*120*120
    return torch.from_numpy(batch_input).float().to(device)


def get_episode_return_and_step(env, act, eval_step) -> (float, int):  # [ElegantRL.2021.10.13]
    device_id = next(act.parameters()).get_device()  # net.parameters() is a python generator.
    device = torch.device('cpu' if device_id == -1 else f'cuda:{device_id}')

    def get_action(_state):
        _states1 = torch.as_tensor(_state['birdeye'], dtype=torch.float32, device=device)
        _states2 = torch.as_tensor(_state['state'], dtype=torch.float32, device=device)
        _states = {"birdeye": _states1, "state": _states2}
        actions = act(_states)
        a_int = actions.argmax(dim=1).detach().cpu().numpy()
        return a_int

    max_step = env.max_time_episode  # env.max_time_episode  # 300
    succession = []
    collision = []
    reward_all = []
    step = 0
    episode_return = 0.0  # sum of rewards in an episode
    time_step_list = []

    while step < eval_step:
        # init-state
        env.reset()
        init_ob1 = env.render(mode='rgb_array')  # 获取屏幕 1 (400, 400, 3)
        init_ob2 = env.render(mode='rgb_array')  # 获取屏幕 2 看一下他们的大小
        init_ob2['birdeye'] = image_process(init_ob1['birdeye'], init_ob2['birdeye'], device)
        state = init_ob2
        time_step = 0
        min_dis = []
        for episode_step in range(max_step):
            action = get_action(state)
            obs, reward, done, _ = env.step(action)
            min_dis.append(env.min_dis)
            episode_return += reward

            last_screen = obs[0]['birdeye']
            current_screen = obs[1]['birdeye'] if not done else None
            obs[1]['birdeye'] = torch.zeros((1, 2, 120, 120), device=device) if done else image_process(
                last_screen, current_screen, device)
            obs[1]['state'] = torch.zeros((29,)) if done else obs[1]['state']
            state = obs[1]

            time_step += 1
            if done:
                step += 1
                time_step_list.append(time_step)
                collision.append(env.collision_num)
                succession.append(env.success_num)
                reward_all.append(episode_return)
                print('step',step,'ave_time', time_step_list[-1])
                episode_return = 0.0
                # print(reward_all, succession, collision)
                break

    return reward_all, succession, collision, min_dis, time_step_list


if "__main__" == __name__:
    for _ in range(1):
        data_rate = list()
        agent_name = agent.__class__.__name__
        data_time = datetime.datetime.now().strftime("%m%d-%H%M")
        vehicle_num = env_params['number_of_vehicles']
        cwd = f'./test_data/{agent_name}_{data_time}_{vehicle_num}'
        os.makedirs(cwd, exist_ok=True)  # 创建文件才行
        save_path = f"{cwd}/test.pth"
        eval_env = gym.make('carla-v1', params=env_params)  # 输入环境参数
        y1, y2, y3, y4, y5 = get_episode_return_and_step(eval_env, agent.act, text_step)
        data_rate.append([y1, y2, y3, y4, y5])
        np.save(save_path, data_rate)
        print('success_avg', sum(y2) / len(y2), 'collision', sum(y3) / len(y3), 'mid_dis', min(y4))
        plt.subplot(221)
        plt.plot(y1)
        plt.subplot(222)
        x1 = np.arange(2)
        y1 = [sum(y2) / len(y2), sum(y3) / len(y3)]
        plt.xticks([0, 1], ["succession rate", "collision rate"])
        plt.bar(x1[0], y1[0], width=0.1, label="succession rate", color="blue", )
        plt.bar(x1[1], y1[1], width=0.1, label="collision rate", color="red", )
        for a, b in zip(x1, y1):
            plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=10)

        plt.legend()
        plt.subplot(223)

        plt.plot(y4)
        plt.subplot(224)
        plt.plot(y5)
        # plt.show()
        plt.savefig(f"{cwd}/{'plot_learning_curve.jpg'}")
        plt.close('all')
