import os
import torch
import datetime
import numpy as np


class Arguments:  # [ElegantRL.2021.10.21]
    def __init__(self, env, agent):
        self.env = env  # the environment for training
        self.max_step = getattr(env, 'max_time_episode', None)  # the max step of an episode
        self.state_dim = getattr(env, 'state_dim', None)  # vector dimension (feature number) of state
        self.action_dim = getattr(env, 'action_dim', None)  # vector dimension (feature number) of action
        self.target_return = None  # target average episode return
        # self.n_state = len(init_state['state'])

        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.net_dim = 2 ** 7  # the network width 256
        self.max_memo = 2 ** 21  # capacity of replay buffer
        self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
        self.target_step = 2 ** 10  # repeatedly update network to keep critic's loss small
        self.repeat_times = 2 ** 0  # collect target_step, then update network
        self.if_per_or_gae = False  # use PER (Prioritized Experience Replay) for sparse reward


        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        '''Arguments for device'''
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.gpu_id = 1  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        # self.workers_gpus = self.learner_gpus  # for GPU_VectorEnv (such as isaac gym)
        # self.ensemble_gpus = None  # for example: (learner_gpus0, ...)
        # self.ensemble_gap = 2 ** 8

        '''Arguments for evaluate and save'''
        self.cwd = None  # the directory path to save the model
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.eval_env = env  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 8  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_gpu_id = 0  # -1 means use cpu, >=0 means use GPU, None means set as learner_gpus[0]
        self.if_overwrite = True  # Save policy networks with different episode return or overwrite

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)  # cpu设置
        torch.set_default_dtype(torch.float32)

        '''agent'''
        assert hasattr(self.agent, 'init')
        assert hasattr(self.agent, 'update_net')
        assert hasattr(self.agent, 'explore_env')

        '''auto set'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            data_time = datetime.datetime.now().strftime("%m%d-%H%M")
            self.cwd = f'./train_data/{agent_name}_{data_time}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Remove cwd: {self.cwd}")
        else:
            print(f"| Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)
