# """[ElegantRL.2021.09.09](https://github.com/AI4Finance-Foundation/ElegantRL)"""
# resnet版本  ppo没有更改
import math
import os
import cv2
import datetime
from tensorboardX import SummaryWriter
import numpy as np
import numpy.random as rd
import torch
from copy import deepcopy
import torch.nn as nn
from collections import deque
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from elegantrl.net import QConv, QConvDuel, QConvTwinDuel, QConvTwin, Actor, ActorSAC, ActorPPO, Critic, CriticAdv, \
    CriticTwin


def image_process(screen1, screen2, device):
    img1 = cv2.threshold(screen1, 1, 255, cv2.THRESH_BINARY_INV)[1]  # 60  1输出彩色
    img2 = cv2.threshold(screen2, 1, 255, cv2.THRESH_BINARY_INV)[1]  # 60  1输出彩色 120*120*3
    img1_hb = np.mean(img1, axis=-1) / 255  # 100*100成功吧彩色变黑白 /255为了归一化
    img2_hb = np.mean(img2, axis=-1) / 255
    new_state = np.concatenate((img1_hb[..., np.newaxis], img2_hb[..., np.newaxis]), axis=2)  # 黑白组合 120*120*2
    batch_input = np.swapaxes(new_state, 0, 2)  # 2*120*120
    batch_input = np.expand_dims(batch_input, 0)  # 1*2*120*120
    return torch.from_numpy(batch_input).float().to(device)


class AgentBase:
    def __init__(self):
        self.gamma = None
        self.device = None
        self.get_obj_critic = None
        self.reward_scale = None

        self.episode_reward = 0
        self.episodes_time = 0
        self.episodes_speed = 0
        self.episodes_dis = 0
        self.episodes_qq = 0

        self.step = 0
        self.speed_step = 0
        self.total_step = 0
        self.if_per = False
        save_train = "/home/wn/PycharmProjects/pythonProject/carla" + "/train/" + datetime.datetime.now().strftime(
            "%m%d-%H%M")  # 图保存的位置
        self.writer = SummaryWriter(save_train)  # 记录

        self.aver_num = 100  # 碰撞多少次平均一次

        self.collision_num = deque(maxlen=self.aver_num)  # 每100次去平均
        self.success_num = deque(maxlen=self.aver_num)
        self.all_reward = deque(maxlen=self.aver_num)

        self.action_dim = None  # 为了下面

        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0
        '''attribute'''
        self.get_obj_critic = None

        self.loss = torch.nn.SmoothL1Loss(reduction='none' if self.if_per else 'mean')
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):
        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration"""
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:
        """Select continuous actions for exploration 连续
        `tensor states` states.shape==(batch_size, state_dim, )
        return `tensor actions` actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """
        action = self.act(states.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu()

    def rest_game(self, env):
        env.reset()
        init_ob1 = env.render(mode='rgb_array')  # 获取屏幕 1 (400, 400, 3)
        init_ob2 = env.render(mode='rgb_array')  # 获取屏幕 2 看一下他们的大小
        init_ob2['birdeye'] = image_process(init_ob1['birdeye'], init_ob2['birdeye'], self.device)
        state = init_ob2
        # _, _, w, l = init_ob2['birdeye'].shape  # 1,2,120,120
        # state_num = len(init_ob2['state'])
        # print(w,l,state_num)
        self.episode_reward = 0
        self.episodes_time = 0
        self.episodes_speed = 0
        self.episodes_dis = 0
        self.episodes_qq = 0
        done = False
        return state

    def explore_env(self, env, buffer, target_step):
        """探索，存储"""
        state = self.rest_game(env)
        i_episode = 0
        reward_traj = list()
        step = 0
        for n_step in range(target_step):
            step += 1
            action = self.select_action(state)
            obs, reward, done, _ = env.step(action)  # action.item()
            self.episode_reward += reward * self.reward_scale
            reward_traj.append((reward * self.reward_scale))
            # 观测新状态 这里ob是【字典1，字典2】
            last_screen = obs[0]['birdeye']
            current_screen = obs[1]['birdeye'] if not done else None
            obs[1]['birdeye'] = torch.zeros((1, 2, 120, 120), device=self.device) if done else image_process(
                last_screen, current_screen, self.device)
            obs[1]['state'] = torch.zeros((29,)) if done else obs[1]['state']
            next_state = obs[1]
            step_v = next_state['state'][3]
            buffer.append_buffer(next_state['birdeye'], next_state['state'], action, reward * self.reward_scale,
                                 0.0 if done else self.gamma)
            self.speed_step += 1
            self.writer.add_scalar('reward/speed', step_v, self.speed_step)
            # 为了记录每episode奖励
            self.episodes_time += env.reward1
            self.episodes_speed += env.reward2
            self.episodes_dis += env.reward3
            self.episodes_qq += env.reward4

            if done:
                self.total_step += 1
                self.collision_num.append(env.collision_num)
                self.success_num.append(env.success_num)
                self.all_reward.append(self.episode_reward)
                self.writer.add_scalar('train/time', step * 0.2, self.total_step)
                self.writer.add_scalar('reward/episode_action', self.episodes_time, self.total_step)
                self.writer.add_scalar('reward/episode_speed', self.episodes_speed, self.total_step)
                self.writer.add_scalar('reward/episode_dis', self.episodes_dis, self.total_step)
                self.writer.add_scalar('reward/episode_danger', self.episodes_qq, self.total_step)

                print("step", step, "total_reward", self.episode_reward, "speed_reward", self.episodes_time,
                      "action_reward", self.episodes_speed, "dis_reward", self.episodes_dis, "danger_reward",
                      self.episodes_qq)
                # 这里要和存储多少个相同
                step = 0
                if (i_episode % self.aver_num) == 0:
                    self.writer.add_scalar('ave/success', np.mean(self.success_num), self.total_step)
                    self.writer.add_scalar('ave/collision', np.mean(self.collision_num), self.total_step)
                    self.writer.add_scalar('ave/average_reward', np.mean(self.all_reward), self.total_step)

            state = self.rest_game(env) if done else next_state

        traj = torch.stack([torch.from_numpy(np.array(reward_traj))])  # 只是为了后面平均奖励
        return traj

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer
        :return float obj_a: the objective value of actor  q估计值
        :return float obj_c: the objective value of critic loss (q估计-q实际)
        """

    def optim_update(self, optimizer, objective, params):
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """save or load the training files for agent from disk.
        `str cwd` current working directory, where to save training files.
        `bool if_save` True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.ClassCri = None  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        self.if_use_dueling = True  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):

        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.cri = QConv(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(self.device)
        self.act = self.cri
        self.cri_target = deepcopy(self.cri)

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)  # 我原来是这个RMSprop 修改

        self.get_obj_critic = self.get_obj_critic_per if self.if_per else self.get_obj_critic_raw  # 修改if per

    def select_action(self, state):  # rd 改为np.random
        if np.random.rand() < self.explore_rate:
            # a_int = torch.tensor([[np.random.randint(self.action_dim)]], device=self.device, dtype=torch.long)
            a_int = rd.randint(self.action_dim, size=1)  # len(state) 我这个是字典这样会选出两个动作
        else:
            states1 = torch.as_tensor(state['birdeye'], dtype=torch.float32, device=self.device)
            states2 = torch.as_tensor(state['state'], dtype=torch.float32, device=self.device)
            states = {"birdeye": states1, "state": states2}
            actions = self.act(states)  # 这里用cri 因为act没有更新
            a_int = actions.argmax(dim=1).detach().cpu().numpy()  # 没有numpy就是张量
            # a_int = self.act(state).max(1)[1].view(1, 1)

        return a_int

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len_before_sample()
        cri_loss = q_value = None  # 为了返回
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            cri_loss, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, cri_loss, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
            self.step += 1
            self.writer.add_scalar('train/loss', cri_loss.item(), self.step)  # 记录loss

        return cri_loss.item(), q_value.mean().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2 = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q  # 256,1
        q_value = self.cri(state).gather(1, action.type(torch.long)).max(dim=1, keepdim=True)[0]  # 256,4 和存的action有关
        loss = self.loss(q_value, q_label)
        return loss, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():  # 这里多了is_weights 重要性对于buffer
            reward, mask, action, state1, state2, next_s1, next_s2, is_weights = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.type(torch.long))
        td_error = self.loss(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        loss = (td_error * is_weights).mean()
        buffer.td_error_update(td_error.detach())  # 这里记录误差
        return loss, q_value


class AgentDuelingDQN(AgentDQN):
    def __init__(self):
        super().__init__()

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):
        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.cri = QConv(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(self.device)
        self.act = self.cri
        self.cri_target = deepcopy(self.cri)

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.get_obj_critic = self.get_obj_critic_per if self.if_per else self.get_obj_critic_raw


class AgentDoubleDQN(AgentDQN):
    def __init__(self):
        super().__init__()
        self.soft_max = torch.nn.Softmax(dim=1)

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):

        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.cri = QConvTwin(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(
            self.device)
        self.act = self.cri
        self.cri_target = deepcopy(self.cri)
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

        self.act = self.cri
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.get_obj_critic = self.get_obj_critic_per if self.if_per else self.get_obj_critic_raw

    def select_action(self, state):  # for discrete action space
        states1 = torch.as_tensor(state['birdeye'], dtype=torch.float32, device=self.device)
        states2 = torch.as_tensor(state['state'], dtype=torch.float32, device=self.device)
        states = {"birdeye": states1, "state": states2}
        actions = self.act(states)

        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.soft_max(actions)
            a_int = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
        else:
            a_int = actions.argmax(dim=1)  # .detach().cpu().numpy()

        return a_int.detach().cpu()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2 = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.loss(q1, q_label) + self.loss(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2, is_weights = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q
        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        td_error = self.loss(q1, q_label) + self.loss(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1


class AgentSAC(AgentBase):
    def __init__(self):
        super().__init__()
        self.target_entropy = None
        self.alpha_log = None
        self.alpha_optimizer = None
        self.target_entropy = None  # * np.log(action_dim)

        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):

        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32, requires_grad=True,
                                      device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), learning_rate)
        self.target_entropy = np.log(action_dim)

        self.cri = CriticTwin(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(
            self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

        self.act = ActorSAC(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(self.device)
        self.act_optim = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        # if torch.cuda.device_count() > 1:
        #     print('Lets use', torch.cuda.device_count(), 'GPUs!')
        #     self.alpha_log = nn.DataParallel(self.alpha_log)
        #     self.cri = nn.DataParallel(self.cri)
        #     self.act = nn.DataParallel(self.act)
        # if isinstance(self.alpha_log,torch.nn.DataParallel):
        #     self.alpha_log = self.alpha_log.module
        # if isinstance(self.cri, torch.nn.DataParallel):
        #     self.cri = self.cri.module
        # if isinstance(self.act, torch.nn.DataParallel):
        #     self.act = self.act.module

        # self.alpha_log = self.alpha_log.to(self.device)
        # self.cri = self.cri.to(self.device)
        # self.act = self.act.to(self.device)
        # self.cri_target = deepcopy(self.cri)

        self.get_obj_critic = self.get_obj_critic_per if self.if_per else self.get_obj_critic_raw  # 修改if per

    def select_action(self, state) -> np.ndarray:
        states1 = torch.as_tensor(state['birdeye'], dtype=torch.float32, device=self.device)
        states2 = torch.as_tensor(state['state'], dtype=torch.float32, device=self.device)
        states = {"birdeye": states1, "state": states2}
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(states)
        else:
            actions = self.act(states)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len_before_sample()
        alpha = self.alpha_log.exp().detach()
        cri_loss = None
        obj_actor = None
        # self.env.pause()
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            '''objective of critic'''
            cri_loss, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = 0.995 * self.obj_critic + 0.0025 * cri_loss.item()  # 新版本多的
            self.optim_update(self.cri_optim, cri_loss, self.cri.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optimizer, obj_alpha, self.alpha_log)  # 因为self.alpha_log是张量

            '''objective of actor'''
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2).detach()
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + logprob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            self.soft_update(self.cri_target, self.cri, soft_update_tau)  # 新版本多的

            self.step += 1
            self.writer.add_scalar('train/loss', cri_loss.item(), self.step)  # 记录loss
            self.writer.add_scalar('train/loss1', self.obj_critic, self.step)  # 记录loss
            self.writer.add_scalar('train/action_pg', alpha.item(), self.step)  # 记录loss
            self.writer.add_scalar('train/obj_actor', obj_actor.item(), self.step)  # 记录loss
        # self.env.resume()
        return cri_loss, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2 = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_a, next_logprob = self.act.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, action))  # 这里还放入了动作

            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics  , action
        loss = self.loss(q1, q_label) + self.loss(q2, q_label)

        return loss, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2, is_weights = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_a, next_logprob = self.act.get_action_logprob(next_s)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)  # twin critics
        td_error = self.loss(q1, q_label) + self.loss(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentDDPG(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):
        self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.explore_noise)
        # I don't recommend to use OU-Noise
        self.cri = Critic(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

        self.act = Actor(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optim = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

        self.get_obj_critic = self.get_obj_critic_per if self.if_per else self.get_obj_critic_raw  # 修改if per

    def select_action(self, state) -> np.ndarray:
        states1 = torch.as_tensor(state['birdeye'], dtype=torch.float32, device=self.device).detach_()
        states2 = torch.as_tensor(state['state'], dtype=torch.float32, device=self.device).detach_()
        states = {"birdeye": states1, "state": states2}
        action = self.act(states)[0].cpu().numpy()
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            ou_noise = torch.as_tensor(self.ou_noise(), dtype=torch.float32, device=self.device).unsqueeze(0)
            action = (action + ou_noise).clamp(-1, 1)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len_before_sample()

        cri_loss = None
        obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            cri_loss, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, cri_loss)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor)
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
            self.step += 1
            self.writer.add_scalar('train/loss', cri_loss.item(), self.step)  # 记录loss

        return cri_loss.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2 = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q  # 256,1
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():  # 这里多了is_weights 重要性对于buffer
            reward, mask, action, state1, state2, next_s1, next_s2, is_weights = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)

        td_error = self.loss(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        loss = (td_error * is_weights).mean()
        buffer.td_error_update(td_error.detach())  # 这里记录误差
        return loss, q_value


class AgentTD3(AgentDDPG):
    def __init__(self):
        super().__init__()
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency, for soft target update

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):

        self.cri = CriticTwin(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(
            self.device)
        self.cri_target = deepcopy(self.cri)
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=learning_rate)

        self.act = Actor(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optim = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

        self.get_obj_critic = self.get_obj_critic_per if self.if_per else self.get_obj_critic_raw  # 修改if per

    def select_action(self, state) -> np.ndarray:
        states1 = torch.as_tensor(state['birdeye'], dtype=torch.float32, device=self.device).detach_()
        states2 = torch.as_tensor(state['state'], dtype=torch.float32, device=self.device).detach_()
        states = {"birdeye": states1, "state": states2}
        action = self.act(states)[0]

        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len_before_sample()

        cri_loss = None
        obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            cri_loss, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, cri_loss)

            if update_c % self.update_freq == 0:  # delay update
                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target is more stable than cri
                self.optim_update(self.act_optim, obj_actor)
                if self.if_use_cri_target:
                    self.soft_update(self.cri_target, self.cri, soft_update_tau)
                if self.if_use_act_target:
                    self.soft_update(self.act_target, self.act, soft_update_tau)
            self.step += 1
            self.writer.add_scalar('train/loss', cri_loss.item(), self.step)  # 记录loss
        return cri_loss.item() / 2, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2 = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.loss(q1, q_label) + self.loss(q2, q_label)  # twin critics
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """Prioritized Experience Replay

        Contributor: Github GyChou
        """
        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2, is_weights = buffer.sample_batch(batch_size)
            state = {"birdeye": state1, "state": state2}
            next_s = {"birdeye": next_s1, "state": next_s2}
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = ((self.loss(q1, q_label) + self.loss(q2, q_label)) * is_weights).mean()

        td_error = (q_label - torch.min(q1, q1).detach()).abs()
        buffer.td_error_update(td_error)
        return obj_critic, state


# ppo的learn没改
class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()

        self.noise = None
        self.compute_reward = None  # attribution

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def init(self, action_dim, n_state, init_dim=2, hide_dim=32, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, gpu_id=0):

        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(
            self.device)
        self.act = ActorPPO(action_dim=action_dim, state1=n_state, init_dim=init_dim, hide_dim=hide_dim).to(self.device)

        self.cri_optim = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_action(self, state) -> tuple:
        """select action for PPO

        :array state: state.shape==(state_dim, )

        :return array action: state.shape==(action_dim, )
        :return array noise: noise.shape==(action_dim, ), the noise
        """
        states1 = torch.as_tensor(state['birdeye'], dtype=torch.float32, device=self.device).detach_()
        states2 = torch.as_tensor(state['state'], dtype=torch.float32, device=self.device).detach_()
        states = {"birdeye": states1, "state": states2}
        actions, noises = self.act.get_action_noise(states)
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()  # todo remove detach()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len_before_sample()

        with torch.no_grad():
            reward, mask, action, state1, state2, next_s1, next_s2, is_weights = buffer.sample_batch(batch_size)
            buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (self.lambda_a_value / (buf_adv_v.std() + 1e-5))
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        update_times = int(buf_len / batch_size * repeat_times)
        for update_i in range(1, update_times + 1):
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            '''PPO: Surrogate objective of Trust Region'''
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item()  # logging_tuple

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum, buf_advantage = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob = self.act.compute_logprob(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.loss(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        return self.act.a_std_log.mean().item(), obj_critic.item()

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_sum = 0  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)  # 归一化
        return buf_r_sum, buf_advantage

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0  # reward sum of previous step
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        return buf_r_sum, buf_advantage
