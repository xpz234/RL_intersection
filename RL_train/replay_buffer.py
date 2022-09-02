
# 修改加了state1存储 多个输入的 11/23
import os
import torch
import random
import numpy as np
import numpy.random as rd
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class ReplayBuffer:
    def __init__(self, max_len, state_dim, state_dim1, action_dim, reward_dim=1, if_per=False, gpu_id=0, ):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim
        self.if_per = if_per
        self.tree = BinarySearchTree(max_len) if if_per else None
        # 我的状态是字典所以加两个状态

        self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.float32, device=self.device)  # 状态是多维的加*
        self.buf_state1 = torch.empty((max_len, state_dim1), dtype=torch.float32, device=self.device)
        self.buf_action = torch.empty((max_len, action_dim), dtype=torch.float32, device=self.device)
        self.buf_reward = torch.empty((max_len, reward_dim), dtype=torch.float32, device=self.device)
        self.buf_gamma = torch.empty((max_len, reward_dim), dtype=torch.float32, device=self.device)

    def append_buffer(self, state, state1, action, reward, gamma):  # 多了state1  # s,s1,a,r,g
        self.buf_state[self.next_idx] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.buf_state1[self.next_idx] = torch.as_tensor(state1, dtype=torch.float32, device=self.device)
        self.buf_action[self.next_idx] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.buf_reward[self.next_idx] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.buf_gamma[self.next_idx] = torch.as_tensor(gamma, dtype=torch.float32, device=self.device)

        if self.if_per:
            self.tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, state1, action, reward, gamma):
        size = len(gamma)
        next_idx = self.next_idx + size

        if self.tree:
            self.tree.update_ids(data_ids=np.arange(self.next_idx, next_idx) % self.max_len)

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_state1[self.next_idx:self.max_len] = state1[:self.max_len - self.next_idx]
            self.buf_action[self.next_idx:self.max_len] = action[:self.max_len - self.next_idx]
            self.buf_reward[self.next_idx:self.max_len] = reward[:self.max_len - self.next_idx]
            self.buf_gamma[self.next_idx:self.max_len] = gamma[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_state1[0:next_idx] = state1[-next_idx:]
            self.buf_action[0:next_idx] = action[-next_idx:]
            self.buf_reward[0:next_idx] = reward[-next_idx:]
            self.buf_gamma[0:next_idx] = gamma[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_state1[0:next_idx] = state1
            self.buf_action[0:next_idx] = action
            self.buf_reward[0:next_idx] = reward
            self.buf_gamma[0:next_idx] = gamma
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:  # r,g,a,s,s1,s_,s1_
        if self.if_per:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices, is_weights = self.tree.get_indices_is_weights(batch_size, beg, end)

            return (self.buf_reward[indices],
                    self.buf_gamma[indices],
                    self.buf_action[indices],
                    self.buf_state[indices],
                    self.buf_state1[indices],
                    self.buf_state[indices + 1],
                    self.buf_state1[indices + 1],
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device))
        else:
            indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device)
            return (self.buf_reward[indices],
                    self.buf_gamma[indices],
                    self.buf_action[indices],
                    self.buf_state[indices],
                    self.buf_state1[indices],
                    self.buf_state[indices + 1],
                    self.buf_state1[indices + 1],)

    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        """我们通过设置now_len=0来清空缓冲区。策略上需要在探索前清空缓冲区
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def td_error_update(self, td_error):
        self.tree.td_error_update(td_error)

    def save_or_load_history(self, cwd, if_save, buffer_id=0):  # [ElegantRL.2021.11.11]
        save_path = f"{cwd}/buffer_{buffer_id}.npz"
        if_load = None

        if if_save:
            self.update_now_len_before_sample()

            state_dim = self.buf_state.shape[1]
            state1_dim = self.buf_state1.shape[1]
            action_dim = self.buf_action.shape[1]
            reward_dim = self.buf_reward.shape[1]
            gamma_dim = self.buf_gamma.shape[1]

            buf_state_data_type = np.float16 \
                if self.buf_state.dtype in {np.float, np.float64, np.float32} \
                else np.uint8

            buf_state = np.empty((self.now_len, state_dim), dtype=buf_state_data_type)
            buf_state1 = np.empty((self.now_len, state1_dim), dtype=buf_state_data_type)
            buf_action = np.empty((self.now_len, action_dim), dtype=np.float16)
            buf_reward = np.empty((self.now_len, reward_dim), dtype=np.float16)
            buf_gamma = np.empty((self.now_len, gamma_dim), dtype=np.float16)

            temp_len = self.now_len - self.next_idx
            # could not broadcast input array from shape (0,2,120,120) into shape (0,2)
            buf_state[0:temp_len] = self.buf_state[self.next_idx:self.now_len].cpu().numpy()  #
            buf_state1[0:temp_len] = self.buf_state1[self.next_idx:self.now_len].cpu().numpy()
            buf_action[0:temp_len] = self.buf_action[self.next_idx:self.now_len].cpu().numpy()
            buf_reward[0:temp_len] = self.buf_reward[self.next_idx:self.now_len].cpu().numpy()
            buf_gamma[0:temp_len] = self.buf_gamma[self.next_idx:self.now_len].cpu().numpy()

            buf_state[temp_len:] = self.buf_state[:self.next_idx].detach().cpu().numpy()
            buf_state1[temp_len:] = self.buf_state1[:self.next_idx].cpu().numpy()
            buf_action[temp_len:] = self.buf_action[:self.next_idx].cpu().numpy()
            buf_reward[temp_len:] = self.buf_reward[:self.next_idx].cpu().numpy()
            buf_gamma[temp_len:] = self.buf_gamma[:self.next_idx].cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other1=buf_state1, buf_action=buf_action,
                                buf_reward=buf_reward, buf_gamma=buf_gamma)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict['buf_state']
            buf_state1 = buf_dict['buf_state1']
            buf_action = buf_dict['buf_action']
            buf_reward = buf_dict['buf_reward']
            buf_gamma = buf_dict['buf_gamma']

            bs = 512
            for i in range(0, buf_state.shape[0], bs):
                tmp_state = torch.as_tensor(buf_state[i:i + bs], dtype=torch.float32, device=self.device)
                tmp_state1 = torch.as_tensor(buf_state1[i:i + bs], dtype=torch.float32, device=self.device)
                tmp_action = torch.as_tensor(buf_action[i:i + bs], dtype=torch.float32, device=self.device)
                tmp_reward = torch.as_tensor(buf_reward[i:i + bs], dtype=torch.float32, device=self.device)
                tmp_gamma = torch.as_tensor(buf_gamma[i:i + bs], dtype=torch.float32, device=self.device)
                self.extend_buffer(tmp_state, tmp_state1, tmp_action, tmp_reward, tmp_gamma)

            self.update_now_len_before_sample()
            print(f"| ReplayBuffer load: {save_path}")
            if_load = True
        else:
            # print(f"| ReplayBuffer FileNotFound: {save_path}")
            if_load = False
        return if_load


class ReplayBuffer1:
    def __init__(self, max_len, state_dim, action_dim, reward_dim=1, if_per=False, if_gpu=True):
        print('单纯图片输入')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim
        self.if_per = if_per
        self.if_gpu = if_gpu
        if if_per:
            self.tree = BinarySearchTree(max_len)
        # 我的状态是字典所以加两个状态
        if self.if_gpu:
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.float32, device=self.device)  # 状态是多维的加*
            self.buf_action = torch.empty((max_len, action_dim), dtype=torch.float32, device=self.device)
            self.buf_reward = torch.empty((max_len, reward_dim), dtype=torch.float32, device=self.device)
            self.buf_gamma = torch.empty((max_len, reward_dim), dtype=torch.float32, device=self.device)
        else:
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
            self.buf_action = np.empty((max_len, action_dim), dtype=np.float32)
            self.buf_reward = np.empty((max_len, reward_dim), dtype=np.float32)
            self.buf_gamma = np.empty((max_len, reward_dim), dtype=np.float32)

    def append_buffer(self, state, state1, action, reward, gamma):  # 多了state1  # s,s1,a,r,g
        if self.if_gpu:  # 变成张量存起来
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            gamma = torch.as_tensor(gamma, dtype=torch.float32, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_action[self.next_idx] = action
        self.buf_reward[self.next_idx] = reward
        self.buf_gamma[self.next_idx] = gamma

        if self.if_per:
            self.tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def sample_batch(self, batch_size) -> tuple:  # r,g,a,s,s1,s_,s1_
        if self.if_per:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices, is_weights = self.tree.get_indices_is_weights(batch_size, beg, end)

            return (self.buf_reward[indices],
                    self.buf_gamma[indices],
                    self.buf_action[indices],
                    self.buf_state[indices],
                    self.buf_state[indices + 1],
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device))
        else:
            indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device) if self.if_gpu \
                else rd.randint(self.now_len - 1, size=batch_size)
            return (self.buf_reward[indices],
                    self.buf_gamma[indices],
                    self.buf_action[indices],
                    self.buf_state[indices],
                    self.buf_state[indices + 1],)

    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        """我们通过设置now_len=0来清空缓冲区。策略上需要在探索前清空缓冲区
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


class BinarySearchTree:
    """Binary Search Tree for PER
    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len
        self.prob_ary = np.zeros((memo_len - 1) + memo_len)  # parent_nodes_num + leaf_nodes_num
        self.max_len = len(self.prob_ary)
        self.now_len = self.memo_len - 1  # pointer
        self.indices = None
        self.depth = int(np.log2(self.max_len))

        # PER.  Prioritized Experience Replay. Section 4
        # alpha, beta = 0.7, 0.5 for rank-based variant
        # alpha, beta = 0.6, 0.4 for proportional variant
        self.per_alpha = 0.6  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)

    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1
        if self.now_len == tree_id:
            self.now_len += 1

        delta = prob - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = prob

        while tree_id != 0:  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):  # 10 is max_prob
        ids = data_ids + self.memo_len - 1
        self.now_len += (ids >= self.now_len).sum()

        upper_step = self.depth - 1
        self.prob_ary[ids] = prob  # here, ids means the indices of given children (maybe the right ones or left ones)
        p_ids = (ids - 1) // 2

        while upper_step:  # propagate the change through tree
            ids = p_ids * 2 + 1  # in this while loop, ids means the indices of the left children
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
        # because we take depth-1 upper steps, ps_tree[0] need to be updated alone

    def get_leaf_id(self, v):
        """Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx
        return min(leaf_idx, self.now_len - 2)  # leaf_idx

    def get_indices_is_weights(self, batch_size, beg, end):
        self.per_beta = min(1., self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.prob_ary[0] / batch_size)

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[beg:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return self.indices, is_weights

    def td_error_update(self, td_error):  # td_error = (q-q).detach_().abs()
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)
