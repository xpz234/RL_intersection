import os
import time
import torch
import numpy as np
from elegantrl.replay_buffer import ReplayBuffer
from elegantrl.evaluator import Evaluator


def train_and_evaluate(args, learner_id=0):  # 2021.11.11
    args.init_before_training()  # necessary!

    '''init: Agent'''
    agent = args.agent
    n_state = 29  # 统计状态数
    a = 2
    print(args.action_dim)
    agent.init(action_dim=args.action_dim, n_state=n_state, init_dim=a, hide_dim=32, gamma=args.gamma,
               reward_scale=args.reward_scale, learning_rate=args.learning_rate, gpu_id=args.gpu_id, )

    agent.save_or_load_agent(args.cwd, if_save=False)

    env = args.env
    '''init Evaluator'''
    eval_env = args.eval_env
    evaluator = Evaluator(cwd=args.cwd, agent_id=0,
                          eval_env=eval_env, eval_gap=args.eval_gap,
                          eval_times1=args.eval_times1, eval_times2=args.eval_times2,
                          target_return=args.target_return, if_overwrite=args.if_overwrite)
    evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    # 这里写的是存储尺寸action_dim

    buffer = ReplayBuffer(max_len=args.max_memo, state_dim=(a, 120, 120),state_dim1=n_state,
                          action_dim=1 if env.discrete else env.action_dim, gpu_id=args.gpu_id)

    buffer.save_or_load_history(args.cwd, if_save=False)

    def update_buffer(_traj_list):  # 存储
        ten_state, ten_state1, ten_action, ten_reward, ten_gamma = _traj_list
        buffer.extend_buffer(ten_state, ten_state1, ten_action, ten_reward, ten_gamma)

        _steps, _r_exp = get_step_r_exp(ten_reward=ten_reward)  # 步数和平均数
        return _steps, _r_exp

    """start training"""
    cwd = args.cwd
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    '''init ReplayBuffer after training start'''
    # if_load = buffer.save_or_load_history(cwd, if_save=False)
    if_load = False

    if not if_load:
        traj_list = agent.explore_env(env, buffer, target_step)
        steps, r_exp = get_step_r_exp(traj_list)  # 步数和平均数
        # steps, r_exp = update_buffer(traj_list)
        evaluator.total_step += steps

    '''start training loop'''
    if_train = True
    while if_train:
        with torch.no_grad():
            traj_list = agent.explore_env(env, buffer, target_step)
            steps, r_exp = get_step_r_exp(traj_list)  # 步数和平均数
            # steps, r_exp = update_buffer(traj_list)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        with torch.no_grad():
            if_reach_goal, if_save = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)
    # buffer.save_or_load_history(cwd, if_save=True)  # 离线策略保存训练的环境状态
    evaluator.save_or_load_recoder(if_save=True)


def get_step_r_exp(ten_reward):
    return len(ten_reward), ten_reward.mean().item()
