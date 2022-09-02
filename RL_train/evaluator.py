import os
import cv2
import time
import torch
import numpy as np


def image_process(screen1,screen2, device):
    img1 = cv2.threshold(screen1, 1, 255, cv2.THRESH_BINARY_INV)[1]  # 60  1输出彩色
    img2 = cv2.threshold(screen2, 1, 255, cv2.THRESH_BINARY_INV)[1]  # 60  1输出彩色 120*120*3
    img1_hb = np.mean(img1, axis=-1)/255  # 100*100成功吧彩色变黑白 /255为了归一化
    img2_hb = np.mean(img2, axis=-1)/255
    new_state = np.concatenate((img1_hb[..., np.newaxis], img2_hb[..., np.newaxis]), axis=2)  # 黑白组合 120*120*2
    batch_input = np.swapaxes(new_state, 0, 2)  # 2*120*120
    batch_input = np.expand_dims(batch_input, 0)  # 1*2*120*120
    return torch.from_numpy(batch_input).float().to(device)


class Evaluator:  # [ElegantRL.2021.10.13]
    def __init__(self, cwd, agent_id, eval_env, eval_gap, eval_times1, eval_times2, target_return, if_overwrite):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'

        self.cwd = cwd
        self.agent_id = agent_id
        self.eval_env = eval_env
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.if_overwrite = if_overwrite
        self.target_return = target_return

        self.r_max = -np.inf
        self.eval_time = 0
        self.used_time = 0
        self.total_step = 0
        self.start_time = time.time()
        print(f"{'#' * 80}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> (bool, bool):  # 2021-09-09
        self.total_step += steps  # update total training steps 全部时间步

        if time.time() - self.eval_time < self.eval_gap:  # 评估间隔
            if_reach_goal = False
            if_save = False
        else:
            self.eval_time = time.time()

            '''evaluate first time 得出 平均奖励，奖励方差 平均步数  步数方差'''
            rewards_steps_list = [get_episode_return_and_step(self.eval_env, act)
                                  for _ in range(self.eval_times1)]  # [(1,2,3,4),...]
            r_avg, r_std, s_avg, s_std, col_avg, success_avg, col_std, success_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            '''evaluate second time  是否达到目标奖励要求'''
            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [get_episode_return_and_step(self.eval_env, act)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std, col_avg, success_avg, col_std, success_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            '''save the policy network'''
            if_save = (r_avg > self.r_max) or (success_avg > 0.90 and col_avg < 0.1)
            if if_save:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)  更新最大奖励

                act_name = 'actor' if self.if_overwrite else f'actor.{self.r_max:08.2f}'
                act_path = f"{self.cwd}/{act_name}.pth"
                torch.save(act.state_dict(), act_path)  # save policy network in *.pth

                print(f"{'ID:'}{self.agent_id:<3}{'Step:'}{self.total_step:3.2e}"
                      f"{'max_reward:':>15}{self.r_max:5.2f}{'success_avg:':>20}{success_avg:5.2f} |")  # 保存不错的结果并打印
            # log——tuple 是loss
            self.recorder.append((self.total_step, r_avg, r_std, r_exp, *log_tuple))  # update recorder

            '''print some information to Terminal'''
            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID:'}{self.agent_id:<3}{'Step:'}{self.total_step:3.2e}{'TargetR:':>10}{self.target_return:3.2f} |"
                      f"{'avgR:':<3}{r_avg:3.2f}{'stdR:':>8}{r_std:3.1f}{'avgS:':>8}{s_avg:1.0f}{'stdS:':>8}{s_std:1.0f} |"
                      f"{'UsedTime:'} {self.used_time:<3} ########")

            print(f"{'ID:'}{self.agent_id:<3}{'Step:'}{self.total_step:3.2e}{'max_reward:':>10}{self.r_max:3.2f} |"
                  f"{'avgR:':<3}{r_avg:3.2f}{'stdR:':>8}{r_std:3.1f}{'avgS:':>8}{s_avg:1.0f}{'stdS:':>8}{s_std:1.0f} |"
                  f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
            self.draw_plot()

            # if hasattr(self.eval_env, 'curriculum_learning_for_evaluator'):
            #     self.eval_env.curriculum_learning_for_evaluator(r_avg)

        return if_reach_goal, if_save

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float32)
        r_avg, s_avg, col_avg, success_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std, col_std, success_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std, col_avg, success_avg, col_std, success_std

    def save_or_load_recoder(self, if_save):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        np.save(self.recorder_path, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)  # 绘制曲线


def get_episode_return_and_step(env, act) -> (float, int):  # [ElegantRL.2021.10.13]
    device_id = next(act.parameters()).get_device()  # net.parameters() is a python generator.
    device = torch.device('cpu' if device_id == -1 else f'cuda:{device_id}')

    episode_return = 0.0  # sum of rewards in an episode

    max_step = env.max_time_episode  # env.max_time_episode  # 300
    info = [0,0]
    # init-state
    env.reset()
    init_ob1 = env.render(mode='rgb_array')  # 获取屏幕 1 (400, 400, 3)
    init_ob2 = env.render(mode='rgb_array')  # 获取屏幕 2 看一下他们的大小
    init_ob2['birdeye'] = image_process(init_ob1['birdeye'], init_ob2['birdeye'], device)
    state = init_ob2

    def get_action(_state):
        _states1 = torch.as_tensor(_state['birdeye'], dtype=torch.float32, device=device)
        _states2 = torch.as_tensor(_state['state'], dtype=torch.float32, device=device)
        _states = {"birdeye": _states1, "state": _states2}
        actions = act(_states)
        a_int = actions.argmax(dim=1).detach().cpu().numpy()
        return a_int

    for episode_step in range(max_step):
        action = get_action(state)
        obs, reward, done, _ = env.step(action)
        episode_return += reward

        last_screen = obs[0]['birdeye']
        current_screen = obs[1]['birdeye'] if not done else None
        obs[1]['birdeye'] = torch.zeros((1, 2, 120, 120), device=device) if done else image_process(
            last_screen, current_screen, device)
        obs[1]['state'] = torch.zeros((29,)) if done else obs[1]['state']
        state = obs[1]

        if done:
            info[0] = env.collision_num
            info[1] = env.success_num
            break
    episode_return = getattr(env, 'episode_return', episode_return)

    return episode_return, episode_step, info[0], info[1]


def save_learning_curve(recorder=None, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
    if recorder is None:
        recorder = np.load(f"{cwd}/recorder.npy")

    recorder = np.array(recorder)
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return')
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()

    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, label='objA', color=color10)
    ax10.tick_params(axis='y', labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
