#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import math

import cv2
import cv2 as cv
import numpy as np
import pygame
import random
import time
from skimage.transform import resize
import torch
import gym
from gym import spaces
from gym.utils import seeding

import carla
import matplotlib
import matplotlib.pyplot as plt
from elegantrl.env.render import BirdeyeRender
from elegantrl.env.route_planner import RoutePlanner
from elegantrl.env.misc import *

# left_location = [(-132, -44 + np.random.uniform(-28, 28), 90)]  朝向是右 90   速度5
# right_location = [(-128, 44 + np.random.uniform(-28, 28), -90)]  朝向是左 -90  速度-5
# ego_location = [(-156 + np.random.uniform(-14, 14), 6.4, 0.5)]
gym.logger.set_level(40)


class CarlaEnv(gym.Env):

    def __init__(self, params):
        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']  # 要绘制的过去步骤数
        self.number_of_vehicles = params['number_of_vehicles']
        self.dt = params['dt']  # 模拟时间
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt']
        self.obs_range = params['obs_range']
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range / 0.125)
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.min_speed = params['min_speed']  # 添加的 5 在前面的导入参数里面加入
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route']
        self.skip = params['frame_skip']  # 跳步
        # 自己加的
        self.reward = 0
        self.reward1 = 0
        self.reward2 = 0
        self.reward3 = 0
        self.reward4 = 0
        self.speed = 0
        self.acc = 0
        self.min_dis = 100.0

        self.dests = [[-110, 6.2], [-131.5, 20], [-121.5, -20]]  #

        # action and observation spaces
        self.discrete = params['discrete']
        self.discrete_act = [params['discrete_acc'], params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc)  # 离散空间的大小
            # self.action_space = spaces.Discrete(self.n_acc * self.n_steer)  # 离散空间的大小
            self.action_dim = self.action_space.n
        else:
            self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], ]),
                                           np.array([params['continuous_accel_range'][1], ]),
                                           dtype=np.float32)  # acc, steer
            # self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
            #                                          params['continuous_steer_range'][0]]),
            #                                np.array([params['continuous_accel_range'][1],
            #                                          params['continuous_steer_range'][1]]),
            #                                dtype=np.float32)  # acc, steer
            self.action_dim = self.action_space.shape[0]
        observation_space_dict = {
            'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)}
        self.observation_space = spaces.Dict(observation_space_dict)  # 观测

        client = carla.Client('localhost', 2000)  # 连接carla
        client.set_timeout(10.0)  # 连接时间限制10秒
        self.world = client.load_world(params['town'])  # 加载地图
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False))
        print('Carla server connected!')
        # 设置天气
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        # 删除信号灯
        for actor in self.world.get_actors().filter('*raffic*'):
            if actor.is_alive:
                actor.destroy()
        # Get spawn points 把出生点限制在十字路口内 最大是40
        spawn_point = []
        for point in self.world.get_map().get_spawn_points():
            if -165 < point.location.x < -65 and -74 < point.location.y < 73:
                spawn_point.append(point)

        # self.vehicle_spawn_points = spawn_point
        self.vehicle_spawn_points = [#(-132, -73.0, 90.4),  # 左最下方车道
                                     # (-132, -61.2, 90.4),
                                     # (-132, -49.4, 90.4),
                                     # (-132, -37.5, 90.4),
                                     # (-132, -25.8, 90.4),
                                     (-132, -14.0, 90.4),

                                     # (-128.5, -73.0, 90.4),  # 左下方第二车道 因为不会碰撞没有用
                                     (-128.5, -61.2, 90.4),
                                     # (-128.5, -49.4, 90.4),
                                     # (-128.5, -37.5, 90.4),
                                     # (-128.5, -25.8, 90.4),
                                     # (-128.5, -14.0, 90.4),

                                     (-124.5, 20.0, -90.4),  # right
                                     (-124.5, 28.2, -90.4),
                                     (-124.5, 39.4, -90.4),
                                     (-124.5, 50.5, -90.4),
                                     (-124.5, 61.8, -90.4),
                                     # (-124.5, 73.0, -90.4),

                                     (-121, 20.0, -90.4),  # right
                                     (-121, 28.2, -90.4),
                                     (-121, 39.4, -90.4),
                                     (-121, 50.5, -90.4),
                                     (-121, 61.8, -90.4),
                                     # (-121, 73.0, -90.4),
                                     ]  # 坐标增加后记得改生成车辆的参数数量
        self.ego_bp = self._create_vehicle_bluepprint('vehicle.lincoln.mkz2017', color='0,0,0')  # 黑色车

        # ego Collision sensor
        self.collision_hist = []  # The collision history 记录碰撞数据
        self.collision_hist_l = 1  # collision history length 之前是1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')  # 碰撞传感器的图纸

        # Set fixed simulation step for synchronous mode  为同步模式设置固定模拟步骤 这样才能完全控制模拟
        self.settings = self.world.get_settings()  # 世界的设置
        self.settings.fixed_delta_seconds = self.dt  # 固定时间

        # Record the time of total steps and resetting steps  记录重置步骤和总步骤的时间
        self.reset_step = 0
        self.total_step = 0
        self.collision_num = None
        self.success_num = None
        self.stop_step = 0
        self.left_lane = []
        self.right_lane = []

        # Initialize the renderer 初始化图像
        self._init_renderer()

    def reset(self):
        self.reward1 = 0
        self.reward2 = 0
        self.reward3 = 0
        self.reward4 = 0
        self.stop_step = 0
        # 传感器清空
        self.collision_sensor = None
        # 加
        # self.collision_sensor1 = None
        # self.collision_sensor2 = None
        # self.collision_sensor3 = None

        self.collision_num = 0
        self.success_num = 0
        # 删除内容
        self._clear_all_actors(['sensor.other.collision', 'sensor.camera.rgb', 'vehicle.*'])
        self._set_synchronous_mode(False)  # 销毁车辆要用异步模式 启动前要异步

        # 生成周边车辆
        random.shuffle(self.vehicle_spawn_points)  # 随机打乱数列的的顺序 生成车辆顺序位置不确定
        count = self.number_of_vehicles  # 其他车数量
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1  # 生成一辆减去

        # Get actors polygon list  获取车辆外形列表 用于绘制过去的步骤
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)

        # 生成ego车
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:  # ego生成是否超时
                print('ego spawn time')
                self.reset()
            self.start = [-159 + np.random.uniform(-5, 5), 6.2, 0.0]  # 前两个是位置 第三个是角度
            transform = set_carla_transform(self.start)  # 转换carla坐标 这里不太好
            if self._try_spawn_ego_vehicle_at(transform):  # 尝试去生成车辆，生成则结束
                break
            else:
                ego_spawn_times += 1  # 等待时间加1
                time.sleep(0.1)

        # Add collision sensor 传感器的(图纸 位置 放在车上)  传感器的位置相对与要放的车辆
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        # 新加
        # self.collision_sensor1 = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle1)
        # self.collision_sensor2 = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle2)
        # self.collision_sensor3 = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle3)

        self.collision_sensor.listen(lambda event: get_collision_hist(event))  # listen使用lambda函数输入加入碰撞记录函数
        # self.collision_sensor1.listen(lambda event: get_collision_hist(event))

        # self.collision_sensor2.listen(lambda event: get_collision_hist(event))

        # self.collision_sensor3.listen(lambda event: get_collision_hist(event))

        # 碰撞检测函数设置
        def get_collision_hist(event):
            impulse = event.normal_impulse  # 脉冲 日志
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:  # 超出范围 去掉最早的
                self.collision_hist.pop(0)

        self.collision_hist = []  # 每次刷新重置 长度大于0则有碰撞

        # Update time steps
        self.time_step = 0
        self.reset_step += 1  # 重置步数加1

        # Enable sync mode  启用同步模式
        self.settings.synchronous_mode = True  # 世界设置的同步为真
        self.world.apply_settings(self.settings)  # 应用设置

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)  # 路径规划器
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()  # 航行点，_ ,前方有无危险车辆

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)  # 初始化中先进行了图像设置

        return self._get_obs()  # 生成状态

    def step(self, action):
        total_reward = 0.0
        done = None
        obs = []
        for i in range(self.skip):
            ob, reward, done, _ = self._step(action)
            # print(self.birdeye_l, self.birdeye_w,self.state_len,'333333333333')
            ob['birdeye'] = torch.zeros((3, self.birdeye_l, self.birdeye_w)) if done else ob['birdeye']  # 防止结束是空
            ob['state'] = torch.zeros((self.state_len,)) if done else ob['state']
            if i == self.skip - 2: obs.append(ob)  # 第2步  ob是字典
            if i == self.skip - 1: obs.append(ob)  # 第4步
            total_reward += reward
            # print(ob['birdeye'].shape,done)
            if done:
                while len(obs) < 2:
                    obs.append(ob)
                break
        return obs, total_reward, done, None

    def _step(self, action):
        if self.discrete:  # 离散动作
            acc = self.discrete_act[0][action[0]]  # // self.n_steer] [0]
            # steer = self.discrete_act[1][action % self.n_steer]
            steer = 0
        else:
            acc = action[0]
            # steer = action[1]  # 有转角
            steer = 0

        # Convert acceleration to throttle and brake  转换油门和刹车
        if acc > 0:
            throttle = np.clip(acc, 0, 1)  # 剪切0-1
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc, 0, 1)

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)
        self.world.tick()  # 同步模式下模拟更新一次

        # Append actors polygon list  增加外形
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')  # 车辆外形
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:  # 超出要绘制的步数删除
            self.vehicle_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()  # 航行点  前方车辆是否危险

        # state information
        info = {'waypoints': self.waypoints,
                'vehicle_front': self.vehicle_front,
                }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return self._get_obs(), self._get_reward(acc), self._terminal(), copy.deepcopy(info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        if mode == 'rgb_array':
            return self._get_obs()

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type. 创造车辆蓝图"""
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer. 初始化鸟瞰视图渲染器。"""
        pygame.init()
        self.display = pygame.display.set_mode((self.display_size, self.display_size),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF)  # 创建窗口 3 个位置
        pixels_per_meter = self.display_size / self.obs_range  # 每米像素  256/32  标准是12
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter  # 标准150
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle}
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode. 设置同步模式
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.
        尝试随机生成车辆
        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        # vehicle.lincoln.mkz2017   vehicle.*
        blueprint = self._create_vehicle_bluepprint('vehicle.lincoln.mkz2017', number_of_wheels=number_of_wheels)  # 车辆图纸和颜色
        blueprint.set_attribute('role_name', 'autopilot')
        transform = set_carla_transform(transform)  # 修改
        vehicle = self.world.try_spawn_actor(blueprint, transform)  # 尝试生成汽车 失败返回None
        # bp = self.world.get_blueprint_library().find('vehicle.audi.tt')
        # bp.set_attribute('color', '255,0,0')
        # transform = set_carla_transform(transform)  # 修改
        # vehicle = self.world.try_spawn_actor(bp, transform)  # 尝试生成汽车 失败返回None
        # time.sleep(0.1)  # 让carla反应一下
        if vehicle is not None:
            vehicle.set_autopilot(enabled=True)
            # angle = set_speed(vehicle)  # 得到角度
            # if 75 < angle < 105:  # 朝右y正反向 90
            #     self.vehicle1 = vehicle
            # if -105 < angle < -75:  # 朝上x正反向-90
            #     self.vehicle2 = vehicle
            # if 165 < angle < 195:  # 朝上x正反向 180
            #     self.vehicle3 = vehicle
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        尝试生成ego车 先检测生成车辆时会不会碰撞其他车辆，然后再决定是否生成
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)  # 直线距离
            if dis > 8:  # 大于8产生车辆，小于不生成
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            v = vehicle.get_velocity()
            v.x = random.randint(5, 8)  # 增加初速度，解决开始车不动ego是x
            v = vehicle.set_velocity(v)
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.
        得到所有符合要求的车辆的外形

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def get_ov(self, ego, vehicle_list):
        # ---- 获取左右车道线离ego最近的车辆（考虑范围为没有超过障碍物3.8m以上的所有） ----- #
        ego_location = ego.get_location()
        nearest_car = []
        nearest_car_list = []
        v_list = []
        self.left_lane = []
        self.right_lane = []
        for target_vehicle in vehicle_list:
            if target_vehicle.id == ego.id:
                continue
            trans = target_vehicle.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw
            # 得到左右车道的车
            if ego_location.x-3.5 < x:  # ego车还没通过
                if ego_location.y < y-3.5:  # 判断左右
                    # right lane
                    if yaw < 0:  # 判断是否是通过车
                        dis_col_right = y - ego_location.y  # 右边
                        dis_col = x - ego_location.x
                        theta_dis = abs(dis_col-dis_col_right)
                        dis = math.sqrt(dis_col_right ** 2 + dis_col ** 2)
                        self.right_lane.append((target_vehicle, theta_dis,dis))
                    else:
                        # 在right lane，left lane通过的车不考虑 可以在列表删除
                        pass
                else:
                    # left lane
                    if 105 > yaw > 0:  # 判断是否是通过车 左边右转没影响
                        # 没通过时再考虑 最近一辆车
                        dis_col_left = ego_location.y - y  # 正的
                        dis_col = x - ego_location.x
                        theta_dis = abs(dis_col - dis_col_left)  # 正的
                        dis = math.sqrt(dis_col_left ** 2 + dis_col ** 2)
                        self.left_lane.append((target_vehicle, theta_dis,dis))
                    else:
                        # 在right lane，left lane通过的车不考虑 可以在列表删除
                        pass

        # 计算每辆车和ego碰撞距离排序越小越近
        self.right_lane.sort(key=lambda x: x[1])
        self.left_lane.sort(key=lambda x: x[1])

        while len(self.right_lane) < 3:  # 不够增加三辆 100为了排序
            self.right_lane.append([None, 100,100])
        while len(self.left_lane) < 3:  # 不够增加三辆 100为了排序
            self.left_lane.append([None, 100,100])

    def _get_obs(self):
        """Get the observations."""
        # Birdeye rendering  鸟瞰渲染
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.waypoints = self.waypoints  # 前方道路点传给图像绘制

        # birdeye view with roadmap and actors  路线图和参与者
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)  # 图像变成矩阵
        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)  # 矩阵变图像
        self.display.blit(birdeye_surface, (0, 0))  # 显示位置

        pygame.display.flip()  # 刷新

        # State observation
        ego_info = get_info(self.ego)  # (x, y, yaw, l, w)
        ego_x, ego_y = ego_info[0], ego_info[1]
        ego_v = get_speed(self.ego) / 3.6
        ego_yaw = ego_info[3] / 180 * np.pi
        self.acc = carla.Vehicle.get_acceleration(self.ego).x
        self.speed = ego_v
        state_list = [ego_x + 126, ego_y - 1.0, ego_yaw, ego_v, self.acc]  # 相对坐标十字路口中心-126，1.5

        vehicle_dict = self.world.get_actors().filter('vehicle.*')
        self.get_ov(self.ego, vehicle_dict)
        for ob_vehicle in self.left_lane[:2]:  # 前两个就是距离ego的前后车
            if ob_vehicle[0] is not None:
                trans = ob_vehicle[0].get_transform()
                ove_x, ove_y = trans.location.x, trans.location.y
                ove_v = get_speed(ob_vehicle[0]) / 3.6
                ove_yaw = trans.rotation.yaw / 180 * np.pi
                acc = carla.Vehicle.get_acceleration(ob_vehicle[0]).y
                state_list += [ove_x + 126, ove_y - 1.0, ob_vehicle[2], ove_v, ove_yaw, acc]
                # print(ove_x, ove_y, dis, ove_v - ego_v, ove_yaw, acc,'22222222222')
            else:
                state_list += [6.0, -40.0, 40.447, 0.0, 90.0, 0.0]

        for ob_vehicle in self.right_lane[:2]:  # 最近2辆车
            if ob_vehicle[0] is not None:
                trans = ob_vehicle[0].get_transform()
                ove_x, ove_y = trans.location.x, trans.location.y
                ove_v = get_speed(ob_vehicle[0]) / 3.6
                ove_yaw = trans.rotation.yaw / 180 * np.pi
                acc = carla.Vehicle.get_acceleration(ob_vehicle[0]).y
                # is_danger, danger_rate = danger_classes(self.ego, ob_vehicle[0]
                state_list += [ove_x + 126, ove_y - 1.0, ob_vehicle[2], ove_v, ove_yaw, acc]
                # print(ove_x, ove_y, dis, ove_v - ego_v, ove_yaw, acc,'22222222222')
            else:
                state_list += [6.0, 40.0, 40.447, 0.0, -90.0, 0.0]

        # 状态改的话，记得dan里面存储时状态结束我手动加的0的个数
        state = torch.tensor(np.array(state_list), dtype=torch.float)
        obs = {'birdeye': birdeye.astype(np.uint8)[::4, ::4, :],  # (256, 256, 3)
               'state': state, }
        self.birdeye_w, self.birdeye_l, _ = birdeye.astype(np.uint8)[::4, ::4, :].shape
        self.state_len = len(state)
        return obs

    def _get_reward(self, acc):  # 时刻和状态相关
        """Calculate the step reward."""
        # 安全
        r_safe = 0.0
        ego_x, ego_y = get_pos(self.ego)
        ego_v = get_speed(self.ego) / 3.6
        ego_a = carla.Vehicle.get_acceleration(self.ego).x
        if len(self.collision_hist) > 0:
            r_safe = -50.  # 最大300回合 要保证其他*300加起来还不多  20
        for dest in self.dests:
            if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                r_safe = 50.

        r_speed = -(ego_v - 10) ** 2 / 10 ** 2  # 缩放到-1~0 速度越小惩罚越陡

        # print(r_speed,'0000000000000')
        vehicle_dict = self.world.get_actors().filter('vehicle.*')
        r_dis = 0.0
        r_near = 0.0
        r_danger = 0.0
        r_action = 0.0
        is_vehicle = False
        is_front_vehicle = False
        is_right_vehicle = False
        is_danger = False
        danger_rate = 0.0
        max_danger_vehicle = None
        right_vehicle = []
        v_max = 10
        vehicle_dict = self.world.get_actors().filter('vehicle.*')
        is_danger = False
        self.min_dis = 100
        for target_vehicle in vehicle_dict:
            if target_vehicle.id == self.ego.id:
                continue
            trans = target_vehicle.get_transform()
            x = trans.location.x
            y = trans.location.y
            dis = math.sqrt((ego_x - x) ** 2 + (ego_y - y) ** 2)
            self.min_dis = min(self.min_dis, dis)
            if ego_v != 0:
                ego_t = (abs(ego_x) - 126) / ego_v
            else:
                ego_t = 100
            t_safe = 1.0
            t_des = 4.0
            # 针对右边车辆
            if y+3.5 > ego_y and x > ego_x and dis < 30:  # 只要右边车辆 距离ego40米的车  40
                ov_t = (y - 5) / v_max  # ov车最块到碰撞区域的时间
                t_gap = ego_t - ov_t
                if t_gap < t_safe:
                    r_action += -1
                    is_danger = True
                elif t_gap > t_des:  # 相差很远
                    r_action += 0
                else:  # 投射到-1~0
                    r_action += -np.exp(-(t_gap - t_safe) ** 2 / (2 * 1.5 ** 2)) # ((t_gap - t_safe) / (t_des - t_safe)) ** 2 - 1
            # 左边车辆
            yaw = trans.rotation.yaw
            if y < ego_y and -126 > x > ego_x and 45 > yaw > 135 and dis < 20:  # 只要右边车辆 距离ego40米的车  5-40
                ov_t = (y - 3) / v_max  # ov车最块到碰撞区域的时间
                t_gap = ego_t - ov_t
                if t_gap < t_safe:
                    r_action += -1
                    is_right_vehicle = True
                elif t_gap > t_des:  # 相差很远
                    r_action += 0
                else:  # 投射到-1~0
                    r_action += -np.exp(-(t_gap - t_safe) ** 2 / (2 * 1.5 ** 2)) # ((t_gap - t_safe) / (t_des - t_safe)) ** 2 - 1

            break_dis = ego_v ** 2 / 5 * 2  # 停止距离 自己假设减速度5米每秒
            break_x = ego_x + break_dis
            # 右边车辆的计算
            if break_x < -146:  # 有危险时 完成停止在危险区域外不惩罚
                r_dis += 0
            elif break_x > -126:  # 123
                r_dis += -1 * np.sign(float(is_danger))
            else:
                r_dis += -((break_x + 146) / 20) ** 2 * np.sign(float(is_danger))
            # 左边车辆的计算
            if break_x < -153:  # 有危险时 完成停止在危险区域外不惩罚
                r_dis += 0
            elif break_x > -143:
                r_dis += -1 * np.sign(float(is_right_vehicle))
            else:
                r_dis += -((break_x + 143) / 10) ** 2 * np.sign(float(is_right_vehicle))

        r_time = 0.0
        if self.time_step > self.max_time_episode:
            r_time = -20.0

        self.reward = r_speed
        self.reward1 = r_speed * 0.1
        self.reward2 = r_action * 0.5
        self.reward3 = r_dis*0.5  # r_dis * 0.2
        self.reward4 = r_danger * 0.1  # 不走的时候很大
        #
        r = r_safe + 0.1 * r_speed + r_dis*0.5 + r_action * 0.5 + r_time  # 距离小于30再考虑 action dis0.5

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode.
        是否结束 碰撞 超时 到达 开出车道"""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            print("----------------collision----------")
            self.collision_num += 1
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            print("reach maximum timestep")
            return True

        # If at destination  目的地
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    self.success_num += 1
                    print("----------------succession----------")
                    return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    actor.destroy()

    def pause(self):
        """pause the simulator"""
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False, synchronous_mode=True))

    def resume(self):
        """resume the simulator from pause"""
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False, synchronous_mode=False))
