#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/carla-simulator/carla>:
# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ----------------------------这里是一些计算小部件----------------------------------还有一部分没看
import math
import numpy as np
import carla
import pygame
from matplotlib.path import Path
import skimage
import random


def danger_classes(ego, target_car, time_step=50):  # 因为模拟是0.1秒这里100是10秒
    last_ego_x, last_ego_y, last_ego_v_x, last_ego_v_y, last_ego_acc_x, last_ego_acc_y = get_acc(ego)
    last_ove_x, last_ove_y, last_ove_v_x, last_ove_v_y, last_ove_acc_x, last_ove_acc_y = get_acc(target_car)

    for i in range(time_step):

        # 判断下一秒是否会碰撞
        e_x, e_y, e_v, e_a = ego_theta(last_ego_x, last_ego_y, last_ego_v_x, last_ego_acc_x)  # ego只要x方向
        o_x, o_y, o_v, o_a = ove_theta(last_ove_x, last_ove_y, last_ove_v_x, last_ove_acc_y)  # ov只要y方向
        last_ego_x, last_ego_y, last_ego_v_x, last_ego_acc_x = e_x, e_y, e_v, e_a
        last_ove_x, last_ove_y, last_ove_v_x, last_ove_acc_y = o_x, o_y, o_v, o_a

        if math.sqrt((e_x - o_x) ** 2 + (e_y - o_y) ** 2) < 6:
            return True, (time_step-i-1)/time_step  # 回立马返回退出循环 返回危险程度
    return False, 0.0

def ego_theta(x, y, v, a):  # v ,a 有方向要判断x，y
    theta_x = v * 0.1 + 0.5 * a * (0.1 ** 2)
    current_x = x + theta_x if theta_x > 0 else x
    v = v + a * 0.1
    return current_x, y, v, a


def ove_theta(x, y, v, a):
    theta_y = v * 0.1 + 0.5 * a * (0.1 ** 2)
    current_y = y + theta_y if (theta_y * v) > 0 else y  # 只要位移改变量和速度方向相同即可
    v = v + a * 0.1
    return x, current_y, v, a


def get_acc(vehicle):
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    vel_x = vehicle.get_velocity().x
    vel_y = vehicle.get_velocity().y
    acc_x = carla.Vehicle.get_acceleration(vehicle).x
    acc_y = carla.Vehicle.get_acceleration(vehicle).y
    return x, y, vel_x, vel_y, acc_x, acc_y


def set_speed(vehicle):
    # 得到角度,速度
    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
    angle = vehicle.get_transform().rotation.yaw
    v = vehicle.get_velocity()
    if -15 < angle < 15:  # 朝上x正反向 0
        v.x = random.randint(5, 8)
    if 75 < angle < 105:  # 朝上y正反向 90
        v.y = random.randint(5, 8)
    if -105 < angle < -75:  # 朝上x正反向-90
        v.y = -random.randint(5, 8)
    if 165 < angle < 195:  # 朝上x正反向 180
        v.x = -random.randint(5, 8)

    vehicle.set_velocity(v)
    return angle


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh  车速km/h
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def get_pos(vehicle):
    """
    Get the position of a vehicle
    :param vehicle: the vehicle whose position is to get
    :return: speed as a float in Kmh
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y


def get_info(vehicle):
    """
    Get the full info of a vehicle
    :param vehicle: the vehicle whose info is to get
    :return: a tuple of x, y positon, yaw angle and half length, width of the vehicle
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    yaw = trans.rotation.yaw / 180 * np.pi
    bb = vehicle.bounding_box  # 车辆外框
    l = bb.extent.x  # 车长？
    w = bb.extent.y  # 车宽？
    info = (x, y, yaw, l, w)
    return info


def get_local_pose(global_pose, ego_pose):
    """
    Transform vehicle to ego coordinate  相对ego车的位置和速度
    :param global_pose: surrounding vehicle's global pose
    :param ego_pose: ego vehicle pose
    :return: tuple of the pose of the surrounding vehicle in ego coordinate
    """
    x, y, yaw = global_pose
    ego_x, ego_y, ego_yaw = ego_pose
    R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                  [-np.sin(ego_yaw), np.cos(ego_yaw)]])
    vec_local = R.dot(np.array([x - ego_x, y - ego_y]))
    yaw_local = yaw - ego_yaw
    local_pose = (vec_local[0], vec_local[1], yaw_local)
    return local_pose


def get_pixel_info(local_info, d_behind, obs_range, image_size):
    """
    Transform local vehicle info to pixel info, with ego placed at lower center of image.
    Here the ego local coordinate is left-handed, the pixel coordinate is also left-handed,
    with its origin at the left bottom.
    :param local_info: local vehicle info in ego coordinate
    :param d_behind: distance from ego to bottom of FOV
    :param obs_range: length of edge of FOV
    :param image_size: size of edge of image
    :return: tuple of pixel level info, including (x, y, yaw, l, w) all in pixels
    """
    x, y, yaw, l, w = local_info
    x_pixel = (x + d_behind) / obs_range * image_size  # 这里乘没有括号的
    y_pixel = y / obs_range * image_size + image_size / 2
    yaw_pixel = yaw
    l_pixel = l / obs_range * image_size
    w_pixel = w / obs_range * image_size
    pixel_tuple = (x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel)
    return pixel_tuple


def get_poly_from_info(info):
    """
    Get polygon for info, which is a tuple of (x, y, yaw, l, w) in a certain coordinate
    :param info: tuple of x,y position, yaw angle, and half length and width of vehicle
    :return: a numpy array of size 4x2 of the vehicle rectangle corner points position
    """
    x, y, yaw, l, w = info
    poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()  # 转置
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])
    poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)  # np.repeat 重复
    return poly


def get_pixels_inside_vehicle(pixel_info, pixel_grid):
    """
    Get pixels inside a vehicle, given its pixel level info (x, y, yaw, l, w)
    :param pixel_info: pixel level info of the vehicle
    :param pixel_grid: pixel_grid of the image, a tall numpy array pf x, y pixels
    :return: the pixels that are inside the vehicle
    """
    poly = get_poly_from_info(pixel_info)
    p = Path(poly)  # make a polygon
    grid = p.contains_points(pixel_grid)
    isinPoly = np.where(grid == True)
    pixels = np.take(pixel_grid, isinPoly, axis=0)[0]
    return pixels


def get_lane_dis(waypoints, x, y):
    """
    Calculate distance from (x, y) to waypoints. 计算从(x，y)到航路点的距离。
    :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
    :param x: x position of vehicle
    :param y: y position of vehicle
    :return: a tuple of the distance and the closest waypoint orientation
    """
    dis_min = 1000
    waypt = waypoints[0]
    for pt in waypoints:
        d = np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
        if d < dis_min:
            dis_min = d
            waypt = pt
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))  # 直线位移
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
    cross = np.cross(w, vec / lv)
    dis = - lv * cross
    return dis, w


def get_preview_lane_dis(waypoints, x, y, idx=2):
    """
    Calculate distance from (x, y) to a certain waypoint
    :param waypoints: a list of list storing waypoints like [[x0, y0], [x1, y1], ...]
    :param x: x position of vehicle
    :param y: y position of vehicle
    :param idx: index of the waypoint to which the distance is calculated  计算距离的航路点的索引
    :return: a tuple of the distance and the waypoint orientation
    """
    waypt = waypoints[idx]
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2] / 180 * np.pi), np.sin(waypt[2] / 180 * np.pi)])
    cross = np.cross(w, vec / lv)
    dis = - lv * cross
    return dis, w


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x,
                              target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0


def compute_magnitude_angle(target_location, current_location, orientation=1):  # 修改加了默认值1
    """
    Compute relative angle and distance between a target_location and a current_location
    计算目标位置和当前位置之间的相对角度和距离

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    # 距离

    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def set_carla_transform(pose):
    """
    Get a carla transform object given pose. 设置Carla的出生点的x,y,z
    :param pose: list if size 3, indicating the wanted [x, y, yaw] of the transform
    :return: a carla transform object
    """
    transform = carla.Transform()  # 得到默认坐标(0,0,0)

    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.location.z = 0.1  # 防止撞地
    transform.rotation.yaw = pose[2]
    return transform


# 下面两个配合使用
def display_to_rgb(display, obs_size):
    """
    Transform image grabbed from pygame display to an rgb image uint8 matrix
    将从pygame显示器获取的图像转换为rgb图像uint8矩阵
    :param display: pygame display input
    :param obs_size: rgb image size
    :return: rgb image uint8 matrix
    """
    rgb = np.fliplr(np.rot90(display, 3))  # flip to regular view 翻转到常规视图
    rgb = skimage.transform.resize(rgb, (obs_size, obs_size))  # resize
    rgb = rgb * 255
    return rgb


def rgb_to_display_surface(rgb, display_size):
    """
    Generate pygame surface given an rgb image uint8 matrix
    给定rgb图像uint8矩阵，生成pygame表面
    :param rgb: rgb image uint8 matrix
    :param display_size: display size
    :return: pygame surface
    """
    surface = pygame.Surface((display_size, display_size)).convert()
    display = skimage.transform.resize(rgb, (display_size, display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface
