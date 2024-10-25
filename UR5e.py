# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:42:54 2021

@author: mcrav
- edits made by: Olivia Gehrke
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import rosbag
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import JointState
from ag_functions import bag_to_csv, elapsed_time, total_time, butter_lowpass_filter
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

# from appleregression import AppleRegression

# beautiful plotting colors
blue = [0.267, 0.467, 0.67]
teal = [0.4, 0.8, 0.933]
green = [0.133, 0.533, 0.2]
yellow = [0.8, 0.733, 0.267]
pink = [0.933, 0.4, 0.467]
purple = [0.667, 0.2, 0.467]
grey = [0.733, 0.733, 0.733]


# plotting function workaround
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class UR5e_ros1:

    # start time is considered the beginning of the "retrieve" command
    def __init__(self, start_time,
                 name):  # wrench_csv_path, joint_csv_path, regression = True use when regression happens

        # define DH params for UR5e
        # theta0, a, d, alpha
        self.UR5e_DH = []

        j1 = [0, 0, 0.1625, np.pi / 2]
        j2 = [0, -0.425, 0, 0]
        j3 = [0, -0.3922, 0, 0]
        j4 = [0, 0, 0.1333, np.pi / 2]
        j5 = [0, 0, 0.0997, -1 * np.pi / 2]
        j6 = [0, 0, 0.09996, 0]

        self.UR5e_DH.append(j1)
        self.UR5e_DH.append(j2)
        self.UR5e_DH.append(j3)
        self.UR5e_DH.append(j4)
        self.UR5e_DH.append(j5)
        self.UR5e_DH.append(j6)

        # #list of transforms between joints
        # self.transforms = []

        # #for each frame (joint) of the robot
        # for frame in self.UR5e_DH:
        #     #calculate the transformation to the next frame in the chain
        #     Ti = self.DH_to_transform(0, frame[0], frame[1], frame[2], frame[3])
        #     #add the transformation to the list
        #     self.transforms.append(Ti)

        # forces and torques for a pick
        self.force = []
        self.torque = []

        # even more debugging
        self.wrench_times = []

        file = str(name)
        bag = rosbag.Bag('./' + file + '.bag')
        topic = 'wrench'

        for topic, msg, t in bag.read_messages(topics=topic):
            Fx = msg.wrench.force.x
            Fy = msg.wrench.force.y
            Fz = msg.wrench.force.z
            nsecs = msg.header.stamp.nsecs
            secs = msg.header.stamp.secs

            new_force = [Fx, Fy, Fz]
            new_time = [nsecs, secs]
            self.force.append(new_force)
            self.wrench_times.append(new_time)

        '''count = 0
        with open(wrench_csv_path, 'r', newline='') as csvfile:  #change to getting stuff from rosbag
            reader = csv.reader(csvfile)
            for row in reader:
                if count!=0:
                    force_list = [float(item) for item in row[5:8]]
                    self.force.append(force_list)
                    torque_list = [float(item) for item in row[8:11]]
                    self.torque.append(torque_list)
                    self.wrench_times.append(float(row[0])) #debugging
                count+=1'''

        # joint angles for a pick
        self.joint_path = []
        self.times_seconds = []
        self.times_nseconds = []

        file = str(name)
        bag = rosbag.Bag('./' + file + '.bag')
        topic2 = 'joint_states'

        for topic2, msg, t in bag.read_messages(topics=topic2):
            # order = [2, 1, 0, 3, 4, 5]
            shoulder_pan = msg.position[2]
            shoulder_lift = msg.position[1]
            elbow = msg.position[0]
            wrist1 = msg.position[3]
            wrist2 = msg.position[4]
            wrist3 = msg.position[5]
            nsec = msg.header.stamp.nsecs
            sec = msg.header.stamp.secs
            joint_state = [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
            self.joint_path.append(joint_state)
            self.times_seconds.append(sec)
            self.times_nseconds.append(nsec)

        '''count = 0
        with open(joint_csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if count!=0:
                    #row[8] = row[8].strip("[]")
                    #theta_list = [float(item) for item in row[8].split(",")]
                    theta_list = [float(item) for item in row[6:12]]
                    order = [2, 1, 0, 3, 4, 5]
                    theta_list = [theta_list[i] for i in order]
                    self.joint_path.append(theta_list)
                    self.times.append(float(row[0]))
                count+=1'''

        # for animating the robot
        # self.figure = plt.figure()
        # self.ax = self.figure.add_subplot(projection='3d')

        self.wrist_path = self.get_path()

        # debugging

        force = [np.linalg.norm(f) for f in self.force]
        # foo = plt.figure()
        # plt.plot(self.times[0:12000], force[0:12000])
        # plt.plot(start_time*np.ones(100), np.linspace(0,40,100))

        # throw out data before start time
        # retrieve_times = np.where(np.array(self.times)>=start_time)
        # print(start_time)
        # print(self.times[0])
        # print(self.times[len(self.times)-1])

        # start_idx = retrieve_times[0][0]
        # print(start_idx)

        # yayyyyyyyyy debugging :) :) :)
        # print(self.wrench_times[start_idx])
        # print(self.times[start_idx])
        # plt.figure()

        dispx = []
        dispy = []
        dispz = []
        mag = []

        for i in range(len(self.wrist_path)):
            pos = self.wrist_path[i][0:3, 3]  # wrist position - used get_path to find wrist_path
            initial = self.wrist_path[0][0:3, 3]
            disp = pos - np.array([0, 0, 0])  # fixed position - change array for point    origin: np.array([0, 0, 0])
            dispx.append(disp[0])
            dispy.append(disp[1])
            dispz.append(disp[2])
            mag.append(np.linalg.norm(disp))

        # tot_time_joint = total_time(self.times[0][:], self.times[1][:])
        # self.etime_joint = elapsed_time(tot_time_joint, tot_time_joint)

        # self.final_times = (np.array(self.times) - self.times[0])
        # self.wrench_times = (np.array(self.wrench_times) - self.wrench_times[0])

        self.x = dispx
        self.y = dispy
        self.z = dispz
        self.mag_disp = mag
        self.final_path = [dispx, dispy, dispz]
        # print(self.final_path)

        '''if regression == True:

            self.wrist_path = self.wrist_path[start_idx+100:start_idx+500]
            self.force = self.force[start_idx+100:start_idx+500]
            self.torque = self.torque[start_idx+100:start_idx+500]
            self.joint_path = self.joint_path[start_idx+100:start_idx+500]'''

        # force = [np.linalg.norm(f) for f in self.force]
        # plt.figure()
        # plt.plot(force)
        # plt.plot([start_idx, start_idx], [0, np.max(force)])
        # print(wrench_csv_path)
        # print(start_time)

        # format wrenched correctly for regression code
        # self.sensor_wrenches = np.zeros([len(self.force), 6])
        # self.sensor_wrenches[:,0:3] = self.torque
        # self.sensor_wrenches[:,3:6] = self.force

    # function to calculate the transformation to the next frame in the chain
    def get_transforms(self, theta_list):
        # list of transforms between joints
        transforms = []

        # for each frame (joint) of the robot
        i = 0
        for frame in self.UR5e_DH:
            # calculate the transformation to the next frame in the chain
            Ti = self.DH_to_transform(theta_list[i], frame[0], frame[1], frame[2], frame[3])
            # add the transformation to the list
            transforms.append(Ti)
            i += 1

        return transforms

    def DH_to_transform(self, theta, theta0, a, d, alpha):
        theta = theta + theta0

        T = np.zeros([4, 4])

        # first row
        T[0, 0] = np.cos(theta)
        T[0, 1] = -1 * np.sin(theta) * np.cos(alpha)
        T[0, 2] = np.sin(theta) * np.sin(alpha)
        T[0, 3] = np.cos(theta) * a

        # second row
        T[1, 0] = np.sin(theta)
        T[1, 1] = np.cos(theta) * np.cos(alpha)
        T[1, 2] = -1 * np.cos(theta) * np.sin(alpha)
        T[1, 3] = np.sin(theta) * a

        # third row
        T[2, 1] = np.sin(alpha)
        T[2, 2] = np.cos(alpha)
        T[2, 3] = d

        # fourth row
        T[3, 3] = 1

        return T

    def draw_robot(self, joint_angles, ax, plot_color):

        x = []
        y = []
        z = []

        # robot_state = self.get_transforms(joint_angles)

        # base_to_joint = robot_state[0]

        # x.append(base_to_joint[0,3])
        # y.append(base_to_joint[1,3])
        # z.append(base_to_joint[2,3])

        # for i in range(len(robot_state)-1):
        #     base_to_joint = np.matmul(base_to_joint, robot_state[i+1])
        #     #print(base_to_joint)
        #     x.append(base_to_joint[0,3])
        #     y.append(base_to_joint[1,3])
        #     z.append(base_to_joint[2,3])

        transforms = self.get_joint_transforms(joint_angles)
        for joint in transforms:
            x.append(joint[0, 3])
            y.append(joint[1, 3])
            z.append(joint[2, 3])

        ax.plot3D(x, y, z, '-o', color=plot_color)

        # print(x)
        # print(y)
        # print(z)

    def get_joint_transforms(self, joint_angles):

        joint_transforms = []

        robot_state = self.get_transforms(joint_angles)

        joint_transforms.append(robot_state[0])

        for i in range(len(robot_state) - 1):
            joint_transforms.append(np.matmul(joint_transforms[i], robot_state[i + 1]))

        return joint_transforms

    def get_path(self):  # this is what does the forward kinematics

        path = []

        for angles in self.joint_path:
            transforms = self.get_joint_transforms(angles)
            path.append(transforms[5])

        return path

    def show_path(self, ax, idx_range, plotcolor=teal):

        start = idx_range[0]
        stop = idx_range[1]

        x = []
        y = []
        z = []

        for transform in self.wrist_path:
            x.append(transform[0, 3])
            y.append(transform[1, 3])
            z.append(transform[2, 3])

        # #debugging code
        x = x[start:stop]
        y = y[start:stop]
        z = z[start:stop]

        ax.plot3D(x, y, z, color=plotcolor)
        # print(z[0])
        # print(z[len(z)-1])

    def show_force_dir(self, idx):

        x = [self.wrist_path[idx][0, 3]]
        y = [self.wrist_path[idx][1, 3]]
        z = [self.wrist_path[idx][2, 3]]

        # todo: rotate to world frame
        R_ws = self.wrist_path[idx][0:3, 0:3]
        F = np.matmul(R_ws, self.force[idx])

        Fhat = F / np.linalg.norm(F) / 5
        t = [Fhat[0]]
        u = [Fhat[1]]
        v = [Fhat[2]]

        self.ax.quiver(x, y, z, t, u, v)


class UR5e_ros2:

    # start time is considered the beginning of the "retrieve" command
    def __init__(self, start_time,
                 name):  # wrench_csv_path, joint_csv_path, regression = True use when regression happens

        # define DH params for UR5e
        # theta0, a, d, alpha
        self.UR5e_DH = []

        j1 = [0, 0, 0.1625, np.pi / 2]
        j2 = [0, -0.425, 0, 0]
        j3 = [0, -0.3922, 0, 0]
        j4 = [0, 0, 0.1333, np.pi / 2]
        j5 = [0, 0, 0.0997, -1 * np.pi / 2]
        j6 = [0, 0, 0.09996, 0]

        self.UR5e_DH.append(j1)
        self.UR5e_DH.append(j2)
        self.UR5e_DH.append(j3)
        self.UR5e_DH.append(j4)
        self.UR5e_DH.append(j5)
        self.UR5e_DH.append(j6)

        # #list of transforms between joints
        # self.transforms = []

        # #for each frame (joint) of the robot
        # for frame in self.UR5e_DH:
        #     #calculate the transformation to the next frame in the chain
        #     Ti = self.DH_to_transform(0, frame[0], frame[1], frame[2], frame[3])
        #     #add the transformation to the list
        #     self.transforms.append(Ti)

        # forces and torques for a pick
        self.force = []
        self.torque = []

        # even more debugging
        self.wrench_times = []

        # create reader instance and open for reading
        with Reader('./' + name) as reader:
            # iterate over messages
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/force_torque_sensor_broadcaster/wrench':
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    Fx = msg.wrench.force.x
                    Fy = msg.wrench.force.y
                    Fz = msg.wrench.force.z
                    nsecs = msg.header.stamp.nanosec
                    secs = msg.header.stamp.sec

                    new_force = [Fx, Fy, Fz]
                    new_time = [nsecs, secs]
                    self.force.append(new_force)
                    self.wrench_times.append(new_time)
        '''count = 0
        with open(wrench_csv_path, 'r', newline='') as csvfile:  #change to getting stuff from rosbag
            reader = csv.reader(csvfile)
            for row in reader:
                if count!=0:
                    force_list = [float(item) for item in row[5:8]]
                    self.force.append(force_list)
                    torque_list = [float(item) for item in row[8:11]]
                    self.torque.append(torque_list)
                    self.wrench_times.append(float(row[0])) #debugging
                count+=1'''

        # joint angles for a pick
        self.joint_path = []
        self.times_seconds = []
        self.times_nseconds = []

        with Reader('./' + name) as reader:
            # iterate over messages
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/joint_states':
                    # order = [2, 1, 0, 3, 4, 5]
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    shoulder_pan = msg.position[2]
                    shoulder_lift = msg.position[1]
                    elbow = msg.position[0]
                    wrist1 = msg.position[3]
                    wrist2 = msg.position[4]
                    wrist3 = msg.position[5]
                    nsec = msg.header.stamp.nanosec
                    sec = msg.header.stamp.sec
                    joint_state = [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
                    self.joint_path.append(joint_state)
                    self.times_seconds.append(sec)
                    self.times_nseconds.append(nsec)

        '''count = 0
        with open(joint_csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if count!=0:
                    #row[8] = row[8].strip("[]")
                    #theta_list = [float(item) for item in row[8].split(",")]
                    theta_list = [float(item) for item in row[6:12]]
                    order = [2, 1, 0, 3, 4, 5]
                    theta_list = [theta_list[i] for i in order]
                    self.joint_path.append(theta_list)
                    self.times.append(float(row[0]))
                count+=1'''

        # for animating the robot
        # self.figure = plt.figure()
        # self.ax = self.figure.add_subplot(projection='3d')

        self.wrist_path = self.get_path()

        # debugging

        force = [np.linalg.norm(f) for f in self.force]
        # foo = plt.figure()
        # plt.plot(self.times[0:12000], force[0:12000])
        # plt.plot(start_time*np.ones(100), np.linspace(0,40,100))

        # throw out data before start time
        # retrieve_times = np.where(np.array(self.times)>=start_time)
        # print(start_time)
        # print(self.times[0])
        # print(self.times[len(self.times)-1])

        # start_idx = retrieve_times[0][0]
        # print(start_idx)

        # yayyyyyyyyy debugging :) :) :)
        # print(self.wrench_times[start_idx])
        # print(self.times[start_idx])
        # plt.figure()

        dispx = []
        dispy = []
        dispz = []
        mag = []

        for i in range(len(self.wrist_path)):
            pos = self.wrist_path[i][0:3, 3]  # wrist position - used get_path to find wrist_path
            initial = self.wrist_path[0][0:3, 3]
            disp = pos - np.array([0, 0, 0])  # fixed position - change array for point    origin: np.array([0, 0, 0])
            dispx.append(disp[0])
            dispy.append(disp[1])
            dispz.append(disp[2])
            mag.append(np.linalg.norm(disp))

        # tot_time_joint = total_time(self.times[0][:], self.times[1][:])
        # self.etime_joint = elapsed_time(tot_time_joint, tot_time_joint)

        # self.final_times = (np.array(self.times) - self.times[0])
        # self.wrench_times = (np.array(self.wrench_times) - self.wrench_times[0])

        self.x = dispx
        self.y = dispy
        self.z = dispz
        self.mag_disp = mag
        self.final_path = [dispx, dispy, dispz]
        # print(self.final_path)

        '''if regression == True:

            self.wrist_path = self.wrist_path[start_idx+100:start_idx+500]
            self.force = self.force[start_idx+100:start_idx+500]
            self.torque = self.torque[start_idx+100:start_idx+500]
            self.joint_path = self.joint_path[start_idx+100:start_idx+500]'''

        # force = [np.linalg.norm(f) for f in self.force]
        # plt.figure()
        # plt.plot(force)
        # plt.plot([start_idx, start_idx], [0, np.max(force)])
        # print(wrench_csv_path)
        # print(start_time)

        # format wrenched correctly for regression code
        # self.sensor_wrenches = np.zeros([len(self.force), 6])
        # self.sensor_wrenches[:,0:3] = self.torque
        # self.sensor_wrenches[:,3:6] = self.force

    # function to calculate the transformation to the next frame in the chain
    def get_transforms(self, theta_list):
        # list of transforms between joints
        transforms = []

        # for each frame (joint) of the robot
        i = 0
        for frame in self.UR5e_DH:
            # calculate the transformation to the next frame in the chain
            Ti = self.DH_to_transform(theta_list[i], frame[0], frame[1], frame[2], frame[3])
            # add the transformation to the list
            transforms.append(Ti)
            i += 1

        return transforms

    def DH_to_transform(self, theta, theta0, a, d, alpha):
        theta = theta + theta0

        T = np.zeros([4, 4])

        # first row
        T[0, 0] = np.cos(theta)
        T[0, 1] = -1 * np.sin(theta) * np.cos(alpha)
        T[0, 2] = np.sin(theta) * np.sin(alpha)
        T[0, 3] = np.cos(theta) * a

        # second row
        T[1, 0] = np.sin(theta)
        T[1, 1] = np.cos(theta) * np.cos(alpha)
        T[1, 2] = -1 * np.cos(theta) * np.sin(alpha)
        T[1, 3] = np.sin(theta) * a

        # third row
        T[2, 1] = np.sin(alpha)
        T[2, 2] = np.cos(alpha)
        T[2, 3] = d

        # fourth row
        T[3, 3] = 1

        return T

    def draw_robot(self, joint_angles, ax, plot_color):

        x = []
        y = []
        z = []

        # robot_state = self.get_transforms(joint_angles)

        # base_to_joint = robot_state[0]

        # x.append(base_to_joint[0,3])
        # y.append(base_to_joint[1,3])
        # z.append(base_to_joint[2,3])

        # for i in range(len(robot_state)-1):
        #     base_to_joint = np.matmul(base_to_joint, robot_state[i+1])
        #     #print(base_to_joint)
        #     x.append(base_to_joint[0,3])
        #     y.append(base_to_joint[1,3])
        #     z.append(base_to_joint[2,3])

        transforms = self.get_joint_transforms(joint_angles)
        for joint in transforms:
            x.append(joint[0, 3])
            y.append(joint[1, 3])
            z.append(joint[2, 3])

        ax.plot3D(x, y, z, '-o', color=plot_color)

        # print(x)
        # print(y)
        # print(z)

    def get_joint_transforms(self, joint_angles):

        joint_transforms = []

        robot_state = self.get_transforms(joint_angles)

        joint_transforms.append(robot_state[0])

        for i in range(len(robot_state) - 1):
            joint_transforms.append(np.matmul(joint_transforms[i], robot_state[i + 1]))

        return joint_transforms

    def get_path(self):  # this is what does the forward kinematics

        path = []

        for angles in self.joint_path:
            transforms = self.get_joint_transforms(angles)
            path.append(transforms[5])

        return path

    def show_path(self, ax, idx_range, plotcolor=teal):

        start = idx_range[0]
        stop = idx_range[1]

        x = []
        y = []
        z = []

        for transform in self.wrist_path:
            x.append(transform[0, 3])
            y.append(transform[1, 3])
            z.append(transform[2, 3])

        # #debugging code
        # x = x[start:stop]
        # y = y[start:stop]
        # z = z[start:stop]

        ax.plot3D(x, y, z, color=plotcolor)
        # print(z[0])
        # print(z[len(z)-1])

    def show_force_dir(self, idx):

        x = [self.wrist_path[idx][0, 3]]
        y = [self.wrist_path[idx][1, 3]]
        z = [self.wrist_path[idx][2, 3]]

        # todo: rotate to world frame
        R_ws = self.wrist_path[idx][0:3, 0:3]
        F = np.matmul(R_ws, self.force[idx])

        Fhat = F / np.linalg.norm(F) / 5
        t = [Fhat[0]]
        u = [Fhat[1]]
        v = [Fhat[2]]

        self.ax.quiver(x, y, z, t, u, v)


if __name__ == "__main__":
    # ur = UR5e('G:/My Drive/IMML/Physical Modeling/Proxy Data/trial1/wrench.csv', 'G:/My Drive/IMML/Physical Modeling/Proxy Data/trial1/joint_states.csv', 1629153173924610000)
    # to_plot = [ur.joint_path[0],ur.joint_path[2000],ur.joint_path[4000],ur.joint_path[6000],ur.joint_path[8000],ur.joint_path[10000],ur.joint_path[12000]]
    # move_animation = animation.FuncAnimation(ur.figure, ur.draw_robot, frames=to_plot, interval=500)
    # move_animation.save('test.gif')

    ur = UR5e_ros1(1630802673.58087,
                   '2023111_realapple1_mode_dual_attempt_1_orientation_0_yaw_0')  # 'G:/My Drive/IMML/Physical Modeling/Proxy Data/00_Example_dataset/rench.csv','G:/My Drive/IMML/Physical Modeling/Proxy Data/00_Example_dataset/oint_states.csv', regression = False

    ur.draw_robot(ur.joint_path[0], ur.ax, plot_color=purple)
    disp_list = len(ur.z)
    numbers = np.arange(disp_list).tolist()
    # for idx in range(disp_list):
    # idx_range = numbers[idx:idx+2]
    ur.show_path(ur.ax, numbers)
    x = []
    y = []
    z = []

    for transform in ur.wrist_path:
        x.append(transform[0, 3])
        y.append(transform[1, 3])
        z.append(transform[2, 3])

    ur.ax.plot3D(x, y, z, color=teal)
    # ur.draw_robot(ur.joint_path[399], ur.ax, plot_color = pink)
    # apple = [-0.065, -0.5, 0.425]
    # apple = [-.40, .10, .423] #MEASURED WITH UR5e
    apple = [.10, -.40, .423]
    # ur.show_force_dir(0)
    # ur.show_force_dir(399)
    ur.ax.plot3D(apple[0], apple[1], apple[2], 'o', color=green)
    # ur.ax.plot3D(apple[0], apple[1], apple[2]+0.105, 'o', color = green)
    expected = ur.wrist_path[0][0:3, 2] * 0.161 + ur.wrist_path[0][0:3, 3] - np.array([0, 0, 0.105])  # 0.119
    ur.ax.plot3D(expected[0], expected[1], expected[2], 'o', color=yellow)

    # ur.draw_robot([0, 0, 0, np.pi/2, 0, 0], ur.ax, plot_color = blue)

    plt.show()

    # ur.show_path(ur.ax)
    # ur.ax.plot3D(0.5, -0.065, 0.425, 'o')
    # ur.ax.plot3D(-0.684238,-0.898033,0.76707, 'o', color = pink)
    # ur.ax.plot3D(-0.05127937, -0.1424965 ,  0.97820532, 'o')
    # ur.ax.plot3D(-1*apple[1], apple[0], apple[2], 'o')
    # ur.ax.plot3D(-6.826863333684514146e-01, -3.434496179643959368e-01, 1.175893031375349340e-01, 'o')

    # # regression = AppleRegression(ur.sensor_wrenches, ur.wrist_path, apple)
    # # predicted_params = regression.process_data()
    # # print(predicted_params)
    plt.figure()
    dispx = []
    dispy = []
    dispz = []
    for i in range(len(ur.wrist_path)):
        pos = ur.wrist_path[i][0:3, 3]
        disp = pos - np.array(apple)
        dispx.append(disp[0])
        dispy.append(disp[1])
        dispz.append(disp[2])
    plt.plot(dispx)
    plt.plot(dispy)
    plt.plot(dispz)
    # rad = [-3.22415,-1.39469,-1.90869,-1.3933,1.57149,-1.9258]
    rad = [-0.610896111, -1.658044478, -3.106610362, -2.408552786, 1.570792198, -7.42E-05]
    # rad =[-1.233890533,	-1.360742257,	-2.990157906,	-2.001360079,	1.476113081,	1.799933791]
    # ur.draw_robot(ur.joint_path[0], ur.ax, plot_color = purple)
    # ur.draw_robot(ur.joint_path[399], ur.ax, plot_color = pink)
    # set_axes_equal(ur.ax)

    plt.show()

    '''#animating the robot
    fig = plt.figure()
    gif_ax = fig.add_subplot(projection='3d')

    def animate(i):
        ur.draw_robot(ur.joint_path[i], gif_ax, purple)
        set_axes_equal(gif_ax)
        return fig


    anim = animation.FuncAnimation(fig, animate, frames = len(ur.joint_path), interval = 20)
    anim.save('gif_test.gif')'''

