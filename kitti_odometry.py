#!/usr/bin/env python
# -*- coding:utf-8 -*
# ############################################################
#
# Copyright (c) 2022 TF.com, Inc. All Rights Reserved
#
# ############################################################

'''
Date: 2022-01-23 13:44:03
Author: tf
LastEditTime: 2022-01-26 18:08:38
LastEditors: tf
Description: 
'''
import re
import os
import shutil
import open3d
import cv2
import math
import tqdm
import logging
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from util import log_conf

class KittiOdometry(object):
    """
    """
    def __init__(self, odometry_dir, seq_id):
        """
        init
        """
        log_conf.set_log('kitti.odometry')
        self.__odometry_dir = odometry_dir
        self.__seq_id = seq_id
        self.__bin_dir = os.path.join(self.__odometry_dir, r"data_odometry_velodyne\dataset\sequences\{}\velodyne".format(self.__seq_id))
        #self.__pcd_dir = os.path.join(self.__odometry_dir, r"data_odometry_velodyne\dataset\sequences\{}\binary_pcd".format(self.__seq_id))
        self.__pcd_dir = os.path.join(self.__odometry_dir, r"data_odometry_velodyne\dataset\sequences\{}\color_pcd".format(self.__seq_id))
        self.__img_dir = os.path.join(self.__odometry_dir, r"data_odometry_color\dataset\sequences\{}\image_2".format(self.__seq_id))
        self.__pcd_map_path = os.path.join(self.__odometry_dir, r"data_odometry_velodyne\dataset\sequences\{}\map\map.pcd".format(self.__seq_id))
        self.___global_camera_pose_path = os.path.join(self.__odometry_dir, r"data_odometry\dataset\poses\{}.txt".format(self.__seq_id))
        self.__local_lidar2camera_pose_path = os.path.join(self.__odometry_dir, r"data_odometry_calib\dataset\sequences\{}\calib.txt".format(self.__seq_id))
        self.__camera_id = 2
        self.__T_lw_mat = None
        self.__T_ic_mat = None
        self.__block_num_max = 5000
        self.__block_num_step = 1

    def gen_global_point_cloud(self):
        """
        """
        self.__get_lidar2camera_pose()
        self.__bin2pcd_for_dir(self.__bin_dir, self.__pcd_dir)
        self.__point_cloud_map(pcd_dir=self.__pcd_dir)

    def __bin2pcd_for_dir(self, bin_dir, pcd_dir):
        """
        """
        bin_paths = self.__matched_files(bin_dir, r'\d+\.bin')
        for bin_id, bin_path in tqdm.tqdm(enumerate(bin_paths)):
            bin_id = int(os.path.splitext(os.path.basename(bin_path))[0])
            if bin_id > self.__block_num_max:
                break
            if bin_id % self.__block_num_step != 0:
                continue
            pcd_path = bin_path.replace(bin_dir, pcd_dir)
            pcd_path = pcd_path.replace('.bin', '.pcd')
            pcd_pr_dir = os.path.dirname(pcd_path)
            if not os.path.exists(pcd_pr_dir):
                os.makedirs(pcd_pr_dir)
            logging.info('[bin]: {} --> [pcd]: {}'.format(bin_path, pcd_path))
            #self.__bin2pcd_binary_for_path(bin_path=bin_path, pcd_path=pcd_path)
            self.__render_pcd2image(bin_path=bin_path, pcd_path=pcd_path)

    def __render_pcd2image(self, bin_path, pcd_path):
        """
        """
        if os.path.exists(pcd_path):
            return True
        
        img_path = os.path.join(self.__img_dir, os.path.basename(bin_path)).replace('.bin', '.png')
        if not os.path.exists(bin_path) or not os.path.exists(img_path):
            return False
            
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        row, col, d = img.shape
        data = np.fromfile(bin_path, dtype=np.float32)
        D = 4
        N = int(data.shape[0] / 4)
        points = np.reshape(data, (N, D))
        pt3d_xyz = []
        pt3d_rgb = []
        for idx, pt in enumerate(points):
            if pt[0] < 0:
                continue
            if math.fabs(pt[0]) > 20 or math.fabs(pt[1]) > 20:
                continue
            pt_uv1 = np.dot(np.dot(self.__T_ic_mat, self.__T_lw_mat), np.array([pt[0], pt[1], pt[2], 1.0]).T)
            pt_uv1 /= pt_uv1[-1]
            u = int(pt_uv1[0])
            v = int(pt_uv1[1])            
            if u < 0 or u > col - 1 or v < 0 or v > row - 1:
                continue
            pt3d_xyz.append(pt[0:3])
            pt3d_rgb.append(img[v,u]/255.0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pt3d_xyz)
        pcd.colors = o3d.utility.Vector3dVector(pt3d_rgb)
        o3d.io.write_point_cloud(pcd_path, pcd, True, True)
        return True
    
    def __bin2pcd_binary_for_path(self, bin_path, pcd_path):
        """
        """
        if os.path.exists(pcd_path):
            return True
        data = np.fromfile(bin_path, dtype=np.float32)
        D = 4
        N = int(data.shape[0] / 4)
        points = np.reshape(data, (N, D))
        ind = []
        for idx, pt in enumerate(points):
            if math.fabs(pt[0]) > 50 or math.fabs(pt[1]) > 50:
                continue
            ind.append(idx)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        pcd = pcd.select_by_index(ind, invert=False)
        o3d.io.write_point_cloud(pcd_path, pcd, True, True)
        return True

    def __load_pcd(self, pcd_path):
        """
        """
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.voxel_down_sample(voxel_size=0.025)
        return pcd

    def __point_cloud_map(self, pcd_dir):
        """
        """
        pcd_map = open3d.geometry.PointCloud()
        pcd_paths = self.__matched_files(pcd_dir, r'\d+\.pcd')
        T_wl_mats = self.__get_lidar2world_pose()
        
        for pcd_id, pcd_path in tqdm.tqdm(enumerate(pcd_paths)):
            logging.info('[render] pcd_path: {}'.format(pcd_path))
            pcd = self.__load_pcd(pcd_path=pcd_path)
            pcd_id = int(os.path.splitext(os.path.basename(pcd_path))[0])
            if pcd_id > self.__block_num_max:
                break
            if pcd_id % self.__block_num_step != 0:
                continue
            t_wl = T_wl_mats[pcd_id]
            pcd.transform(t_wl)
            if pcd is None:
                continue
            pcd_map += pcd
        
        down_pcd_map = pcd_map.voxel_down_sample(voxel_size=0.05)
        # down_pcd_map, ind = pcd_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.02)
        pcd_map_dir = os.path.dirname(self.__pcd_map_path)
        if not os.path.exists(pcd_map_dir):
            os.makedirs(pcd_map_dir)
        logging.info('[write] write pcd map: {}'.format(self.__pcd_map_path))
        o3d.io.write_point_cloud(self.__pcd_map_path, down_pcd_map, True, True)
    
    def __get_lidar2camera_pose(self):
        """
        """
        logging.info('[pose]: load lidar2camera pose, path: {}'.format(self.__local_lidar2camera_pose_path))

        def get_mat44(line):
            lines = [item.strip(' ') for item in  line.split(':')]
            vec = re.split(r'\s+', lines[1].strip('\n'))
            vec = [float(item) for item in vec]
            t_wc = np.eye(4, dtype=np.float32)
            for i in range(len(vec)):
                t_wc[int(i/4), int(i%4)] = vec[i]
            return t_wc

        if not os.path.exists(self.__local_lidar2camera_pose_path):
            return False
        with open(self.__local_lidar2camera_pose_path, 'r') as f:
            lines =  f.readlines()
            camera_px_mats = [get_mat44(lines[i]) for i in range(4)]
            self.__T_lw_mat = get_mat44(lines[4])
            self.__T_ic_mat = camera_px_mats[self.__camera_id][0:3, :]
        return True

    def __get_lidar2world_pose(self):
        """
        """
        logging.info('[pose] load pose lidar->world, path: {}'.format(self.___global_camera_pose_path))
        T_wl_mats = []
        if not os.path.exists(self.___global_camera_pose_path):
            return T_wl_mats
        with open(self.___global_camera_pose_path, 'r') as f:
            for line in f.readlines():
                vec = re.split(r'\s+', line.strip('\n'))
                vec = [float(item) for item in vec]
                t_wc = np.eye(4, dtype=np.float32)
                t_cl = t_wc
                for i in range(len(vec)):
                    t_wc[int(i/4), int(i%4)] = vec[i]
                qua_c_before = Rotation.from_matrix(t_wc[0:3, 0:3]).as_quat()
                rot_c_after = Rotation.from_quat([qua_c_before[2], -qua_c_before[0], -qua_c_before[1], qua_c_before[3]]).as_matrix()
                t_cl[0:3, 0:3] = rot_c_after
                t_cl[0, 3] = vec[11] + 0.27
                t_cl[1, 3] = -vec[3]
                t_cl[2, 3] = -vec[7] -0.08
                t_wl = t_cl
                T_wl_mats.append(t_wl)

        return T_wl_mats

    def __matched_files(self, cur_dir, re_str):
        """
        """
        matched_files = []
        for root, dirs, files in os.walk(cur_dir):
            for file in files:
                if not re.search(re_str, file):
                    continue
                matched_files.append(os.path.join(root, file))
        return matched_files

if __name__ == '__main__':
    ko = KittiOdometry(odometry_dir=r'G:\BaiduNetdiskDownload\KITTI\odometry', seq_id=r'00')
    ko.gen_global_point_cloud()
