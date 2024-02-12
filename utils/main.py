#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from fit_plane import fit_plane, filter_planes
from error_funcs import ransac_error, msac_error, mlesac_error
from plot_results import *

if __name__ == '__main__':
    
    current_path = Path(__file__).parent
    # Selects which single-plane file to use
    pointcloud_idx = 1
    #########################################################
    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.02
    min_sample_distance = 0.8
    error_functions = [ransac_error, msac_error, mlesac_error]
    error_function_idx = 0
    
    #########################################################
    # Multi-Plane parameters
    multi_plane_names = ['desk', 'door', 'kitchen']
    multi_plane_idx = 0

    # RANSAC parameters:
    min_points_prop = 0.04
    confidence_multi = 0.9999
    inlier_threshold_multi = 0.03
    min_sample_distance_multi = 0.1
    error_function_idx_multi = 0

    voxel_size_multi = 0.01

    #########################################################

    # Read Pointcloud for multiple plane detection

    pcd_multi = o3d.io.read_point_cloud(
        str(current_path.joinpath("pointclouds/" + multi_plane_names[multi_plane_idx] + ".pcd")))
    if not pcd_multi.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Down-sample the loaded point cloud to reduce computation time
    pcd_multi_sampled = pcd_multi.uniform_down_sample(10)
    #pcd_multi_sampled = pcd_multi.voxel_down_sample(voxel_size=voxel_size_multi)
    
    plane_eqs, plane_pcds, filtered_pcd = filter_planes(
        pcd=pcd_multi_sampled,
        min_points_prop=min_points_prop,
        confidence=confidence_multi,
        inlier_threshold=inlier_threshold_multi,
        min_sample_distance=min_sample_distance_multi,
        error_func=error_functions[error_function_idx_multi]
    )
    
    print(plane_eqs)
    print()
    print(type(o3d.geometry.PointCloud(plane_pcds[0])))
    print()
    print(filtered_pcd)
    
    # plot_multiple_planes(
    #     plane_eqs=plane_eqs,
    #     plane_pcds=plane_pcds,
    #     filtered_pcd=filtered_pcd
    # )

    plot_dominant_plane(plane_pcds[0], [True] * 12078, plane_pcds[0])