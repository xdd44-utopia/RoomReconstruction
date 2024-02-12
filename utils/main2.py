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
    # Selects which single-plane file to use
    pointcloud_idx = 1
    #########################################################
    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.02
    min_sample_distance = 0.8
    error_functions = [ransac_error, msac_error, mlesac_error]
    error_function_idx = 0

    voxel_size = 0.005
    #########################################################

    # Read Pointcloud
    current_path = Path(__file__).parent

    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("pointclouds/image00")) + str(pointcloud_idx) + ".pcd")
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Down-sample the loaded point cloud to reduce computation time
    #pcd_sampled = pcd.uniform_down_sample(13)
    pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Apply plane-fitting algorithm
    best_plane, best_inliers, num_iterations = fit_plane(pcd=pcd_sampled,
                                                         confidence=confidence,
                                                         inlier_threshold=inlier_threshold,
                                                         min_sample_distance=min_sample_distance,
                                                         error_func=error_functions[error_function_idx])

    # Plot the result
    print(best_inliers)
    print(len(np.asarray(pcd_sampled.points)))
    plot_dominant_plane(pcd_sampled, best_inliers, best_plane)

    
