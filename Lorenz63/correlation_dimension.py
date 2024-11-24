#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def estimate_correlation_dimension(point_cloud, min_radius=1e-3, max_radius=1.0, num_radii=50):
    """
    Estimate the correlation dimension of a point cloud using pairwise distances.
    
    Parameters:
    - point_cloud: np.array of shape (n_points, n_dims) representing the point cloud.
    - min_radius: float, the minimum radius to consider.
    - max_radius: float, the maximum radius to consider.
    - num_radii: int, the number of radius values to consider between min_radius and max_radius.

    Returns:
    - radii: list of radius values used.
    - correlation_sum: list of correlation sums for each radius.
    - dimension_estimate: slope of the log-log plot (an estimate of the correlation dimension).
    """
    
    # Step 1: Calculate pairwise distances between all points
    distances = pdist(point_cloud)  # Pairwise distances (condensed form)
    
    # Step 2: Define a range of radii for correlation sum calculation
    radii = np.logspace(np.log10(min_radius), np.log10(max_radius), num=num_radii)
    correlation_sum = []

    # Step 3: Calculate correlation sum C(r) for each radius
    for r in radii:
        C_r = np.sum(distances < r) / len(distances)
        correlation_sum.append(C_r)

    # Step 4: Estimate the correlation dimension from the slope of log(C(r)) vs. log(r)
    log_radii = np.log(radii)
    log_corr_sum = np.log(correlation_sum)
    idx = np.isfinite(log_corr_sum)

    # Perform linear fit on log-log data to get the slope (dimension estimate)
    slope, intercept = np.polyfit(log_radii[idx], log_corr_sum[idx], 1)
    dimension_estimate = slope  # The slope of the log-log plot is the correlation dimension

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(log_radii, log_corr_sum, 'o', label="Log-log plot of C(r) vs r")
    plt.plot(log_radii, slope * log_radii + intercept, 'r--', label=f"Fit line: slope = {slope:.2f}")
    plt.xlabel("log(r)")
    plt.ylabel("log(C(r))")
    plt.title("Correlation Dimension Estimation")
    plt.legend()
    plt.grid(True)
    plt.show()

    return radii, correlation_sum, dimension_estimate

# Example usage
# Assuming `point_cloud` is a numpy array with shape (n_points, 3) containing the Lorenz63 data
# point_cloud = np.load('lorenz63_point_cloud.npy')  # Load or generate your data here
# radii, correlation_sum, dimension_estimate = estimate_correlation_dimension(point_cloud)

# print(f"Estimated Correlation Dimension: {dimension_estimate:.2f}")
