import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline

"""
This document allows the clustering of data, and the display of the number of cables
"""

def _distance_to_line(point, theta, rho):

    """
    Calculates the perpendicular distance of a point to a line defined by theta and rho parameters.

    Args:
        point (tuple): (x, y) coordinates of the point.
        theta (float): Angle of the line in radians.
        rho (float): Distance from the origin to the line along the normal vector.

    Returns:
        float: The calculated distance of the point to the line.
    """

    x, y = point
    a = np.cos(theta)
    b = np.sin(theta)
    c = rho
    distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
    return distance

def _hough_transform_clustering(cluster_data):
    """
    Performs Hough transform-based clustering on a cloud of points in order to find the clusters

    Args:
        cluster_data (DataFrame): Contains 2 columns with coordinates named 'x', 'y'.
                                  Data of points in a cluster.

    Returns:
        dict: A dictionary mapping cluster indices to lists of point indices.
    """
    points = cluster_data[['x', 'y']].values

    theta_values = np.deg2rad(np.arange(-90, 90))
    rho_max = np.sqrt(np.sum(np.square(points.max(axis=0))))
    rho_values = np.linspace(-rho_max, rho_max, num=500)

    accumulator = np.zeros((len(rho_values), len(theta_values)))

    for point in points:
        x, y = point
        for theta_index, theta in enumerate(theta_values):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_index = np.argmin(np.abs(rho_values - rho))
            accumulator[rho_index, theta_index] += 1

    threshold = 0.7 * np.max(accumulator)
    significant_rho_indices, significant_theta_indices = np.where(accumulator > threshold)

    #Merge close couples
    min_distance = 0.75
    best_couples = list(zip(theta_values[significant_theta_indices], rho_values[significant_rho_indices]))
    merged_couples = []

    for theta, rho in best_couples:
        if not any(abs(theta - t) < min_distance and abs(rho - r) < min_distance for t, r in merged_couples):
            merged_couples.append((theta, rho))

    # Assign each point to the closest cluster
    point_to_cluster = {idx: [] for idx in range(len(merged_couples))}
    for point_idx, point in enumerate(points):
        distances = [_distance_to_line(point, theta, rho) for theta, rho in merged_couples]
        closest_line_idx = np.argmin(distances)
        point_to_cluster[closest_line_idx].append(cluster_data.index[point_idx])

    return point_to_cluster

def clustering(df):
    """
    Performs clustering on the given DataFrame using DBSCAN and Hough transform-based sub-clustering.

    Args:
        df (DataFrame): Input data with 'x' and 'z' columns corresponding to coordinates.

    Returns:
        DataFrame: The input data with additional 'cluster' and 'cluster_advanced' columns.
    """

    df_coordinates = df
    df_coordinates['cluster'] = -1
    df_coordinates['cluster_advanced'] = -1

    dbscan = DBSCAN(eps=0.6, min_samples=5)

    df_coordinates['cluster'] = dbscan.fit_predict(df_coordinates[['x', 'z']])

    for cluster_id in df_coordinates['cluster'].unique():
        if cluster_id == -1:
            continue

        cluster_data = df_coordinates[df_coordinates['cluster'] == cluster_id]
        point_to_cluster = _hough_transform_clustering(cluster_data)

        for line_idx, cluster_indices in point_to_cluster.items():
            cluster_id = df_coordinates.loc[cluster_indices[0], 'cluster']
            advanced_cluster_id = f'Cluster_{cluster_id}_{line_idx}'
            df_coordinates.loc[cluster_indices, 'cluster_advanced'] = advanced_cluster_id

    splines = []

    for cluster_id in df_coordinates['cluster_advanced'].unique():
        if cluster_id == -1:
            continue

        cluster_data = df_coordinates[df_coordinates['cluster_advanced'] == cluster_id]
        x_values, y_values = cluster_data['x'], cluster_data['y']

        sorted_indices = np.argsort(x_values)
        x_values = x_values.iloc[sorted_indices]
        y_values = y_values.iloc[sorted_indices]

        spline = UnivariateSpline(x_values, y_values, k=2)
        splines.append(spline)

    # We assign each point to the closest spline
    point_to_spline = {}

    for point_idx, row in df_coordinates.iterrows():
        point = np.array([row['x'], row['y']])
        min_distance = float('inf')
        closest_spline_idx = None

        for spline_idx, spline in enumerate(splines):
            y_interp = spline(point[0])
            distance = np.linalg.norm(point - np.array([point[0], y_interp]), ord=2)

            if distance < min_distance:
                min_distance = distance
                closest_spline_idx = spline_idx

        if closest_spline_idx is not None:
            if closest_spline_idx in point_to_spline:
                point_to_spline[closest_spline_idx].append(point_idx)
            else:
                point_to_spline[closest_spline_idx] = [point_idx]

    for spline_idx, point_indices in point_to_spline.items():
        advanced_cluster_id = f'AdvancedCluster_{spline_idx}'
        df_coordinates.loc[point_indices, 'cluster_advanced'] = advanced_cluster_id

    return df_coordinates

def number_of_wire(df):
    """
    Computes the total quantity of wire based on clustering results.

    Args:
        df (DataFrame): Input data with 'x' and 'z' columns corresponding to coordinates.

    Returns:
        int: The total quantity of wire estimated from the clustering.
    """
    df_coordinates = df
    df_coordinates['cluster'] = -1

    dbscan = DBSCAN(eps=0.6, min_samples=5)

    df_coordinates['cluster'] = dbscan.fit_predict(df_coordinates[['x', 'z']])

    quantity_of_wire = 0

    for cluster_id in df_coordinates['cluster'].unique():
        if cluster_id == -1:
            continue

        cluster_data = df_coordinates[df_coordinates['cluster'] == cluster_id]
        point_to_cluster = _hough_transform_clustering(cluster_data)
        quantity_of_wire += len(point_to_cluster)

    return quantity_of_wire