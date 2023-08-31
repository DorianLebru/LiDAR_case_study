import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

"""
This document provides a 2D and 3D representation of the data with the colours associated with their cluster.
"""

def graph_3d(df, cluster):

    """
    Displays a 3D plot of data with colors associated with their clusters.

    Args:
        df (DataFrame): Contains 3 columns with coordinates named 'x', 'y', and 'z'.
                        The data to be displayed.
        cluster (str): The name of the column containing cluster information, if no cluster available, use 'False'
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if cluster == False:
        ax.scatter(df['x'], df['y'], df['z'], c='b', marker='o', s=8)

    else:
        # Browse clusters and display points with different colours
        for cluster_id in df[cluster].unique():
            if cluster_id == -1:  # Points considered as noise
                continue
            cluster_data = df[df[cluster] == cluster_id]
            ax.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'], label=f'Cluster {cluster_id}', s=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if cluster != False:
        ax.legend()

    plt.show()

def graph_2d(df, cluster, axis1, axis2):
    """
    Displays a 2D plot of data with colors associated with their clusters.

    Args:
        df (DataFrame): The data to be displayed.
        cluster (str): The name of the column containing cluster information, if no cluster available, use 'False'
        axis1 (str): The name of the x-axis in the df.
        axis2 (str): The name of the y-axis in the df.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if cluster == False:
        ax.scatter(df[axis1], df[axis2], s=8)

    else:
        for cluster_id in df[cluster].unique():
            if cluster_id == -1:
                continue

            cluster_data = df[df[cluster] == cluster_id][[axis1, axis2]]

            ax.scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster_id}', s=8)

    ax.set_xlabel(axis1.upper())
    ax.set_ylabel(axis2.upper())

    ax.set_title('2D Scatter Plot')

    if cluster != False:
        ax.legend()

    plt.show()


def plot_assigned_catenaries(assigned_catenaries, df_coordinates):
    """
    Plots clusters and their assigned catenaries in 3D space.

    Args:
        assigned_catenaries (dict): A dictionary mapping cluster IDs to arrays of catenary points in 3D space.
        df_coordinates (DataFrame): Input data with 'cluster_advanced', 'x', 'y', and 'z' columns.
    """
    num_clusters = len(assigned_catenaries)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    for idx, (cluster_id, catenary_3d) in enumerate(assigned_catenaries.items()):
        cluster_data = df_coordinates[df_coordinates['cluster_advanced'] == cluster_id][['x', 'y', 'z']]
        ax.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'], color=colors[idx], alpha=0.2, s=10)

        ax.plot(catenary_3d[:, 0], catenary_3d[:, 1], catenary_3d[:, 2], color=colors[idx], label=f'Cluster {cluster_id}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Clusters and Assigned Catenaries')
    ax.legend()

    plt.tight_layout()
    plt.show()
