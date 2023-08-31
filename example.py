from lidar_wire_detection_package.clustering import clustering, number_of_wire
from lidar_wire_detection_package.catenary import calculate_assigned_catenaries
from lidar_wire_detection_package.visualization import graph_2d, graph_3d, plot_assigned_catenaries

import pandas as pd
import matplotlib.pyplot as plt

file_path = "files/lidar_cable_points_extrahard.parquet"
df_coordinates = pd.read_parquet(file_path)

graph_3d(df_coordinates, False)
graph_2d(df_coordinates, False, 'x', 'z')

print(f'Number of wires : {number_of_wire(df_coordinates)}')

clusters = clustering(df_coordinates)
graph_3d(df_coordinates, 'cluster_advanced')

catenaries = calculate_assigned_catenaries(clusters)
plot_assigned_catenaries(catenaries, df_coordinates)