import sys
sys.path.append('../')

import unittest
import pandas as pd
import numpy as np
from lidar_wire_detection_package.clustering import number_of_wire, clustering

class TestWireDetection(unittest.TestCase):

    ### number_of_wire tests

    def test_number_of_wire_easy(self):
        file_path = "files/lidar_cable_points_easy.parquet"
        df = pd.read_parquet(file_path)

        result = number_of_wire(df)

        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_number_of_wire_medium(self):
        file_path = "files/lidar_cable_points_medium.parquet"
        df = pd.read_parquet(file_path)

        result = number_of_wire(df)

        expected_result = 7
        self.assertEqual(result, expected_result)

    def test_number_of_wire_hard(self):
        file_path = "files/lidar_cable_points_hard.parquet"
        df = pd.read_parquet(file_path)

        result = number_of_wire(df)

        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_number_of_wire_extrahard(self):
        file_path = "files/lidar_cable_points_extrahard.parquet"
        df = pd.read_parquet(file_path)

        result = number_of_wire(df)

        expected_result = 3
        self.assertEqual(result, expected_result)

    ### clustering tests

    def catenary_function(self, x, y0, c, x0):
        return y0 + c * (np.cosh((x - x0) / c)-1)

    def generate_noisy_points_3d(self, cluster_id, num_points, x_range, y_range, noise_factor, x0, y0, c):
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = np.random.uniform(y_range[0], y_range[1], num_points)
        z_values = self.catenary_function(x_values, y0, c, x0) + np.random.normal(0, noise_factor, num_points)
        return pd.DataFrame({'x': x_values, 'y': y_values, 'z': z_values, 'cluster_id': cluster_id})


    def test_clustering(self):
        cluster_id_a = 'Cluster1'
        num_points = 500
        x_range = (-5, -4)
        y_range = (-5, -4)
        noise_factor = 0.001
        x0 = 6.5
        y0 = -0.1
        c = 150

        clustered_df1  = self.generate_noisy_points_3d(cluster_id_a, num_points, x_range, y_range, noise_factor, x0, y0, c)

        cluster_id_b = 'Cluster2'
        num_points = 500
        x_range = (-5, -4)
        y_range = (-5, -4)
        noise_factor = 0.001
        x0 = 10
        y0 = 200
        c = 0.4

        clustered_df2  = self.generate_noisy_points_3d(cluster_id_b, num_points, x_range, y_range, noise_factor, x0, y0, c)

        combined_df = pd.concat([clustered_df1, clustered_df2], ignore_index=True)

        clusters = clustering(combined_df)

        self.assertEqual(2, clusters['cluster_advanced'].nunique())

        cluster_a_indices = combined_df[combined_df['cluster_id'] == cluster_id_a].index
        cluster_b_indices = combined_df[combined_df['cluster_id'] == cluster_id_b].index

        for cluster in clusters['cluster_advanced'].unique():
            cluster_points = clusters[clusters['cluster_advanced'] == cluster]
            combined_points = combined_df[combined_df.index.isin(cluster_points.index)]

            if cluster in cluster_a_indices:
                self.assertTrue(combined_points['cluster_id'].eq(cluster_id_a).all())
            elif cluster in cluster_b_indices:
                self.assertTrue(combined_points['cluster_id'].eq(cluster_id_b).all())

            self.assertTrue(combined_points[['x', 'y', 'z']].equals(cluster_points[['x', 'y', 'z']]))


    def test_clustering_one_cluster(self):
        cluster_id_a = 'Cluster1'
        num_points = 500
        x_range = (-5, -4)
        y_range = (-5, -4)
        noise_factor = 0.001
        x0 = 6.5
        y0 = -0.1
        c = 150

        clustered_df1  = self.generate_noisy_points_3d(cluster_id_a, num_points, x_range, y_range, noise_factor, x0, y0, c)

        cluster_id_b = 'Cluster2'
        num_points = 500
        x_range = (-4, -3)
        y_range = (-4, -3)
        noise_factor = 0.001
        x0 = 6.5
        y0 = -0.1
        c = 150

        clustered_df2  = self.generate_noisy_points_3d(cluster_id_b, num_points, x_range, y_range, noise_factor, x0, y0, c)

        combined_df = pd.concat([clustered_df1, clustered_df2], ignore_index=True)

        clusters = clustering(combined_df)

        cluster_a_indices = combined_df[combined_df['cluster_id'] == cluster_id_a].index
        cluster_b_indices = combined_df[combined_df['cluster_id'] == cluster_id_b].index

        self.assertEqual(1, clusters['cluster_advanced'].nunique())

        for cluster in clusters['cluster_advanced'].unique():
            cluster_points = clusters[clusters['cluster_advanced'] == cluster]
            combined_points = combined_df[combined_df.index.isin(cluster_points.index)]

            if cluster in cluster_a_indices:
                self.assertTrue(combined_points['cluster_id'].eq(cluster_id_a).all())
            elif cluster in cluster_b_indices:
                self.assertTrue(combined_points['cluster_id'].eq(cluster_id_b).all())

            self.assertTrue(combined_points[['x', 'y', 'z']].equals(cluster_points[['x', 'y', 'z']]))



if __name__ == '__main__':
    unittest.main()
