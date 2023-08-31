import sys
sys.path.append('../')

import unittest
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from lidar_wire_detection_package.catenary import calculate_assigned_catenaries

class TestCalculateAssignedCatenaries(unittest.TestCase):

    def catenary_function(self, x, y0, c, x0):
        return y0 + c * (np.cosh((x - x0) / c)-1)

    def generate_noisy_points_3d(self, cluster_id, num_points, x_range, y_range, noise_factor, x0, y0, c):
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = np.random.uniform(y_range[0], y_range[1], num_points)
        z_values = self.catenary_function(x_values, y0, c, x0) + np.random.normal(0, noise_factor, num_points)
        return pd.DataFrame({'x': x_values, 'y': y_values, 'z': z_values, 'cluster_advanced': cluster_id})


    def test_assigned_catenaries_proximity(self):
        cluster_id = 'Cluster1'
        num_points = 5000
        x_range = (-5, -4)
        y_range = (-5, -5)
        x0 = -4.5
        y0 = -4.5
        c = 150
        noise_factor = 0.1

        generated_df = self.generate_noisy_points_3d(cluster_id, num_points, x_range, y_range, noise_factor, x0, y0, c)

        assigned_catenaries = calculate_assigned_catenaries(generated_df)

        print(assigned_catenaries)

        self.assertIn(cluster_id, assigned_catenaries)

        catenary_points = assigned_catenaries[cluster_id]

        cluster_data = generated_df[generated_df['cluster_advanced'] == cluster_id][['x', 'y', 'z']].values
        distances = cdist(cluster_data, catenary_points)
        min_distances = np.min(distances, axis=1)

        max_allowed_distance = 0.5

        self.assertTrue(np.all(min_distances < max_allowed_distance))


if __name__ == '__main__':
    unittest.main()
