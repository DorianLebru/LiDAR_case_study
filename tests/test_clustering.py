import sys
sys.path.append('../')

import unittest
import pandas as pd
from lidar_wire_detection_package.clustering import number_of_wire

class TestWireDetection(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
