import numpy as np
from scipy.optimize import curve_fit
from skspatial.objects import Plane

"""
This document is used to calculate the coordinates of the wires linked to the catenaries
"""
def _catenary_function(x, y0, c, x0):
    """
    Catenary function equation.

    Args:
        x (array_like): x values.
        y0 (float): Vertical shift parameter.
        c (float): Scale parameter.
        x0 (float): Horizontal shift parameter.

    Returns:
        array_like: Calculated y values using the catenary function.
    """

    return y0 + c * np.cosh((x - x0) / c)

def calculate_assigned_catenaries(df_coordinates):
    """
    Calculates assigned catenaries for each cluster in the DataFrame.

    Args:
        df_coordinates (DataFrame): Input data with 'cluster_advanced', 'x', 'y', and 'z' columns.

    Returns:
        dict: A dictionary mapping cluster IDs to arrays of catenary points in 3D space.
    """
    assigned_catenaries = {}

    for cluster_id in df_coordinates['cluster_advanced'].unique():
        if cluster_id == -1:
            continue

        cluster_data = df_coordinates[df_coordinates['cluster_advanced'] == cluster_id][['x', 'y', 'z']]
        points = cluster_data.values

        plane = Plane.best_fit(points)
        plane_point = plane.point
        plane_normal = plane.normal

        base_vector_Z = plane_normal / np.linalg.norm(plane_normal)

        if base_vector_Z[0] != 0 or base_vector_Z[1] != 0:
            temp_vector = np.array([0, 0, 1])
        else:
            temp_vector = np.array([1, 0, 0])

        base_vector_X = np.cross(base_vector_Z, temp_vector)
        base_vector_X /= np.linalg.norm(base_vector_X)

        base_vector_Y = np.cross(base_vector_X, base_vector_Z)

        new_coordinates = []

        for point in points:
            x, y, z = point

            x_prime = np.dot([x - plane_point[0], y - plane_point[1], z - plane_point[2]], base_vector_X)
            y_prime = np.dot([x - plane_point[0], y - plane_point[1], z - plane_point[2]], base_vector_Y)

            new_coordinates.append([x_prime, y_prime])

        data = np.array(new_coordinates)

        x_prime_data = data[:, 0]
        y_prime_data = data[:, 1]


        params, covariance = curve_fit(_catenary_function, x_prime_data, y_prime_data)
        y0_fit, c_fit, x0_fit = params

        x_fit = np.linspace(min(x_prime_data), max(x_prime_data), 100)
        y_fit = _catenary_function(x_fit, y0_fit, c_fit, x0_fit)

        catenary_fit = np.column_stack((x_fit, y_fit))

        catenary_3d = []

        for x_prime, y_prime in catenary_fit:
            x_3d = x_prime * base_vector_X + y_prime * base_vector_Y + plane_point
            catenary_3d.append(x_3d)

        catenary_3d = np.array(catenary_3d)

        assigned_catenaries[cluster_id] = catenary_3d

    return assigned_catenaries