# LiDAR_case_study
The task is to create a python package that can identify how many wires are present in a lidar point cloud and generate 3D catenary models of these wires.

## Approach:

### Clustering:
The initial step involves creating coarse clusters using the DBSCAN algorithm in 2D, specifically along the x and z axes. This separation of clusters is based on altitude differences. Subsequently, a Hough transform is employed to ascertain the primary directions of the points. This results in the drawing of interconnected lines, with each point being associated with the cluster of the nearest line. The process then continues with a quadratic polynomial approximation to define a new curve. This curve is once again utilized to assign each point to the nearest curve.

### Best Parameters:
The process of finding the optimal plane is undertaken, leading to the establishment of an associated 2D coordinate system. Within this coordinate system, the search for the finest parameters for each catenary is conducted. These parameters are then translated back into 3D coordinates.
