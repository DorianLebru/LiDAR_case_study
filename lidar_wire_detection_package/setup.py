from setuptools import setup, find_packages

setup(
    name='lidar_wire_detection_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-spatial',
        'pyarrow',
        'skspatial',
        'scipy',
        'scikit-learn'
    ],
)
