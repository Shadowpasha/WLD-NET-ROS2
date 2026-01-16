from setuptools import setup
import os
from glob import glob

package_name = 'wld_net'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS 2 package for WLD-Net image dehazing',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dehazing_node = wld_net.dehazing_node:main',
            'webcam_dehazing_node = wld_net.webcam_dehazing_node:main',
            'video_dehazing_node = wld_net.video_dehazing_node:main',
        ],
    },
)
