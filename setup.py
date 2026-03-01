from setuptools import setup, find_packages

package_name = 'agibotx2_fiera'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/demo_launch.py']),
        ('share/' + package_name + '/config', [
            'config/demo_config.yaml',
            'config/joint_limits.yaml',
        ]),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'pyyaml',
        'scipy',
    ],
    zip_safe=True,
    maintainer='Demo Team',
    maintainer_email='demo@example.com',
    description='Demo Fiera Agibot X2 - Pick & Place Cialde Caffe',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'demo_node = src.main_node:main',
        ],
    },
)
