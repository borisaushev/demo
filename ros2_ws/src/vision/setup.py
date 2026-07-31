import os
from setuptools import find_packages, setup

package_name = 'vision'

# Автоматически читаем зависимости для pip
the_lib_dir = os.path.dirname(os.path.realpath(__file__))
requirement_path = os.path.join(the_lib_dir, 'requirements.txt')
install_requires = ['setuptools']

if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires.extend(f.read().splitlines())

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=install_requires, 
    zip_safe=True,
    maintainer='bossb',
    maintainer_email='borisaushev.com@gmail.com',
    description='package responsible for distance estimation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'distance_estimation = vision.main:main'
        ],
    },
)
