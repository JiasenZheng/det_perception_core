from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['det_perception_core'],
    package_dir={'': 'python'},
    scripts=['scripts/infer_real_time']
)
setup(**d)
