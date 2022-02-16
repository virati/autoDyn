from setuptools import setup

setup(
   name='autodyn',
   version='0.1',
   description='A module for differentiable dynamical systems',
   author='Vineet Tiruvadi',
   author_email='virati@gmail.com',
   package_dir={'':'src'},
   packages=['autodyn'],  #same as name
   install_requires=['wheel'], #external packages as dependencies
)