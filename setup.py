from setuptools import setup, find_packages

setup(
    name="autodyn",
    version="0.1.4",
    description="A module for differentiable dynamical systems",
    author="Vineet Tiruvadi",
    author_email="virati@gmail.com",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["wheel"],  # external packages as dependencies
)
