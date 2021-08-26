from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='ProHMR as a package',
    name='prohmr',
    packages=find_packages()
)