from setuptools import setup, find_packages

setup(name='pointparticlesinaflow',
      version='0.1',
      packages=find_packages(),
      author='Daniel J. Ruth',
      url=r'https://github.com/DeikeLab/point-particles-in-a-flow/',
      description='Simulation of point-particles in flow fields',
      long_description=open('README.md').read(),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          ],
      )