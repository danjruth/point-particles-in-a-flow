from setuptools import setup, find_packages

setup(name='point_bubble_JHTDB',
      packages=find_packages(),
      install_requires=[
          'scipy',
          'matplotlib',
          'numpy',
          'pickle',
          'time',
          'pyJHTDB'
          ],
      )