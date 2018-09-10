import subprocess
from setuptools import setup

# Install numpy separately as ecos dependency for cvxpy installation fails if numpy
# is not already installed
subprocess.Popen(["python", '-m', 'pip', 'install', 'numpy'])

setup(name='SigNet',
      version='0.1.0',
      description='A package for clustering signed networks',
      long_description=open('README.md').read(),
      author='Peter Davies, Aldo Glielmo',
      author_email='p.w.davies@warwick.ac.uk, aldo.glielmo@kcl.ac.uk',
      packages=['signet'],
      install_requires=['scikit-learn','cvxpy', 'networkx'],
      zip_safe=False)