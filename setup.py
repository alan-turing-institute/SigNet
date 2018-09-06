from setuptools import setup

setup(name='signet',
      version='0.1.0',
      description='A package for clustering signed networks',
      long_description=open('README.md').read(),
      author='Peter Davies, Aldo Glielmo',
      author_email='p.w.davies@warwick.ac.uk, aldo.glielmo@kcl.ac.uk',
      packages=['signet'],
      install_requires=['numpy','scipy','scikit-learn','cvxpy', 'networkx'],
      zip_safe=False)