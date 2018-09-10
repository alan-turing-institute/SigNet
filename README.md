# SigNet
A package for clustering of Signed Networks, the following algorithms were implemented:

- Standard spectral clustering with
  - Adjacency matrix (with multiple normalisations)
  - Signed Laplacian matrix (with multiple normalisations)
  - Balance Ratio Cut
  - Balance Normalised Cut
  
- Semidefinite programming clustering (with exact and approximate solvers)

- Generalised eigenproblem clustering (with multiple normalisations)

- Clustering using a signed variant of the Bethe Hessian matrix

## Typical usage

```python
from signet.cluster import Cluster 
from signet.block_models import SSMB
from sklearn.metrics import adjusted_rand_score


# simple test on the signed stochastic block model 

n = 50000 # number of nodes
eta = 0.1 # sign flipping probability
p = 0.0002 # sparsity

(Ap, An), true_assignment = SSBM(n, k, pin, etain) # construct a graph

c = Cluster((Ap, An))

predictions = c.spectral_cluster_laplacian(k = k, normalisation='sym') # cluster with the signed laplacian
score = adjusted_rand_score(predictions, true_assignment)

print(score)
```


## Installation

Install the latest version from this Github repository.

```
pip install git+https://github.com/alan-turing-institute/signet.git
```


## API Reference

The documentation of this package was automatically generated using Sphinx. To generate
the documentation locally:
1. Install sphinx and the readthedocs theme
  - `pip install sphinx -U`
  - `pip install sphinx_rtd_theme -U`
2. Switch to the `docs` folder and build the docs with `make html`

## Tests

TODO: add some jupyter tests

## Current Authors

If you have problems with the code please contact

- Peter Davies: p.w.Davies@warwick.ac.uk
- Aldo Glielmo: aldo.glielmo@kcl.ac.uk
