# SigNet
A package for clustering of Signed Networks, the following algorithms were implemented

- Standard spectral clustering with
  - Adjacency matrix (with multiple normalisations)
  - Signed Laplacian matrix (with multiple normalisations)
  - Balance Ratio Cut
  - Balance Normalised Cut
  
- Semidefinite clustering (with exact and approximate solvers)

- Generalised eigenproblem clustering (with multiple normalisations)

## Typical usage

```python
from signet.cluster import Cluster 
from signet.block_models import SSMB
from sklearn.metrics import adjusted_rand_score

(Ap, An), true_assignment = SSBM(n, k, pin, etain)

c = Cluster((Ap, An))

predictions = c.spectral_cluster_laplacian(k = k, normalisation='sym')
score = adjusted_rand_score(predictions, true_assignment)

print(score)
```


## Installation

The easiest way to install the dependencies is by using conda or pip

```
coda install numpy scipy sklearn cvxpy
```


While at the moment the package can only be installed from source by downloading the folder and running

```
python setup.py install
```

in the main directory.


## API Reference

The documentation of this package was automatically generated using Sphinx 

## Tests

TODO: add some jupyter tests

## Current Authors

If you have problems with the code please contact

- Peter Davies: p.w.Davies@warwick.ac.uk
- Aldo Glielmo: aldo.glielmo@kcl.ac.uk
