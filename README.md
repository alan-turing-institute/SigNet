# SigNet

[![DOI](https://zenodo.org/badge/147539304.svg)](https://zenodo.org/badge/latestdoi/147539304)

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
from signet.block_models import SSBM
from sklearn.metrics import adjusted_rand_score


# simple test on the signed stochastic block model 

    n = 50000  # number of nodes
    k = 2      # number of clusters
    eta = 0.1  # sign flipping probability
    p = 0.0002 # edge probability

(Ap, An), true_assignment = SSBM(n = n, k = k, pin = p, etain = eta) # construct a graph

c = Cluster((Ap, An))

predictions = c.spectral_cluster_laplacian(k = k, normalisation='sym') # cluster with the signed laplacian
score = adjusted_rand_score(predictions, true_assignment)

print(score)
```


## Installation

Install the latest version from this Github repository.
```
pip install git+https://github.com/alan-turing-institute/SigNet.git
```


## API Reference

The documentation of this package was automatically generated using Sphinx. To generate
the documentation locally:
1. Install sphinx and the readthedocs theme
  - `pip install sphinx -U`
  - `pip install sphinx_rtd_theme -U`
2. Switch to the `docs` folder and build the docs with `make html`

Alternatively, the documentation can be found at https://signet.readthedocs.io/en/latest/index.html.


## Tests

To check that the code is working for you, try to download and run the jupyter notebook inside the "tests" folder.

## Current Authors

If you have problems with the code please contact

- Peter Davies: p.w.Davies@warwick.ac.uk
- Aldo Glielmo: aldo.glielmo@kcl.ac.uk


## Reference

- The generalised eigenproblem clustering has been proposed and analysed in:

  [1] Cucuringu, M., Davies, P., Glielmo, A., Tyagi, H. *SPONGE: A generalized eigenproblem for clustering signed networks.* Proceedings of Machine Learning Research 89 (2019). http://proceedings.mlr.press/v89/cucuringu19a.html
  


