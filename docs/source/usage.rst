Typical usage of the package
============================

A typical usage of SigNet involves the initialisation of the Cluster class with a given pair of adjjacency matrices and a subsequent of a specific method.


.. code-block:: python

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

