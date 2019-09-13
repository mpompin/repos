from sklearn.neighbors import KDTree


def ANN(X, k,metric='chebyshev'):
    tree = KDTree(X, leaf_size=1, metric=metric)
    dists, nnidx = tree.query(X, k=k) #oi kNN tou kathe leaf
    return nnidx, dists


def ANNR(X, rV,metric='chebyshev'):
    tree = KDTree(X, leaf_size=1, metric=metric)
    nnnidx = tree.query_radius(X, r=rV, count_only=True) #mporei bug, thelei kai edw maxnorm
    return nnnidx


def nneighforgivenr(X, rV,metric='chebyshev'):
    npV = ANNR(X, rV,metric)
    npV[npV == 0] = 1
    return npV
