import math
import numpy as np
import scipy.sparse as ss
import sklearn.cluster as sl

def objscore(labels, k, mat1, mat2=None):
    """Scores a clustering using the objective matrix given

    Args:
        labels (list of int): Clustering assignment.
        k (int): Number of clusters.
        mat1 (csc matrix): Numerator matrix of objective score.
        mat2 (csc matrix): Denominator matrix of objective score. Default is no denominator.

    Returns:
        float: Score.

    """
    
    tot = 0
    row=np.empty(k,dtype=object)
    for pos, item in enumerate(labels):
        if type(row[item])!=list:
            row[item] = [pos]
        else:
            row[item].append(pos)
    for j in range(k):
        num = mat1[:,row[j]].tocsr()[row[j],:].sum()
        if mat2!=None:
            den= mat2[:,row[j]].tocsr()[row[j],:].sum()
            if den==0:
                den=1
            num = num/den
        tot += num
    return float(round(tot,2))

def sizeorder(labels,k,pos,neg,largest=True):
    n = len(labels)
    eye=ss.eye(n,format='csc')
    clusscores=np.empty(k)
    lsize=0
    lclus=-1
    for j in range(k):
        row = [i for i in range(n) if labels[i] == j]
        col = [0 for i in range(n) if labels[i] == j]
        dat = [1 for i in range(n) if labels[i] == j]
        if largest==False and len(dat)>lsize:
            lsize=len(dat)
            lclus=j
        vec = ss.coo_matrix((dat, (row, col)), shape=(n, 1))
        vec = vec.tocsc()
        x = vec.transpose() * pos * vec
        y = vec.transpose() * (neg+eye)* vec
        z=float(x[0,0])/float(y[0,0])
        clusscores[j] = z
    new=[x for x in range(n) if labels[x]!=lclus]
    scores = [clusscores[labels[i]] for i in new]
    return [x for _,x in sorted(zip(scores,new))]
    
def invdiag(M):
    """Inverts a positive diagonal matrix.

    Args:
        M (csc matrix): matrix to invert

    Returns:
        scipy sparse matrix of inverted diagonal

    """

    d = M.diagonal()
    dd = [1 / max(x, 1 / 999999999) for x in d]
    return ss.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()


def sqrtinvdiag(M):
    """Inverts and square-roots a positive diagonal matrix.

    Args:
        M (csc matrix): matrix to invert

    Returns:
        scipy sparse matrix of inverted square-root of diagonal
    """

    d = M.diagonal()
    dd = [1 / max(np.sqrt(x), 1 / 999999999) for x in d]

    return ss.dia_matrix((dd, [0]), shape=(len(d), len(d))).tocsc()


def merge(elemlist):
    """Merges pairs of clusters randomly. 

    Args:
        elemlist (list of lists of int): Specifies the members of each cluster in the current clustering

    Returns:
        list of lists of int: New cluster constituents
        boolean: Whether last cluster was unable to merge
        list of int: List of markers for current clustering, to use as starting vectors.

    """
    k = len(elemlist)
    dc = False
    elemlist.append([])
    perm = np.random.permutation(k)
    match = [k] * k
    for i in range(math.floor(k / 2)):
        me = perm[2 * i]
        you = perm[2 * i + 1]
        match[me] = you
        match[you] = me
    if k % 2 != 0:
        dontcut = perm[k - 1]
        dc = True
    nelemlist = [elemlist[i] + elemlist[match[i]] for i in range(k) if i < match[i] < k]
    numbers = [len(elemlist[i]) for i in range(k) if i < match[i] < k]
    if dc:
        nelemlist.append(elemlist[dontcut])
    return nelemlist, dc, numbers

def cut(elemlist, matrix, numbers, dc,mini):
    """Cuts clusters by separately normalised PCA.

    Args:
        elemlist (list of lists of int): Specifies the members of each cluster in the current clustering
        matrix (csc matrix): Matrix objective with which to cut.
        numbers (list of int): Marks previous clustering to use as starting vector.
        dc (boolean): Whether to skip cutting last cluster
        mini (boolean): Whether to minimise (instead of maximise) matrix objective.

    Returns:
        list of lists of int: new cluster constituents
    """
    nelemlist = []
    if dc:
        nelemlist.append(elemlist.pop())
    count = 0
    for i in elemlist:
        l = len(i)
        if l > 2:
            matrix1 = matrix[:, i].tocsr()
            matrix1 = matrix1[i, :].tocsc()
            val = 1 / math.sqrt(l)
            v = [-val] * numbers[count]
            w = [val] * (l - numbers[count])
            v = v + w
            if not mini:
                (w, v) = ss.linalg.eigsh(matrix1, 2, which='LA', maxiter=l, v0=v)
            else:
                (w, v) = ss.linalg.eigsh(matrix1, 2, which='SA', maxiter=l, v0=v)
            x = sl.KMeans(n_clusters=2,n_init=3,max_iter=100).fit(v)
            c1 = [i[y] for y in range(l) if x.labels_[y]==0]
            c2 = [i[y] for y in range(l) if x.labels_[y]==1]
            nelemlist.append(c1)
            nelemlist.append(c2)
        elif len(i) == 2:
            if matrix[i[0], i[1]] > 0:
                nelemlist.append(i)
                nelemlist.append([])
            else:
                nelemlist.append([i[0]])
                nelemlist.append([i[1]])
        elif len(i) == 1:
            nelemlist.append(i)
            nelemlist.append([])
        else:
            nelemlist.append([])
            nelemlist.append([])
        count += 1
    return nelemlist


    
    
    
    