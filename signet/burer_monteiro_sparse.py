import numpy as np
from scipy import optimize as opt


def augmented_lagrangian(A, r, printing=False, init=None):
    """Augmented Lagrangian optimisation of the BM problem.

    It finds the matrix X which maximises the Frobenius norm (A, X.dot(X.T))
    with the constraint of having unit elements along the diagonal of X.dot(X.T).

    Args:
        A (csc matrix): The adjacency matrix
        r (int): The rank of the final solution
        printing (bool): Whether to print optimisation information
        init (array): Initial guess for the solution. If None a random matrix is used.

    Returns:
        array: The optimal matrix of dimensions n x r
    """

    n, _ = A.shape
    y = np.ones(n).reshape((-1, 1))
    if init is None:
        X = np.random.uniform(-1, 1, size=(n, r))
    else:
        X = init
    penalty = 1
    gamma = 10
    eta = .25
    target = .01  # 0.01
    vec = _constraint_term_vec(n, X)
    v = vec.reshape((1, -1)).dot(vec)
    v_best = v
    while v > target:
        Rv = _matrix_to_vector(X)
        if printing == True:
            print('Starting L-BFGS-B on augmented Lagrangian..., v is ', v)
        optimizer = opt.minimize(lambda R_vec: _augmented_lagrangian_func(
            R_vec, A, y, penalty, n, r), Rv, jac=lambda R_vec: _jacobian(R_vec, A, n, y, penalty, r), method="L-BFGS-B")
        if printing == True:
            print('Finishing L-BFGS-B on augmented Lagrangian...')
        X = _vector_to_matrix(optimizer.x, r)
        vec = _constraint_term_vec(n, X)
        v = vec.reshape((1, -1)).dot(vec)
        if printing == True:
            print('Finish updating variables...')
        if v < eta * v_best:
            y = y - penalty * vec
            v_best = v
        else:
            penalty = gamma * penalty
    if printing == True:
        print('Augmented Lagrangian terminated.')
    return X


def _generate_random_rect(n, k):
    """
    Returns a random initialization of matrix.
    """

    X = np.random.uniform(-1, 1, (n, k))
    for i in range(n):
        X[i, :] = X[i, :] / np.linalg.norm(X[i, :])
    return X


def _basis_vector(size, index):
    """
    Returns a basis vector with 1 on certain index.
    """

    vec = np.zeros(size)
    vec[index] = 1
    return vec


def _trace_vec(X):
    """
    Returns a vector containing norm square of row vectors of X.
    """

    vec = np.einsum('ij, ij -> i', X, X)

    return vec.reshape((-1, 1))


def _constraint_term_vec(n, X):
    """
    Returns the vector required to compute objective function value.
    """

    vec = _trace_vec(X)
    constraint = vec - np.ones(n).reshape((-1, 1))

    return constraint


def _augmented_lagrangian_func(Xv, A, y, penalty, n, k):
    """
    Returns the value of objective function of augmented Lagrangian.
    """

    X = _vector_to_matrix(Xv, k)

    vec = _constraint_term_vec(n, X)

    AX = A.dot(X)

    objective1 = - np.einsum('ij, ij -> ', X, AX)  # Trace(Y*X*X.T)

    objective2 = - y.reshape((1, -1)).dot(vec)

    objective3 = + penalty / 2 * vec.reshape((1, -1)).dot(vec)

    objective = objective1 + objective2 + objective3

    return objective


def _vector_to_matrix(Xv, k):
    """
    Returns a matrix from reforming a vector.
    """
    U = Xv.reshape((-1, k))
    return U


def _matrix_to_vector(X):
    """
    Returns a vector from flattening a matrix.
    """

    u = X.reshape((1, -1)).ravel()
    return u


def _jacobian(Xv, Y, n, y, penalty, k):
    """
    Returns the Jacobian matrix of the augmented Lagrangian problem.
    """

    X = _vector_to_matrix(Xv, k)

    vec_trace_A_ = _trace_vec(X).ravel() - 1.

    vec_second_part = np.einsum('ij, i -> ij', X, y.ravel())

    vec_third_part = np.einsum('ij, i -> ij', X, vec_trace_A_)

    jacobian = - 2 * Y.dot(X) - 2 * vec_second_part + \
               2 * penalty * vec_third_part

    jac_vec = _matrix_to_vector(jacobian)
    return jac_vec.reshape((1, -1)).ravel()


if __name__ == "__main__":
    from block_models import fSSBM

    np.set_printoptions(precision=1)
    n = 1000
    At, assig = fSSBM(n=n, k=2, p=0.9, eta=0.1)
    A = At[0] - At[1]

    r = int(np.sqrt(2 * n))
    X = augmented_lagrangian(A=A, r=r, printing=True, init=None)
