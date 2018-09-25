import scipy.sparse as ss
import scipy as sp
import sklearn.cluster as sl
import sklearn.metrics as sm
import numpy as np
import math
from signet.utils import sqrtinvdiag, invdiag, cut, merge, objscore
from signet.burer_monteiro_sparse import augmented_lagrangian

np.set_printoptions(2)


class Cluster:
	"""Class containing all clustering algorithms for signed networks.

	This should be initialised with a tuple of two csc matrices, representing positive and negative adjacency
	matrix respectively (A^+ and A^-). It contains clustering algorithms as methods and graph specifications
	as attributes.

	Args:
		data (tuple): Tuple containing positive and negative adjacency matrix (A^+, A^-).

	Attributes:
		p (csc matrix): positive adjacency matrix.
		n (csc matrix): negative adjacency matrix.
		A (csc matrix): total adjacency matrix.
		D_p (csc matrix): diagonal degree matrix of positive adjacency.
		D_n (csc matrix): diagonal degree matrix of negative adjacency.
		Dbar (csc matrix): diagonal signed degree matrix.
		normA (csc matrix): symmetrically normalised adjacency matrix.
		size (int): number of nodes in network

	"""

	def __init__(self, data):
		self.p = data[0]
		self.n = data[1]
		self.A = (self.p - self.n).tocsc()
		self.D_p = ss.diags(self.p.sum(axis=0).tolist(), [0]).tocsc()
		self.D_n = ss.diags(self.n.sum(axis=0).tolist(), [0]).tocsc()
		self.Dbar = (self.D_p + self.D_n)
		d = sqrtinvdiag(self.Dbar)
		self.normA = d * self.A * d
		self.size = self.p.shape[0]

	def spectral_cluster_adjacency(self, k=2, normalisation='sym_sep', eigens=None, mi=None):

		"""Clusters the graph using eigenvectors of the adjacency matrix.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			normalisation (string): How to normalise for cluster size:
				'none' - do not normalise.
				'sym' - symmetric normalisation.
				'rw' - random walk normalisation.
				'sym_sep' - separate symmetric normalisation of positive and negative parts.
				'rw_sep' - separate random walk normalisation of positive and negative parts.

		Returns:
			array of int, or list of array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.

		"""
		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if eigens == None:
			eigens = k
		if mi == None:
			mi = self.size

		symmetric = True

		if normalisation == 'none':
			matrix = self.A

		elif normalisation == 'sym':
			d = sqrtinvdiag(self.Dbar)
			matrix = d * self.A * d

		elif normalisation == 'rw':
			d = invdiag(self.Dbar)
			matrix = d * self.A
			symmetric = False

		elif normalisation == 'sym_sep':
			d = sqrtinvdiag(self.D_p)
			matrix = d * self.p * d
			d = sqrtinvdiag(self.D_n)
			matrix = matrix - (d * self.n * d)

		elif normalisation == 'rw_sep':
			d = invdiag(self.D_p)
			matrix = d * self.p
			d = invdiag(self.D_n)
			matrix = matrix - (d * self.n)
			symmetric = False

		elif normalisation == 'neg':
			pos = self.p
			d = invdiag(self.D_n)
			neg = d * self.n
			x = (pos.sum() / neg.sum())
			neg = neg * x
			matrix = pos - neg

		if symmetric:
			(w, v) = ss.linalg.eigsh(matrix, eigens, maxiter=mi, which='LA')
		else:
			(w, v) = ss.linalg.eigs(matrix, eigens, maxiter=mi, which='LR')
		v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative
		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 1 - x:])).labels_ for x in kk]

	def spectral_cluster_adjacency_reg(self, k=2, normalisation='sym_sep', tau_p=None, tau_n=None, eigens=None,
	                                   mi=None):
		"""Clusters the graph using eigenvectors of the regularised adjacency matrix.

		Args:
			k (int): The number of clusters to identify.
			normalisation (string): How to normalise for cluster size:
				'none' - do not normalise.
				'sym' - symmetric normalisation.
				'rw' - random walk normalisation.
				'sym_sep' - separate symmetric normalisation of positive and negative parts.
				'rw_sep' - separate random walk normalisation of positive and negative parts.
			tau_p (int): Regularisation coefficient for positive adjacency matrix.
			tau_n (int): Regularisation coefficient for negative adjacency matrix.

		Returns:
			array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.

		"""

		if eigens == None:
			eigens = k

		if mi == None:
			mi = self.size

		if tau_p == None or tau_n == None:
			tau_p = 0.25 * np.mean(self.Dbar.data) / self.size
			tau_n = 0.25 * np.mean(self.Dbar.data) / self.size

		symmetric = True

		p_tau = self.p.copy()
		n_tau = self.n.copy()
		p_tau.data += tau_p
		n_tau.data += tau_n

		Dbar_c = self.size - self.Dbar.diagonal()

		Dbar_tau_s = (p_tau + n_tau).sum(axis=0) + (Dbar_c * abs(tau_p - tau_n))[None, :]

		Dbar_tau = ss.diags(Dbar_tau_s.tolist(), [0])

		if normalisation == 'none':
			matrix = self.A
			delta_tau = tau_p - tau_n

			def mv(v):
				return matrix.dot(v) + delta_tau * v.sum()


		elif normalisation == 'sym':
			d = sqrtinvdiag(Dbar_tau)
			matrix = d * self.A * d
			dd = d.diagonal()
			tau_dd = (tau_p - tau_n) * dd

			def mv(v):
				return matrix.dot(v) + tau_dd * dd.dot(v)

		elif normalisation == 'sym_sep':

			diag_corr = ss.diags([self.size * tau_p] * self.size).tocsc()
			dp = sqrtinvdiag(self.D_p + diag_corr)

			matrix = dp * self.p * dp

			diag_corr = ss.diags([self.size * tau_n] * self.size).tocsc()
			dn = sqrtinvdiag(self.D_n + diag_corr)

			matrix = matrix - (dn * self.n * dn)

			dpd = dp.diagonal()
			dnd = dn.diagonal()
			tau_dp = tau_p * dpd
			tau_dn = tau_n * dnd

			def mv(v):
				return matrix.dot(v) + tau_dp * dpd.dot(v) - tau_dn * dnd.dot(v)

		else:
			print('Error: choose normalisation')

		matrix_o = ss.linalg.LinearOperator(matrix.shape, matvec=mv)

		if symmetric:
			(w, v) = ss.linalg.eigsh(matrix_o, eigens, maxiter=mi, which='LA')
		else:
			(w, v) = ss.linalg.eigs(matrix_o, eigens, maxiter=mi, which='LR')

		v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative
		v = np.atleast_2d(v)
		x = sl.KMeans(n_clusters=k).fit(v)
		return x.labels_

	def spectral_cluster_bnc(self, k=2, normalisation='sym', eigens=None, mi=None):

		"""Clusters the graph by using the Balance Normalised Cut or Balance Ratio Cut objective matrix.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			normalisation (string): How to normalise for cluster size:
				'none' - do not normalise.
				'sym' - symmetric normalisation.
				'rw' - random walk normalisation.

		Returns:
			array of int, or list of array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.

		"""

		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if eigens == None:
			eigens = k
		if mi == None:
			mi = self.size

		symmetric = True

		if normalisation == 'none':
			matrix = self.A + self.D_n

		elif normalisation == 'sym':
			d = sqrtinvdiag(self.Dbar)
			matrix = d * (self.A + self.D_n) * d

		elif normalisation == 'rw':
			d = invdiag(self.Dbar)
			matrix = d * (self.A + self.D_n)
			symmetric = False

		if symmetric:
			(w, v) = ss.linalg.eigsh(matrix, eigens, maxiter=mi, which='LA')
		else:
			(w, v) = ss.linalg.eigs(matrix, eigens, maxiter=mi, which='LR')

		v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative

		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 1 - x:])).labels_ for x in kk]

	def spectral_cluster_laplacian(self, k=2, normalisation='sym_sep', eigens=None, mi=None):

		"""Clusters the graph using the eigenvectors of the graph signed Laplacian.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			normalisation (string): How to normalise for cluster size:
				'none' - do not normalise.
				'sym' - symmetric normalisation.
				'rw' - random walk normalisation.
				'sym_sep' - separate symmetric normalisation of positive and negative parts.
				'rw_sep' - separate random walk normalisation of positive and negative parts.

		Returns:
			array of int, or list of array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.

		"""
		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if eigens == None:
			eigens = k
		if mi == None:
			mi = self.size

		symmetric = True
		eye = ss.eye(self.size, format="csc")

		if normalisation == 'none':
			matrix = self.Dbar - self.A

		elif normalisation == 'sym':
			d = sqrtinvdiag(self.Dbar)
			matrix = eye - d * self.A * d

		elif normalisation == 'rw':
			d = invdiag(self.Dbar)
			matrix = eye - d * self.A
			symmetric = False

		elif normalisation == 'sym_sep':
			d = sqrtinvdiag(self.D_p)
			matrix = d * self.p * d
			d = sqrtinvdiag(self.D_n)
			matrix = matrix - (d * self.n * d)
			matrix = eye - matrix

		elif normalisation == 'rw_sep':
			d = invdiag(self.D_p)
			matrix = d * self.p
			d = invdiag(self.D_n)
			matrix = matrix - (d * self.n)
			matrix = eye - matrix
			symmetric = False

		if symmetric:
			(w, v) = ss.linalg.eigsh(matrix, eigens, maxiter=mi, which='SA')
		else:
			(w, v) = ss.linalg.eigs(matrix, eigens, maxiter=mi, which='SR')

		v = v / w  # weight eigenvalues by eigenvectors, since smaller eigenvectors are more likely to be informative
		v = np.atleast_2d(v)
		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 0:k - 1])).labels_ for x in kk]

	def geproblem_adjacency(self, k=4, normalisation='multiplicative', eigens=None, mi=None, nudge=0.5):

		"""Clusters the graph by solving a adjacency-matrix-based generalised eigenvalue problem.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			normalisation (string): How to normalise for cluster size:
				'none' - do not normalise.
				'additive' - add degree matrices appropriately.
				'multiplicative' - multiply by degree matrices appropriately.

		Returns:
			array of int, or list of array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.
			nudge (int): Amount added to diagonal to bound eigenvalues away from 0.

		"""
		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if eigens == None:
			# eigens = min(math.floor(math.sqrt(self.size)), math.ceil(k * math.log(k, 2) - 1))
			eigens = k
		if mi == None:
			mi = self.size

		eye = ss.eye(self.size, format="csc")
		if normalisation == 'none':
			matrix1 = self.n
			matrix2 = self.p

		elif normalisation == 'additive':
			matrix1 = self.n + self.D_p
			matrix2 = self.p + self.D_n

		elif normalisation == 'multiplicative':
			d = sqrtinvdiag(self.D_n)
			matrix1 = d * self.n * d
			d = sqrtinvdiag(self.D_p)
			matrix2 = d * self.p * d

		(w, v) = ss.linalg.eigsh(matrix1, k=1, maxiter=mi, which='SA')
		matrix1 = matrix1 + eye * (nudge - w[0])

		(w, v) = ss.linalg.eigsh(matrix2, k=1, maxiter=mi, which='SA')
		matrix2 = matrix2 + eye * (nudge - w[0])

		v0 = np.random.normal(0, 1, (self.p.shape[0], eigens))
		(w, v) = ss.linalg.lobpcg(matrix1, v0, B=matrix2, maxiter=mi, largest=False)

		v = v / w
		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 1 - x:])).labels_ for x in kk]

	def geproblem_laplacian(self, k=4, normalisation='multiplicative', eigens=None, mi=None, tau=1.):
		"""Clusters the graph by solving a Laplacian-based generalised eigenvalue problem.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			normalisation (string): How to normalise for cluster size:
				'none' - do not normalise.
				'additive' - add degree matrices appropriately.
				'multiplicative' - multiply by degree matrices appropriately.

		Returns:
			array of int, or list of array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.
			nudge (int): Amount added to diagonal to bound eigenvalues away from 0.

		"""
		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if eigens == None:
			eigens = k
		if mi == None:
			mi = self.size

		eye = ss.eye(self.size, format="csc")

		if normalisation == 'none':
			matrix1 = self.D_p - self.p
			matrix2 = self.D_n - self.n

		elif normalisation == 'additive':
			matrix1 = self.Dbar - self.p
			matrix2 = self.Dbar - self.n

		elif normalisation == 'multiplicative':

			d = sqrtinvdiag(self.D_n)
			matrix = d * self.n * d
			matrix2 = eye - matrix
			d = sqrtinvdiag(self.D_p)
			matrix = d * self.p * d
			matrix1 = eye - matrix

		matrix1 = matrix1 + eye * tau
		matrix2 = matrix2 + eye * tau

		v0 = np.random.normal(0, 1, (self.p.shape[0], eigens))
		(w, v) = ss.linalg.lobpcg(matrix1, v0, B=matrix2, maxiter=mi, largest=False)

		v = v / w
		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 0:x - 1])).labels_ for x in kk]

	def SPONGE(self, k=4, tau_p=1, tau_n=1, eigens=None, mi=None):
		"""Clusters the graph using the Signed Positive Over Negative Generalised Eigenproblem (SPONGE) clustering.

		The algorithm tries to minimises the following ratio (Lbar^+ + tau_n D^-)/(Lbar^- + tau_p D^+).
		The parameters tau_p and tau_n can be typically set to one.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			tau_n (float): regularisation of the numerator
			tau_p (float): regularisation of the denominator

		Returns:
			array of int, or list of array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.
			nudge (int): Amount added to diagonal to bound eigenvalues away from 0.

		"""

		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if eigens == None:
			eigens = k - 1
		if mi == None:
			mi = self.size

		matrix1 = self.D_p - self.p
		matrix2 = self.D_n - self.n

		matrix1 = matrix1 + tau_n * self.D_n
		matrix2 = matrix2 + tau_p * self.D_p

		v0 = np.random.normal(0, 1, (self.p.shape[0], eigens))
		(w, v) = ss.linalg.lobpcg(matrix1, v0, B=matrix2, maxiter=mi, largest=False)

		v = v / w
		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 0:x - 1])).labels_ for x in kk]



	def SPONGE_sym(self, k=4, tau_p=1, tau_n=1, eigens=None, mi=None):
		"""Clusters the graph using the symmetric normalised version of the SPONGE clustering algorithm.

		The algorithm tries to minimises the following ratio (Lbar_sym^+ + tau_n Id)/(Lbar_sym^- + tau_p Id).
		The parameters tau_p and tau_n can be typically set to one.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			tau_n (float): regularisation of the numerator
			tau_p (float): regularisation of the denominator

		Returns:
			array of int, or list of array of int: Output assignment to clusters.

		Other parameters:
			eigens (int): The number of eigenvectors to take. Defaults to k.
			mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.
			nudge (int): Amount added to diagonal to bound eigenvalues away from 0.

		"""

		listk = False

		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if eigens == None:
			eigens = k - 1
		if mi == None:
			mi = self.size

		eye = ss.eye(self.size, format="csc")

		d = sqrtinvdiag(self.D_n)
		matrix = d * self.n * d
		matrix2 = eye - matrix

		d = sqrtinvdiag(self.D_p)
		matrix = d * self.p * d
		matrix1 = eye - matrix

		matrix1 = matrix1 + tau_n * eye
		matrix2 = matrix2 + tau_p * eye

		v0 = np.random.normal(0, 1, (self.p.shape[0], eigens))
		(w, v) = ss.linalg.lobpcg(matrix1, v0, B=matrix2, maxiter=mi, largest=False)

		v = v / w
		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 0:x - 1])).labels_ for x in kk]

	def find_eigenvalues(self, k=100, matrix='laplacian'):
		"""Find top or bottom k eigenvalues of adjacency or laplacian matrix.

		The list of the top (bottom) k eigenvalues of the adjacency (laplacian) matrix is returned.
		This can be useful in identifying the number of clusters.

		Note:
			The Laplacian matrix used is the signed symmetric Laplacian.

		Args:
			k (int): Number of eigenvalues to return
			matrix (str): Type of matrix to diagonalise (either 'adjacency' or 'laplacian')

		Returns:
			array of float: An array of the first k eigenvalues, ordered in ascending or descending order
				(depending on the matrix type) """

		k = min(self.size, k)
		if matrix == 'adjacency':
			(w, v) = ss.linalg.eigsh(A_, k, which='LA')
			w = w[::-1]
		elif matrix == 'laplacian':
			(w, v) = ss.linalg.eigsh(self.symLbar, k, which='SA')

		else:
			raise ValueError('please select a valid matrix type')
		return w

	def spectral_cluster_bethe_hessian(self, k, mi=1000, r=None, justpos=True):
		"""Clustering based on signed Bethe Hessian.

		A low dimensional embedding is obtained via the lowest eigenvectors of the signed Bethe Hessian matrix Hbar and
		k-means is performed in this space.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			mi (int): Maximum number of iterations of the eigensolver.
			type (str): Types of normalisation of the Laplacian matrix. 'unnormalised', 'symmetric', 'random_walk'.

		Returns:
			array of int, or list of array of int: Label assignments.
			int: Suggested number of clusters for network.

		"""
		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if r is None:

			d = np.mean(self.Dbar.data)
			dsq = np.mean(self.Dbar.data ** 2)

			# r = np.sqrt(d)  # SSB
			r = np.sqrt(dsq / d - 1)  # general
		else:
			pass
		eigens = k - 1

		self.Hbar_p = (r ** 2 - 1) * ss.identity(self.size, format='csc') - r * self.A + self.Dbar

		(w, v) = ss.linalg.eigsh(self.Hbar_p, eigens, which='SA', maxiter=mi)
		if not justpos:
			r = - r

			self.Hbar_n = (r ** 2 - 1) * ss.identity(self.size, format='csc') - r * self.A + self.Dbar

			(wn, vn) = ss.linalg.eigsh(self.Hbar_n, eigens, which='SA', maxiter=mi)
			w = np.hstack((w, wn))
			v = np.hstack((v, vn))
			eigens = 2 * eigens
		klen = len([x for x in range(eigens) if w[x] < 0])
		idx = np.argsort(w)[0:k - 1]
		v = v[:, idx]
		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_, klen
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 0:x - 1])).labels_ for x in kk], klen + 1

	def SDP_cluster(self, k, solver='BM_proj_grad', normalisation='sym_sep'):
		"""Clustering based on a SDP relaxation of the clustering problem.

		A low dimensional embedding is obtained via the lowest eigenvectors of positive-semidefinite matrix Z
		which maximises its Frobenious product with the adjacency matrix and k-means is performed in this space.

		Args:
			k (int, or list of int) : The number of clusters to identify. If a list is given, the output is a corresponding list.
			solver (str): Type of solver for the SDP formulation.
				'interior_point_method' - Interior point method.
				'BM_proj_grad' - Burer Monteiro method using projected gradient updates.
				'BM_aug_lag' - Burer Monteiro method using augmented Lagrangian updates.

		Returns:
			array of int, or list of array of int: Label assignments.

		"""

		listk = False
		if isinstance(k, list):
			kk = k
			k = max(k)
			listk = True

		if normalisation == 'none':
			matrix = self.A

		elif normalisation == 'sym':
			d = sqrtinvdiag(self.Dbar)
			matrix = d * self.A * d

		elif normalisation == 'sym_sep':
			d = sqrtinvdiag(self.D_p)
			matrix = d * self.p * d
			d = sqrtinvdiag(self.D_n)
			matrix = matrix - (d * self.n * d)

		if solver == 'interior_point_method':
			import cvxpy as cvx

			# Define a cvx optimization variable
			Z = cvx.Variable((self.size, self.size), PSD=True)
			ones = np.ones(self.size)
			# Define constraints
			constraints = [cvx.diag(Z) == ones]
			# Define an objective function
			obj = cvx.Maximize(cvx.trace(self.A * Z))
			# Define an optimisation problem
			prob = cvx.Problem(obj, constraints)
			# Solve optimisation problem

			prob.solve(solver='CVXOPT')

			print("status:", prob.status)
			print("optimal value", prob.value)
			# print("optimal var", Z.value)
			print(Z.value)

			# Diagonalise solution
			(w, v) = sp.linalg.eigh(Z.value, eigvals=(self.size - k, self.size - 1))
			v = v * w

		elif solver == 'BM_proj_grad':

			r = math.floor(np.sqrt(2 * self.size) + 1)
			X = np.random.normal(0, 1, (self.size, r))
			ones = np.ones((self.size, 1))
			step = 2
			traces = []
			i = 0
			while True:
				AX = matrix.dot(X)
				G = 2 * AX
				X = X + step * G
				trace = np.einsum('ij, ij -> ', X, AX)
				traces.append(trace)
				Norms = np.linalg.norm(X, axis=1)
				X = np.divide(X, Norms[:, None])
				delta_trace = abs(traces[-1] - traces[-2]) / abs(traces[-2]) if i > 0 else 100.
				if delta_trace <= 0.01:
					break
				i += 1
			Z = X.T.dot(X)
			(w, v) = sp.linalg.eigh(Z, eigvals=(r - k, r - 1))
			v = X.dot(v)
			v = v * w

		elif solver == 'BM_aug_lag':
			r = int(np.sqrt(2 * self.size))
			X = augmented_lagrangian(A=matrix, r=r, printing=False, init=None)
			Z = X.T.dot(X)
			(w, v) = sp.linalg.eigh(Z, eigvals=(r - k, r - 1))
			v = X.dot(v)
			v = v * w

		else:
			raise ValueError('please specify a valid solver')

		if not listk:
			v = np.atleast_2d(v)
			x = sl.KMeans(n_clusters=k).fit(v)
			return x.labels_
		else:
			return [sl.KMeans(n_clusters=x).fit(np.atleast_2d(v[:, 1 - x:])).labels_ for x in kk]

	def waggle(self, k, labs, matrix=None, rounds=50, mini=False):
		"""Postprocessing based on iteratively merging and cutting clusters of the provided solution.

		Pairs of clusters are merged randomly.
		Merged clusters are then partitioned in two by spectral clustering on input matrix.

		Args:
			k (int): The number of clusters to identify.
			labs (array of int): Initial assignment to clusters.
			matrix (csc matrix): Matrix to use for partitioning. Defaults to un-normalised adjacency matrix.

		Returns:
			array of int: Output assignment to clusters.

		Other parameters:
			rounds (int): Number of iterations to perform.
			mini (boolean): Whether to minimise (rather than maximise) the input matrix objective when partitioning.

		"""
		if matrix == None:
			matrix = self.A
		elemlist = [[x for x in range(self.size) if labs[x] == i] for i in range(k)]
		if k == 2:
			rounds = 0
		for i in range(rounds):
			elemlist, dc, numbers = merge(elemlist)
			elemlist = cut(elemlist, matrix, numbers, dc, mini)
		cluster = [0] * self.size
		for i in range(len(elemlist)):
			for j in elemlist[i]:
				cluster[j] = i
		return cluster


if __name__ == "__main__":
	from block_models import SSBM

	n = 10000
	k = 10
	p = 35 / n
	eta = 0.
	Ac, truth = SSBM(n, k, p, eta)

	print('Network constructed')

	m = Cluster(Ac)

	pcapreds = m.spectral_cluster_laplacian(k, normalisation='sym')
	rscore = sm.adjusted_rand_score(truth, pcapreds)
	print('Symmetric Laplacian score is ', rscore)

	pcapreds = m.spectral_cluster_adjacency(k, normalisation='sym')
	rscore = sm.adjusted_rand_score(truth, pcapreds)
	print('Symmetric Adjacency score is ', rscore)

	pcapreds = m.SPONGE(k)
	rscore = sm.adjusted_rand_score(truth, pcapreds)
	print('SPONGE core is ', rscore)

	pcapreds = m.SPONGE_sym(k)
	rscore = sm.adjusted_rand_score(truth, pcapreds)
	print('SPONGE sym score is ', rscore)
