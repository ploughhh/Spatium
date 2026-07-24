import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix, csgraph
from scipy.sparse.linalg import eigsh


def build_graph_lap_pe(adata, basis, k=6, n_pe=8, flip_sign=True):
    coords = adata.obsm[basis]
    n_cells = coords.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    A = lil_matrix((n_cells, n_cells), dtype=np.float32)
    for i in range(n_cells):
        A[i, indices[i, 1:]] = 1
    A = A.maximum(A.T)

    L = csgraph.laplacian(A, normed=True)

    eigvals, eigvecs = eigsh(L, k=n_pe+1, which='SM')
    lap_pe = eigvecs[:, 1:n_pe+1]

    if flip_sign:
        sign_flip = np.random.choice([-1, 1], size=(1, lap_pe.shape[1]))
        lap_pe = lap_pe * sign_flip

    return lap_pe
