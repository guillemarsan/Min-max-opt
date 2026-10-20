import os
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pypower.api as pp


def random_problem(
    lat_dim: int,
    num_neur: int,
    num_ncoup: int,
    rand_seed: int,
    sphere: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random minmax problem instance with Q lxl symmetric, b lx1, E lxN, and Tp Nx1.
    E has num_ncoup non-coupled constraints (i.e. num_ncoup columns are orthogonal to all the eigenvectors with positive eigenvalues of Q)

    Parameters
    ----------
    lat_dim : int
        Dimensionality of the latent space (l).

    num_neur : int
        Number of constraints (N).

    num_ncoup : int
        Number of non-coupled constraints.

    rand_seed : int
        Random seed for reproducibility.

    sphere : bool
        If True, generate the problem on a sphere.

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    """

    np.random.seed(rand_seed)

    # Generate random symmetric Q
    A = np.random.randn(lat_dim, lat_dim)
    Q = A + A.T

    # Ensure Q is indefinite if there are non-coupled constraints
    if num_ncoup > 0:
        eigvals, _ = np.linalg.eigh(Q)
        while sum(eigvals > 0) == 0 or sum(eigvals < 0) == 0:
            A = np.random.randn(lat_dim, lat_dim)
            Q = A + A.T
            eigvals, _ = np.linalg.eigh(Q)

    # Generate random b
    b = np.random.randn(lat_dim)

    # Generate random E
    E = np.random.randn(num_neur, lat_dim)

    # Make some constraints non-coupled
    if num_ncoup > 0:
        eigvals, eigvecs = np.linalg.eigh(Q)
        pos_eigvecs = eigvecs[:, eigvals < 0]
        for i in range(num_ncoup):
            # Delete the component of E[i, :] along the positive eigenvectors of Q
            E[i, :] = E[i, :] - pos_eigvecs @ (pos_eigvecs.T @ E[i, :])

    if sphere:
        E = E / np.linalg.norm(E, axis=1, keepdims=True)  # Normalize rows of E
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E, decimals=7), axis=0, return_index=True)
        E = vals[np.argsort(idx)]
        num_neur = E.shape[0]
        Tp = np.ones(num_neur) / 2
    else:
        # Generate random Tp
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E, decimals=7), axis=0, return_index=True)
        E = vals[np.argsort(idx)]
        num_neur = E.shape[0]
        Tp = np.random.randn(num_neur)

    return Q, b, E, Tp


def random_problem_ccvcvx(
    lat_dim: int,
    num_neur: int,
    num_ncoup: int,
    rand_seed: int,
    sphere: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Callable,
]:
    """
    Generate a random minmax problem instance with Q lxl symmetric, b lx1, E lxN, and Tp Nx1.
    Q = ((Q_aa, Q_ab), (Q_ab^T, Q_bb)) fulfills:
    - Q_aa - Q_ab Q_bb^-1 Q_ab^T > 0
    - Q_bb < 0

    Return also its diagonal form
    Q_diag = diag(Q_aa - Q_ab Q_bb^-1 Q_ab^T, Q_bb)

    E_diag has num_ncoup non-coupled constraints w.r.t Q_diag
    (i.e. num_ncoup columns are orthogonal to all the eigenvectors with positive eigenvalues of Q_diag)

    Parameters
    ----------
    lat_dim : int
        Dimensionality of the latent space (l).

    num_neur : int
        Number of constraints (N).

    num_ncoup : int
        Number of non-coupled constraints.

    rand_seed : int
        Random seed for reproducibility.

    sphere : bool
        If True, generate the problem on a sphere.

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    Q_diag : np.ndarray
        Diagonalized coupling matrix of shape (l, l).

    b_diag : np.ndarray
        Diagonalized bias vector of shape (l,).

    E_diag : np.ndarray
        Diagonalized constraint matrix of shape (N, l).

    Tp_diag : np.ndarray
        Diagonalized constraint thresholds of shape (N,).

    decoder : callable
        Function that takes as input a vector in the diagonalized space and outputs the corresponding vector in the original space.

    """

    np.random.seed(rand_seed)

    # def SPD_sample(n):
    #     A = np.random.randn(n, n)
    #     alpha = np.random.uniform(1.5, 3.0)
    #     return (A.T + A) / (2 * np.sqrt(n)) + alpha * np.eye(n)

    def SPD_sample(n):
        # Eigvals
        eigvals = np.abs(0.25 * np.random.randn(n) + 1)

        # Random orthogonal Q
        X = np.random.randn(n, n)
        Q, R = np.linalg.qr(X)
        Q *= np.sign(np.diag(R))

        # Spectral construction
        M = Q @ np.diag(eigvals) @ Q.T

        return M

    # Choose randomly how many minimization and maximization variables
    mins = np.random.randint(1, lat_dim)
    maxs = lat_dim - mins

    # Generate random symmetric Q_bb < 0
    Q_bb = -SPD_sample(maxs)

    # Generate random Q_ab
    Q_ab = np.random.randn(mins, maxs)

    # Generate random Aux > 0
    Aux = SPD_sample(mins)
    # And now sample Q_aa s.t. Q_aa - Q_ab Q_bb^-1 Q_ab.T = Aux
    Q_aa = Aux + Q_ab @ np.linalg.inv(Q_bb) @ Q_ab.T

    # Make sure everything worked
    assert np.all(np.linalg.eigvals(Q_aa - Q_ab @ np.linalg.inv(Q_bb) @ Q_ab.T) > 0)
    assert np.all(np.linalg.eigvals(Q_bb) < 0)

    # Assemble Q
    Q = np.block([[Q_aa, Q_ab], [Q_ab.T, Q_bb]])

    # Construct Q_diag
    Q_diag = np.zeros_like(Q)
    Q_diag[:mins, :mins] = Q_aa - Q_ab @ np.linalg.inv(Q_bb) @ Q_ab.T
    Q_diag[mins:, mins:] = Q_bb

    # Generate random b_diag s.t. Q_diag^-1 b_diag is gaussian
    Q_diag_inv_b_diag = np.random.randn(lat_dim)
    b_diag = Q_diag @ Q_diag_inv_b_diag

    # Construct b
    b = np.zeros_like(b_diag)
    b[:mins] = b_diag[:mins] + b_diag[mins:] @ np.linalg.inv(Q_bb) @ Q_ab.T
    b[mins:] = b_diag[mins:]

    # Generate E_diag with num_ncoup non-coupled constraints w.r.t Q_diag
    E_diag = np.random.randn(num_neur, lat_dim)
    if num_ncoup > 0:
        eigvals, eigvecs = np.linalg.eigh(Q_diag)
        neg_eigvecs = eigvecs[:, eigvals < 0]
        for i in range(num_ncoup):
            # Delete the component of E_diag[i, :] along the negative eigenvectors of Q_diag
            E_diag[i, :] = E_diag[i, :] - neg_eigvecs @ (neg_eigvecs.T @ E_diag[i, :])

    if sphere:
        E_diag = E_diag / np.linalg.norm(
            E_diag, axis=1, keepdims=True
        )  # Normalize rows of E_diag
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E_diag, decimals=7), axis=0, return_index=True)
        E_diag = vals[np.argsort(idx)]
        num_neur = E_diag.shape[0]
        Tp_diag = np.ones(num_neur)
    else:
        # Generate random Tp
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E_diag, decimals=7), axis=0, return_index=True)
        E_diag = vals[np.argsort(idx)]
        num_neur = E_diag.shape[0]
        Tp_diag = np.random.randn(num_neur)

    # Construct E
    E = np.zeros_like(E_diag)
    E[:, :mins] = E_diag[:, :mins] + E_diag[:, mins:] @ np.linalg.inv(Q_bb) @ Q_ab.T
    E[:, mins:] = E_diag[:, mins:]

    Tp = Tp_diag.copy()

    decoder = lambda x: np.vstack(
        (
            x[:mins, :] + Q_ab @ np.linalg.inv(Q_bb) @ x[mins:, :],
            x[mins:, :],
        )
    )

    return Q, b, E, Tp, Q_diag, b_diag, E_diag, Tp_diag, decoder


def random_problem_ccvcvx2(
    lat_dim: int,
    num_neur: int,
    num_ncoup: int,
    rand_seed: int,
    sphere: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Callable,
]:
    """
    Generate a random minmax problem instance with Q lxl symmetric, b lx1, E lxN, and Tp Nx1.
    Q = ((Q_aa, Q_ab), (Q_ab^T, Q_bb)) fulfills:
    - Q_aa - Q_ab Q_bb^-1 Q_ab^T > 0
    - Q_bb < 0

    Return also its diagonal form
    Q_diag = diag(Q_aa - Q_ab Q_bb^-1 Q_ab^T, Q_bb)

    E_diag has num_ncoup non-coupled constraints w.r.t Q_diag
    (i.e. num_ncoup columns are orthogonal to all the eigenvectors with positive eigenvalues of Q_diag)

    Parameters
    ----------
    lat_dim : int
        Dimensionality of the latent space (l).

    num_neur : int
        Number of constraints (N).

    num_ncoup : int
        Number of non-coupled constraints.

    rand_seed : int
        Random seed for reproducibility.

    sphere : bool
        If True, generate the problem on a sphere.

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    Q_diag : np.ndarray
        Diagonalized coupling matrix of shape (l, l).

    b_diag : np.ndarray
        Diagonalized bias vector of shape (l,).

    E_diag : np.ndarray
        Diagonalized constraint matrix of shape (N, l).

    Tp_diag : np.ndarray
        Diagonalized constraint thresholds of shape (N,).

    decoder : callable
        Function that takes as input a vector in the diagonalized space and outputs the corresponding vector in the original space.

    """

    np.random.seed(rand_seed)

    # def SPD_sample(n):
    #     A = np.random.randn(n, n)
    #     alpha = np.random.uniform(1.5, 3.0)
    #     return (A.T + A) / (2 * np.sqrt(n)) + alpha * np.eye(n)

    def SPD_sample(n):
        # Eigvals
        eigvals = np.abs(np.random.randn(n))

        # Random orthogonal Q
        X = np.random.randn(n, n)
        Q, R = np.linalg.qr(X)
        Q *= np.sign(np.diag(R))

        # Spectral construction
        M = Q @ np.diag(eigvals) @ Q.T

        return M

    # Choose randomly how many minimization and maximization variables
    mins = np.random.randint(1, lat_dim)
    maxs = lat_dim - mins

    # Generate random symmetric Q_bb < 0
    Q_bb = -SPD_sample(maxs)

    # Generate random Q_ab
    Q_ab = np.random.randn(mins, maxs)

    # Generate random Aux > 0
    Aux = SPD_sample(mins)
    # And now sample Q_aa s.t. Q_aa - Q_ab Q_bb^-1 Q_ab.T = Aux
    Q_aa = Aux + Q_ab @ np.linalg.inv(Q_bb) @ Q_ab.T

    # Make sure everything worked
    assert np.all(np.linalg.eigvals(Q_aa - Q_ab @ np.linalg.inv(Q_bb) @ Q_ab.T) > 0)
    assert np.all(np.linalg.eigvals(Q_bb) < 0)

    # Assemble Q
    Q = np.block([[Q_aa, Q_ab], [Q_ab.T, Q_bb]])

    # Generate random b
    b = np.random.randn(lat_dim)

    # Construct Q_diag
    Q_diag = np.zeros_like(Q)
    Q_diag[:mins, :mins] = Q_aa - Q_ab @ np.linalg.inv(Q_bb) @ Q_ab.T
    Q_diag[mins:, mins:] = Q_bb

    # Construct b_diag
    b_diag = np.zeros_like(b)
    b_diag[:mins] = b[:mins] - b[mins:] @ np.linalg.inv(Q_bb) @ Q_ab.T
    b_diag[mins:] = b[mins:]

    # Generate E_diag with num_ncoup non-coupled constraints w.r.t Q_diag
    E_diag = np.random.randn(num_neur, lat_dim) + (
        np.random.randn(num_neur, lat_dim) * 0.3
    )
    if num_ncoup > 0:
        eigvals, eigvecs = np.linalg.eigh(Q_diag)
        pos_eigvecs = eigvecs[:, eigvals < 0]
        for i in range(num_ncoup):
            # Delete the component of E_diag[i, :] along the positive eigenvectors of Q_diag
            E_diag[i, :] = E_diag[i, :] - pos_eigvecs @ (pos_eigvecs.T @ E_diag[i, :])

    if sphere:
        E_diag = E_diag / np.linalg.norm(
            E_diag, axis=1, keepdims=True
        )  # Normalize rows of E_diag
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E_diag, decimals=7), axis=0, return_index=True)
        E_diag = vals[np.argsort(idx)]
        num_neur = E_diag.shape[0]
        Tp_diag = np.ones(num_neur) / 2
    else:
        # Generate random Tp
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E_diag, decimals=7), axis=0, return_index=True)
        E_diag = vals[np.argsort(idx)]
        num_neur = E_diag.shape[0]
        Tp_diag = np.random.randn(num_neur)

    # Construct E
    E = np.zeros_like(E_diag)
    E[:, :mins] = E_diag[:, :mins] + E_diag[:, mins:] @ np.linalg.inv(Q_bb) @ Q_ab.T
    E[:, mins:] = E_diag[:, mins:]

    Tp = Tp_diag.copy()

    decoder = lambda x: np.vstack(
        (
            x[:mins, :],
            x[mins:, :] - np.linalg.inv(Q_bb) @ Q_ab.T @ x[:mins, :],
        )
    )

    return Q, b, E, Tp, Q_diag, b_diag, E_diag, Tp_diag, decoder


def random_problem_ccvcvx3(
    lat_dim: int,
    num_neur: int,
    num_ncoup: int,
    rand_seed: int,
    sphere: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Callable,
]:
    """
    Generate a random minmax problem instance with Q lxl symmetric, b lx1, E lxN, and Tp Nx1.
    Q = ((Q_aa, Q_ab), (Q_ab^T, Q_bb)) fulfills:
    - Q_aa - Q_ab Q_bb^-1 Q_ab^T > 0
    - Q_bb < 0

    Return also its diagonal form
    Q_diag = diag(Q_aa - Q_ab Q_bb^-1 Q_ab^T, Q_bb)

    E_diag has num_ncoup non-coupled constraints w.r.t Q_diag
    (i.e. num_ncoup columns are orthogonal to all the eigenvectors with positive eigenvalues of Q_diag)

    Parameters
    ----------
    lat_dim : int
        Dimensionality of the latent space (l).

    num_neur : int
        Number of constraints (N).

    num_ncoup : int
        Number of non-coupled constraints.

    rand_seed : int
        Random seed for reproducibility.

    sphere : bool
        If True, generate the problem on a sphere.

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    Q_diag : np.ndarray
        Diagonalized coupling matrix of shape (l, l).

    b_diag : np.ndarray
        Diagonalized bias vector of shape (l,).

    E_diag : np.ndarray
        Diagonalized constraint matrix of shape (N, l).

    Tp_diag : np.ndarray
        Diagonalized constraint thresholds of shape (N,).

    decoder : callable
        Function that takes as input a vector in the diagonalized space and outputs the corresponding vector in the original space.

    """

    np.random.seed(rand_seed)

    def SPD_sample(n):
        A = np.random.randn(n, n)
        alpha = np.random.uniform(1.5, 3.0)
        return (A.T + A) / (2 * np.sqrt(n)) + alpha * np.eye(n)

    # def SPD_sample(n):
    #     # Eigvals
    #     eigvals = np.abs(np.random.randn(n))

    #     # Random orthogonal Q
    #     X = np.random.randn(n, n)
    #     Q, R = np.linalg.qr(X)
    #     Q *= np.sign(np.diag(R))

    #     # Spectral construction
    #     M = Q @ np.diag(eigvals) @ Q.T

    #     return M

    # Choose randomly how many minimization and maximization variables
    mins = np.random.randint(1, lat_dim)
    maxs = lat_dim - mins

    # Generate random symmetric Q_bb > 0
    Q_bb = SPD_sample(mins)

    # Generate random Q_ba
    Q_ba = np.random.randn(mins, maxs)

    # Generate random Aux < 0
    Aux = -SPD_sample(maxs)
    # And now sample Q_aa s.t. Q_aa - Q_ba.T Q_bb^-1 Q_ba = Aux
    Q_aa = Aux + Q_ba.T @ np.linalg.inv(Q_bb) @ Q_ba

    # Make sure everything worked
    assert np.all(np.linalg.eigvals(Q_aa - Q_ba.T @ np.linalg.inv(Q_bb) @ Q_ba) < 0)
    assert np.all(np.linalg.eigvals(Q_bb) > 0)

    # Assemble Q
    Q = -np.block([[Q_bb, Q_ba], [Q_ba.T, Q_aa]])

    # Generate random b
    b = -np.random.randn(lat_dim)

    # Construct Q_diag
    Q_diag = np.zeros_like(Q)
    Q_diag[:mins, :mins] = Q_bb
    Q_diag[mins:, mins:] = Q_aa - Q_ba.T @ np.linalg.inv(Q_bb) @ Q_ba

    # Construct b_diag
    b_diag = np.zeros_like(b)
    b_diag[:mins] = b[:mins]
    b_diag[mins:] = b[mins:] - Q_ba.T @ np.linalg.inv(Q_bb) @ b[:mins]

    # Generate E_diag with num_ncoup non-coupled constraints w.r.t Q_diag
    E_diag = np.random.randn(num_neur, lat_dim)
    if num_ncoup > 0:
        eigvals, eigvecs = np.linalg.eigh(Q_diag)
        pos_eigvecs = eigvecs[:, eigvals > 0]
        for i in range(num_ncoup):
            # Delete the component of E_diag[i, :] along the positive eigenvectors of Q_diag
            E_diag[i, :] = E_diag[i, :] - pos_eigvecs @ (pos_eigvecs.T @ E_diag[i, :])

    if sphere:
        E_diag = E_diag / np.linalg.norm(
            E_diag, axis=1, keepdims=True
        )  # Normalize rows of E_diag
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E_diag, decimals=7), axis=0, return_index=True)
        E_diag = vals[np.argsort(idx)]
        num_neur = E_diag.shape[0]
        Tp_diag = np.ones(num_neur) / 2
    else:
        # Generate random Tp
        # Ensure there are no repeated neurons
        vals, idx = np.unique(np.round(E_diag, decimals=7), axis=0, return_index=True)
        E_diag = vals[np.argsort(idx)]
        num_neur = E_diag.shape[0]
        Tp_diag = np.random.randn(num_neur)

    # Construct E
    E = np.zeros_like(E_diag)
    E[:, :mins] = E_diag[:, :mins]
    E[:, mins:] = E_diag[:, mins:] + E_diag[:, :mins] @ np.linalg.inv(Q_bb) @ Q_ba

    Tp = Tp_diag.copy()

    decoder = lambda x: np.vstack(
        (
            x[:mins, :] - np.linalg.inv(Q_bb) @ Q_ba @ x[mins:, :],
            x[mins:, :],
        )
    )

    return Q, b, E, Tp, Q_diag, b_diag, E_diag, Tp_diag, decoder


def mpc_problem_pv() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate a model predictive control problem instance with
    x_{t+1} = Ak x_t + Bk v_t + Ck w_t

    max_w min_v sum_t v_t^T Nk v_t - w_t^T Rk w_t
    s.t. Ft x_t + Gt v_t + Ht w_t <= It

    We can rewrite this as a maxmin problem by collapsing time
    max_w min_v (w v)^T Q (w v) + b^T (w v)
    s.t. E (w v) <= Tp

    with Q = (-R, 0; 0, N), b = (0 0),
    E = (H + FC, G + FB), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, R, N
    and time-constraint matrices F, G, H, I

    Then we can transform it into a minmax by flipping the signs
    min_w max_v -(w v)^T Q (w v) - b^T (w v)
    s.t. E (w v) <= Tp

    In this case we model a simple (p,v) system


    Parameters
    ----------

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    """

    # Define matrices Ak, Bk, Dk, Qk, Rk, Nk, Ek, Fk, Gk
    Ak = np.array(
        [[1, 0.1], [0, 1]]
    )  # state transition matrix (position and velocity) (s x s)
    Bk = np.array([[0], [0.8]])  # control affects velocity (s x c)
    Ck = np.array([[0], [0.4]])  # disturbance affects velocity (s x d)
    Nk = np.array([[0.5]])  # control cost (c x c)
    Rk = np.array([[0.5]])  # disturbance cost (d x d)

    # Collapse time by transforming into Toeplitz matrices
    T = 5  # time horizon
    s = 2  # state dimension
    c = 1  # control dimension
    d = 1  # disturbance dimension
    # A = (Ak, Ak^2, Ak^3 ... Ak^T) (sT x s)
    A = np.zeros((s * T, s))
    for t in range(T):
        A[s * t : s * (t + 1), :] = np.linalg.matrix_power(Ak, t + 1)
    # B = (Bk, 0, 0, ... ; AkBk, B, 0, ... ; Ak^2Bk, AkBk, Bk, ... ; ...) (sT x cT)
    B = np.zeros((s * T, c * T))
    for t in range(T):
        for tau in range(t + 1):
            B[s * t : s * (t + 1), c * tau : c * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Bk
            )
    # C = (Ck, 0, 0, ... ; AkCk, Ck, 0, ... ; Ak^2Ck, AkCk, Ck, ... ; ...) (sT x dT)
    C = np.zeros((s * T, d * T))
    for t in range(T):
        for tau in range(t + 1):
            C[s * t : s * (t + 1), d * tau : d * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Ck
            )
    # N = (Nk, 0, 0, ... ; 0, Nk, 0, ... ; ... ) (cT x cT)
    N = Nk * np.eye(c * T)
    # R = (Rk, 0, 0, ... ; 0, Rk, 0, ... ; ... ) (dT x dT)
    R = Rk * np.eye(d * T)

    # Time evolution constraints
    # Fk = np.array([[0, 0], [0, 0]])  # state constraints during (cons x s)
    # Gk = np.array([[0], [0]])  # control constraint during (cons x c)
    # Hk = np.array([[0], [0]])  # disturbance constraint during (cons x d)
    # Ik = np.array([0, 0])  # constraint thresholds during (cons x 1)

    # Final constraints
    FT = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1]]
    )  # final state constraints (cons x s)
    cons = FT.shape[0]
    F = np.zeros((cons, s * T))
    F[:, s * (T - 1) : s * T] = FT
    GT = np.array([[0], [0], [0], [0]])  # final control constraints (cons x c)
    G = np.zeros((cons, c * T))
    G[:, c * (T - 1) : c * T] = GT
    HT = np.array([[0], [0], [0], [0]])  # final disturbance constraints (cons x d)
    H = np.zeros((cons, d * T))
    H[:, d * (T - 1) : d * T] = HT

    goal = np.array([1, 0])  # goal state
    delta = 0.1  # tolerance of goal
    IT = np.array(
        [goal[0] + delta, goal[1] + delta, -(goal[0] - delta), -(goal[1] - delta)]
    )  # final constraint thresholds (cons x 1)
    I = np.zeros((cons,))
    I = IT

    # Initial state
    x0 = np.array([0, -1])

    # Compute Q, b, E for the maxmin problem
    Q = np.block([[-R, np.zeros((c * T, d * T))], [np.zeros((d * T, c * T)), N]])
    b = np.zeros((s * T,))
    E = np.block([(H + F @ C), (G + F @ B)])
    Tp = I - F @ A @ x0

    # Flip signs to make it minmax
    Q = -Q
    b = -b

    def decoder(X):
        w = X[:T].reshape(1, -1)
        v = X[T:].reshape(1, -1)
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 2).T
        # Add x0 to the trajectory
        x = np.concatenate((x0[:, None], x), axis=1)
        return x, v, w

    def score(v, w):
        return v.T @ N @ v - w.T @ R @ w, v.T @ N @ v, w.T @ R @ w

    problem = {
        "x0": x0,
        "goal": goal,
        "delta": delta,
        "decoder": decoder,
        "score": score,
    }

    return Q, b, E, Tp, problem


def mpc_problem_2D() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate a model predictive control problem instance with
    x_{t+1} = Ak x_t + Bk v_t + Ck w_t

    max_w min_v sum_t v_t^T Nk v_t - w_t^T Rk w_t
    s.t. Ft x_t + Gt v_t + Ht w_t <= It

    We can rewrite this as a maxmin problem by collapsing time.
    max_w min_v (w v)^T Q (w v) + b^T (w v)
    s.t. E (w v) <= Tp

    with Q = (-R, 0; 0, N), b = (0 0),
    E = (H + FC, G + FB), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, R, N
    and time-constraint matrices F, G, H, I

    Then we can transform it into a minmax by flipping the signs
    min_w max_v -(w v)^T Q (w v) - b^T (w v)
    s.t. E (w v) <= Tp

    In this case we model a 2D system with (p_1, p_2, v_1, v_2)


    Parameters
    ----------

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    """

    # Define matrices Ak, Bk, Dk, Qk, Rk, Nk, Ek, Fk, Gk
    Ak = np.array(
        [[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]]
    )  # state transition matrix (position and velocity) (s x s)
    Bk = np.array(
        [[0, 0], [0, 0], [1.4, 0], [0, 1.4]]
    )  # control affects velocity (s x c)
    Ck = np.array(
        [[0, 0], [0, 0], [0.7, 0], [0, 0.7]]
    )  # disturbance affects velocity (s x d)
    Nk = np.array([[0.5]])  # control cost (c x c)
    Rk = np.array([[0.5]])  # disturbance cost (d x d)

    # Collapse time by transforming into Toeplitz matrices
    T = 5  # time horizon
    s = 4  # state dimension
    c = 2  # control dimension
    d = 2  # disturbance dimension
    # A = (Ak, Ak^2, Ak^3 ... Ak^T) (sT x s)
    A = np.zeros((s * T, s))
    for t in range(T):
        A[s * t : s * (t + 1), :] = np.linalg.matrix_power(Ak, t + 1)
    # B = (Bk, 0, 0, ... ; AkBk, B, 0, ... ; Ak^2Bk, AkBk, Bk, ... ; ...) (sT x cT)
    B = np.zeros((s * T, c * T))
    for t in range(T):
        for tau in range(t + 1):
            B[s * t : s * (t + 1), c * tau : c * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Bk
            )
    # C = (Ck, 0, 0, ... ; AkCk, Ck, 0, ... ; Ak^2Ck, AkCk, Ck, ... ; ...) (sT x dT)
    C = np.zeros((s * T, d * T))
    for t in range(T):
        for tau in range(t + 1):
            C[s * t : s * (t + 1), d * tau : d * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Ck
            )
    # N = (Nk, 0, 0, ... ; 0, Nk, 0, ... ; ... ) (cT x cT)
    N = Nk * np.eye(c * T)
    # R = (Rk, 0, 0, ... ; 0, Rk, 0, ... ; ... ) (dT x dT)
    R = Rk * np.eye(d * T)

    # Time evolution constraints
    cons_k = 0  # number of constraints at each time step
    # Fk = np.array(
    #     [
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [-1, 0, 0, 0],
    #         [0, -1, 0, 0],
    #     ]
    # )  # state constraints during (cons_k x s)
    # Gk = np.array([[0], [0]])  # control constraint during (cons x c)
    # Hk = np.array([[0], [0]])  # disturbance constraint during (cons x d)
    # Ik = np.array([0, 0])  # constraint thresholds during (cons x 1)

    # Final constraints
    cons_T = 8  # number of constraints at final time step
    FT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ]
    )  # final state constraints (cons_T x s)
    cons = cons_k * (T - 1) + cons_T
    F = np.zeros((cons, s * T))
    # for t in range(T):
    #     F[cons_k * t : cons_k * (t + 1), s * t : s * (t + 1)] = Fk
    F[(T - 1) * cons_k : (T - 1) * cons_k + cons_T, s * (T - 1) : s * T] = FT
    GT = np.zeros((cons, c))  # final control constraints (cons x c)
    G = np.zeros((cons, c * T))
    G[:, c * (T - 1) : c * T] = GT
    HT = np.zeros((cons, d))  # final disturbance constraints (cons x d)
    H = np.zeros((cons, d * T))
    H[:, d * (T - 1) : d * T] = HT

    # Obstacle
    # constraint thresholds during (cons_k * (T-1) x 1)
    # x_lims = [0.45, 0.55]
    # y_lims = [-0.1, 0.1]
    # vel = -0.1
    # Ik = np.zeros((cons_k * (T - 1),))
    # for t in range(T - 1):
    #     Ik[cons_k * t : cons_k * (t + 1)] = [
    #         x_lims[1],
    #         y_lims[1] + t * vel,
    #         -x_lims[0],
    #         -(y_lims[0] + t * vel),
    #     ]

    # Goal
    goal = np.array([1, 0, 0, 0])  # goal state
    delta = 0.1  # tolerance of goal
    IT = np.array(
        [
            goal[0] + delta,
            goal[1] + delta,
            -(goal[0] - delta),
            -(goal[1] - delta),
            goal[2] + delta,
            goal[3] + delta,
            -(goal[2] - delta),
            -(goal[3] - delta),
        ]
    )  # final constraint thresholds (cons_T x 1)
    I = np.zeros((cons,))
    I[cons_k * (T - 1) :] = IT

    # Initial state
    x0 = np.array([0, 0, -1, -1])

    # Compute Q, b, E for the maxmin problem
    Q = np.block([[-R, np.zeros((c * T, d * T))], [np.zeros((d * T, c * T)), N]])
    b = np.zeros((s * T,))
    E = np.block([(H + F @ C), (G + F @ B)])
    Tp = I - F @ A @ x0

    # Flip signs to make it minmax
    Q = -Q
    b = -b

    def decoder(X):
        w = X[: 2 * T].reshape(-1, 2).T
        v = X[2 * T :].reshape(-1, 2).T
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 4).T
        # Add x0 to the trajectory
        x = np.concatenate((x0[:, None], x), axis=1)
        return x, v, w

    def score(vf, wf):
        return vf @ N @ vf - wf @ R @ wf, vf.T @ N @ vf, wf.T @ R @ wf

    problem = {
        "x0": x0,
        "goal": goal,
        "delta": delta,
        "decoder": decoder,
        "score": score,
    }

    return Q, b, E, Tp, problem


def mpc_problem_2Dtraj() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate a model predictive control problem instance with
    x_{t+1} = Ak x_t + Bk v_t + Ck w_t

    max_w min_v sum_t v_t^T Nk v_t - w_t^T Rk w_t
    s.t. Ft x_t + Gt v_t + Ht w_t <= It

    We can rewrite this as a maxmin problem by collapsing time.
    max_w min_v (w v)^T Q (w v) + b^T (w v)
    s.t. E (w v) <= Tp

    with Q = (-R, 0; 0, N), b = (0 0),
    E = (H + FC, G + FB), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, R, N
    and time-constraint matrices F, G, H, I

    Then we can transform it into a minmax by flipping the signs
    min_w max_v -(w v)^T Q (w v) - b^T (w v)
    s.t. E (w v) <= Tp

    In this case we model a 2D system with (p_1, p_2, v_1, v_2)
    and force it to do a trajectory


    Parameters
    ----------

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    """
    dt = 0.1
    # Define matrices Ak, Bk, Dk, Qk, Rk, Nk, Ek, Fk, Gk
    Ak = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
    )  # state transition matrix (position and velocity) (s x s)
    Bk = np.array(
        [[0, 0], [0, 0], [1.4, 0], [0, 1.4]]
    )  # control affects velocity (s x c)
    Ck = np.array(
        [[0, 0], [0, 0], [0.7, 0], [0, 0.7]]
    )  # disturbance affects velocity (s x d)
    Nk = np.array([[0.4]])  # control cost (c x c)
    Rk = np.array([[0.5]])  # disturbance cost (d x d)

    # Collapse time by transforming into Toeplitz matrices
    T = 5  # time horizon
    s = 4  # state dimension
    c = 2  # control dimension
    d = 2  # disturbance dimension
    # A = (Ak, Ak^2, Ak^3 ... Ak^T) (sT x s)
    A = np.zeros((s * T, s))
    for t in range(T):
        A[s * t : s * (t + 1), :] = np.linalg.matrix_power(Ak, t + 1)
    # B = (Bk, 0, 0, ... ; AkBk, B, 0, ... ; Ak^2Bk, AkBk, Bk, ... ; ...) (sT x cT)
    B = np.zeros((s * T, c * T))
    for t in range(T):
        for tau in range(t + 1):
            B[s * t : s * (t + 1), c * tau : c * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Bk
            )
    # C = (Ck, 0, 0, ... ; AkCk, Ck, 0, ... ; Ak^2Ck, AkCk, Ck, ... ; ...) (sT x dT)
    C = np.zeros((s * T, d * T))
    for t in range(T):
        for tau in range(t + 1):
            C[s * t : s * (t + 1), d * tau : d * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Ck
            )
    # N = (Nk, 0, 0, ... ; 0, Nk, 0, ... ; ... ) (cT x cT)
    N = Nk * np.eye(c * T)
    # R = (Rk, 0, 0, ... ; 0, Rk, 0, ... ; ... ) (dT x dT)
    R = Rk * np.eye(d * T)

    # Time evolution constraints
    cons_k = 4  # number of constraints at each time step
    Fk = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
        ]
    )  # state constraints during (cons_k x s)
    # Gk = np.array([[0], [0]])  # control constraint during (cons x c)
    # Hk = np.array([[0], [0]])  # disturbance constraint during (cons x d)
    # Ik = np.array([0, 0])  # constraint thresholds during (cons x 1)

    # Final constraints
    cons_T = 4  # number of constraints at final time step
    FT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
        ]
    )  # final state constraints (cons_T x s)
    cons = cons_k * (T - 1) + cons_T
    F = np.zeros((cons, s * T))
    for t in range(T):
        F[cons_k * t : cons_k * (t + 1), s * t : s * (t + 1)] = Fk
    F[(T - 1) * cons_k : (T - 1) * cons_k + cons_T, s * (T - 1) : s * T] = FT
    GT = np.zeros((cons, c))  # final control constraints (cons x c)
    G = np.zeros((cons, c * T))
    G[:, c * (T - 1) : c * T] = GT
    HT = np.zeros((cons, d))  # final disturbance constraints (cons x d)
    H = np.zeros((cons, d * T))
    H[:, d * (T - 1) : d * T] = HT

    # Trajectory
    # constraint thresholds during (cons_k * (T-1) x 1)
    thetas = np.linspace(0, 2 * np.pi, T + 1)
    ptorig = np.array([2, 0])[:, None]
    pts = np.array([np.cos(thetas), np.sin(thetas)]) + ptorig
    delta = 0.1
    Ik = np.zeros((cons_k * (T - 1),))
    for t in range(T - 1):
        Ik[cons_k * t : cons_k * (t + 1)] = [
            pts[0, t + 1] + delta,
            pts[1, t + 1] + delta,
            -(pts[0, t + 1] - delta),
            -(pts[1, t + 1] - delta),
        ]

    # Goal
    goal = np.array([pts[0, 0], pts[1, 0], 0, 0])  # goal state
    IT = np.array(
        [
            goal[0] + delta,
            goal[1] + delta,
            -(goal[0] - delta),
            -(goal[1] - delta),
        ]
    )  # final constraint thresholds (cons_T x 1)
    I = np.zeros((cons,))
    I[: cons_k * (T - 1)] = Ik
    I[cons_k * (T - 1) :] = IT

    # Initial state
    # start at the first point of the trajectory with velocity towards the second point
    x0 = np.array(
        [
            pts[0, 0],
            pts[1, 0],
            (pts[0, 1] - pts[0, 0]) / dt,
            (pts[1, 1] - pts[1, 0]) / dt,
        ]
    )

    # Compute Q, b, E for the maxmin problem
    Q = np.block([[-R, np.zeros((c * T, d * T))], [np.zeros((d * T, c * T)), N]])
    b = np.zeros((s * T,))
    E = np.block([(H + F @ C), (G + F @ B)])
    Tp = I - F @ A @ x0

    # Flip signs to make it minmax
    Q = -Q
    b = -b

    # Get rid of constraints that atuomatically are satisfied for the
    # first time-step
    E = E[4:, :]
    Tp = Tp[4:]

    def decoder(X):
        w = X[: 2 * T].reshape(-1, 2).T
        v = X[2 * T :].reshape(-1, 2).T
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 4).T
        # Add x0 to the trajectory
        x = np.concatenate((x0[:, None], x), axis=1)
        return x, v, w

    def score(vf, wf):
        return vf @ N @ vf - wf @ R @ wf, vf.T @ N @ vf, wf.T @ R @ wf

    problem = {
        "x0": x0,
        "points": pts,
        "goal": goal,
        "delta": delta,
        "decoder": decoder,
        "score": score,
    }

    return Q, b, E, Tp, problem


def mpc_problem_plot_pv(
    vars_op: np.ndarray, vars: np.ndarray, problem: dict, basepath: str
) -> None:
    """
    Plot the trajectory of the MPC problem in state space.
    The system is (p,v)

    Parameters
    ----------
    vars_op : np.ndarray
        The optimal optimization variables (v, w) of shape (2T,).

    vars : np.ndarray
        The optimization variables from the SNN (v, w) of shape (2T,).

    problem : dict
        A dictionary containing the problem parameters including the decoder function.

    basepath : str
        Path where the plot file will be saved.

    Returns
    -------
    None

    """

    def subplot(ax, vars, title):

        # Plot trajectory
        decoder = problem["decoder"]
        x, v, w = decoder(vars)

        ax.plot(x[0, :], x[1, :], marker="o")

        # Plot force vectors
        for t in range(v.shape[1]):
            # plot a vertical arrow with length v
            ax.arrow(
                x[0, t],
                x[1, t],
                0,
                v[0, t],
                head_width=0.01 * np.linalg.norm(v[:, t]),
                head_length=0.2 * np.linalg.norm(v[:, t]),
                fc="blue",
                ec="blue",
                label="Control" if t == 0 else None,
            )
            # plot a vertical arrow  with length w
            ax.arrow(
                x[0, t],
                x[1, t],
                0,
                w[0, t],
                head_width=0.02 * np.linalg.norm(w[:, t]),
                head_length=0.2 * np.linalg.norm(w[:, t]),
                fc="red",
                ec="red",
                label="Disturbance" if t == 0 else None,
            )

        # Plot goal region
        goal = problem["goal"]
        delta = problem["delta"]
        square = mpatches.Rectangle(
            (goal[0] - delta, goal[1] - delta),
            2 * delta,
            2 * delta,
            color="green",
            alpha=0.5,
        )
        ax.add_patch(square)

        # Write score
        score, control_cost, disturbance_cost = problem["score"](
            v.T.flatten(), w.T.flatten()
        )
        ax.text(
            0.05,
            0.95,
            f"Score: {score:.2f}\nControl cost: {control_cost:.2f}\nDisturbance cost: {disturbance_cost:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("MPC Trajectory - " + title)
        ax.grid()
        ax.legend()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    subplot(ax1, vars_op, "Solver")
    subplot(ax2, vars, "SNN")

    # Set axis of plot 2 to be the same as plot 1
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    fig.savefig(basepath + "-mpc.svg", dpi=300, bbox_inches="tight", pad_inches=0.1)


def mpc_problem_plot_2D(
    vars_op: np.ndarray, vars: np.ndarray, problem: dict, basepath: str
) -> None:
    """
    Plot the trajectory of the MPC problem in state space.
    The system is (p_1, p_2, v_1, v_2)

    Parameters
    ----------
    vars_op : np.ndarray
        The optimal optimization variables (v, w) of shape (2T,).

    vars : np.ndarray
        The optimization variables from the SNN (v, w) of shape (2T,).

    problem : dict
        A dictionary containing the problem parameters including the decoder function.

    basepath : str
        Path where the plot file will be saved.

    Returns
    -------
    None

    """

    def subplot(ax, vars, title):

        # Plot trajectory
        decoder = problem["decoder"]
        x, v, w = decoder(vars)

        # Plot trajectory in position space (p_1, p_2)
        ax.plot(x[0, :], x[1, :], marker="o")

        vel = x[2:, :]
        for t in range(vel.shape[1]):
            # Plot an arrow of the velocity vector (v_1, v_2) at each point of the trajectory
            ax.arrow(
                x[0, t],
                x[1, t],
                vel[0, t],
                vel[1, t],
                head_width=0.01 * np.linalg.norm(vel[:, t]),
                head_length=0.2 * np.linalg.norm(vel[:, t]),
                fc="gray",
                ec="gray",
                label="Velocity" if t == 0 else None,
            )

        # Plot force vectors
        for t in range(v.shape[1]):
            # plot a vertical arrow with length v
            ax.arrow(
                x[0, t],
                x[1, t],
                v[0, t],
                v[1, t],
                head_width=0.01 * np.linalg.norm(v[:, t]),
                head_length=0.2 * np.linalg.norm(v[:, t]),
                fc="blue",
                ec="blue",
                label="Control" if t == 0 else None,
            )
            # plot a vertical arrow  with length w
            ax.arrow(
                x[0, t],
                x[1, t],
                w[0, t],
                w[1, t],
                head_width=0.01 * np.linalg.norm(w[:, t]),
                head_length=0.2 * np.linalg.norm(w[:, t]),
                fc="red",
                ec="red",
                label="Disturbance" if t == 0 else None,
            )

        # Plot goal region
        goal = problem["goal"]
        delta = problem["delta"]
        square = mpatches.Rectangle(
            (goal[0] - delta, goal[1] - delta),
            2 * delta,
            2 * delta,
            color="green",
            alpha=0.5,
        )
        ax.add_patch(square)

        # Write score
        score, control_cost, disturbance_cost = problem["score"](
            v.T.flatten(), w.T.flatten()
        )
        ax.text(
            0.05,
            0.95,
            f"Score: {score:.2f}\nControl cost: {control_cost:.2f}\nDisturbance cost: {disturbance_cost:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("MPC Trajectory - " + title)
        ax.grid()
        ax.legend()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    subplot(ax1, vars_op, "Solver")
    subplot(ax2, vars, "SNN")

    # Set axis of plot 2 to be the same as plot 1
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    fig.savefig(basepath + "-mpc2D.svg", dpi=300, bbox_inches="tight", pad_inches=0.1)


def mpc_problem_plot_2Dtraj(
    vars_op: np.ndarray, vars: np.ndarray, problem: dict, basepath: str
) -> None:
    """
    Plot the trajectory of the MPC problem in state space.
    The system is (p_1, p_2, v_1, v_2)
    and we force it to do a trajectory

    Parameters
    ----------
    vars_op : np.ndarray
        The optimal optimization variables (v, w) of shape (2T,).

    vars : np.ndarray
        The optimization variables from the SNN (v, w) of shape (2T,).

    problem : dict
        A dictionary containing the problem parameters including the decoder function.

    basepath : str
        Path where the plot file will be saved.

    Returns
    -------
    None

    """

    def subplot(ax, vars, title):

        # Plot trajectory
        decoder = problem["decoder"]
        x, v, w = decoder(vars)

        # Plot trajectory in position space (p_1, p_2)
        ax.plot(x[0, :], x[1, :], marker="o")

        vel = x[2:, :]
        for t in range(vel.shape[1]):
            # Plot an arrow of the velocity vector (v_1, v_2) at each point of the trajectory
            ax.arrow(
                x[0, t],
                x[1, t],
                vel[0, t],
                vel[1, t],
                head_width=0.01 * np.linalg.norm(vel[:, t]),
                head_length=0.2 * np.linalg.norm(vel[:, t]),
                fc="gray",
                ec="gray",
                label="Velocity" if t == 0 else None,
            )

        # Plot force vectors
        for t in range(v.shape[1]):
            # plot a vertical arrow with length v
            ax.arrow(
                x[0, t],
                x[1, t],
                v[0, t],
                v[1, t],
                head_width=0.01 * np.linalg.norm(v[:, t]),
                head_length=0.2 * np.linalg.norm(v[:, t]),
                fc="blue",
                ec="blue",
                label="Control" if t == 0 else None,
            )
            # plot a vertical arrow  with length w
            ax.arrow(
                x[0, t],
                x[1, t],
                w[0, t],
                w[1, t],
                head_width=0.01 * np.linalg.norm(w[:, t]),
                head_length=0.2 * np.linalg.norm(w[:, t]),
                fc="red",
                ec="red",
                label="Disturbance" if t == 0 else None,
            )

        # Plot goal region
        goal = problem["goal"]
        delta = problem["delta"]
        square = mpatches.Rectangle(
            (goal[0] - delta, goal[1] - delta),
            2 * delta,
            2 * delta,
            color="green",
            alpha=0.5,
        )
        ax.add_patch(square)

        # Write score
        score, control_cost, disturbance_cost = problem["score"](
            v.T.flatten(), w.T.flatten()
        )
        ax.text(
            0.05,
            0.95,
            f"Score: {score:.2f}\nControl cost: {control_cost:.2f}\nDisturbance cost: {disturbance_cost:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("MPC Trajectory - " + title)
        ax.grid()
        ax.legend()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    subplot(ax1, vars_op, "Solver")
    subplot(ax2, vars, "SNN")

    # Set axis of plot 2 to be the same as plot 1
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    fig.savefig(basepath + "-mpc.svg", dpi=300, bbox_inches="tight", pad_inches=0.1)


def mpc_problem_2Ddist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a model predictive control problem instance with
    x_{t+1} = Ak x_t + Bk v_t + Ck w_t

    min_v max_w sum_t x_t^T Pk x_t + v_t^T Rk v_t - w_t^T Nk w_t
    s.t. Fk x_t + Gk v_t + Hk w_t <= Ik

    We can rewrite this as a minmax problem by collapsing time.
    min_v max_w (v w)^T Q (v w) + b^T (v w)
    s.t. E (v w) <= Tp

    with Q = (B^T P B + R, B^T P C; C^T P B, C^T P C - N), b = 2(B^T P A x_0; C^T P A x_0),
    E = (G + FB, H + FC), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, P, R, N, F, G, H, I

    The system is (p_1, p_2, v_1, v_2)
    in this case we measure the cost of the disturbance

    Parameters
    ----------

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    """

    # Define matrices Ak, Bk, Dk, Qk, Rk, Nk, Ek, Fk, Gk
    Ak = np.array(
        [[1, 0, 0.1, 0], [0, 1, 0, 0.1], [0, 0, 1, 0], [0, 0, 0, 1]]
    )  # state transition matrix (position and velocity) (s x s)
    Bk = np.array(
        [[0, 0], [0, 0], [1.4, 0], [0, 1.4]]
    )  # control affects velocity (s x c)
    Ck = np.array(
        [[0, 0], [0, 0], [0.7, 0], [0, 0.7]]
    )  # disturbance affects velocity (s x d)
    Pk = np.array([[0.5]])  # state cost (s x s)
    Nk = np.array([[0.5]])  # control cost (c x c)
    Rk = np.array([[0.5]])  # disturbance cost (d x d)

    # Collapse time by transforming into Toeplitz matrices
    T = 5  # time horizon
    s = 4  # state dimension
    c = 2  # control dimension
    d = 2  # disturbance dimension
    # A = (Ak, Ak^2, Ak^3 ... Ak^T) (sT x s)
    A = np.zeros((s * T, s))
    for t in range(T):
        A[s * t : s * (t + 1), :] = np.linalg.matrix_power(Ak, t + 1)
    # B = (Bk, 0, 0, ... ; AkBk, B, 0, ... ; Ak^2Bk, AkBk, Bk, ... ; ...) (sT x cT)
    B = np.zeros((s * T, c * T))
    for t in range(T):
        for tau in range(t + 1):
            B[s * t : s * (t + 1), c * tau : c * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Bk
            )
    # C = (Ck, 0, 0, ... ; AkCk, Ck, 0, ... ; Ak^2Ck, AkCk, Ck, ... ; ...) (sT x dT)
    C = np.zeros((s * T, d * T))
    for t in range(T):
        for tau in range(t + 1):
            C[s * t : s * (t + 1), d * tau : d * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Ck
            )
    # P = (Pk, 0, 0, ... ; 0, Pk, 0, ... ; ... ) (sT x sT)
    P = Pk * np.eye(s * T)
    # N = (Nk, 0, 0, ... ; 0, Nk, 0, ... ; ... ) (cT x cT)
    N = Nk * np.eye(c * T)
    # R = (Rk, 0, 0, ... ; 0, Rk, 0, ... ; ... ) (dT x dT)
    R = Rk * np.eye(d * T)

    # Time evolution constraints
    cons_k = 0  # number of constraints at each time step
    # Fk = np.array(
    #     [
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [-1, 0, 0, 0],
    #         [0, -1, 0, 0],
    #     ]
    # )  # state constraints during (cons_k x s)
    # Gk = np.array([[0], [0]])  # control constraint during (cons x c)
    # Hk = np.array([[0], [0]])  # disturbance constraint during (cons x d)
    # Ik = np.array([0, 0])  # constraint thresholds during (cons x 1)

    # Final constraints
    cons_T = 8  # number of constraints at final time step
    FT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ]
    )  # final state constraints (cons_T x s)
    cons = cons_k * (T - 1) + cons_T
    F = np.zeros((cons, s * T))
    # for t in range(T):
    #     F[cons_k * t : cons_k * (t + 1), s * t : s * (t + 1)] = Fk
    F[(T - 1) * cons_k : (T - 1) * cons_k + cons_T, s * (T - 1) : s * T] = FT
    GT = np.zeros((cons, c))  # final control constraints (cons x c)
    G = np.zeros((cons, c * T))
    G[:, c * (T - 1) : c * T] = GT
    HT = np.zeros((cons, d))  # final disturbance constraints (cons x d)
    H = np.zeros((cons, d * T))
    H[:, d * (T - 1) : d * T] = HT

    # Obstacle
    # constraint thresholds during (cons_k * (T-1) x 1)
    # x_lims = [0.45, 0.55]
    # y_lims = [-0.1, 0.1]
    # vel = -0.1
    # Ik = np.zeros((cons_k * (T - 1),))
    # for t in range(T - 1):
    #     Ik[cons_k * t : cons_k * (t + 1)] = [
    #         x_lims[1],
    #         y_lims[1] + t * vel,
    #         -x_lims[0],
    #         -(y_lims[0] + t * vel),
    #     ]

    # Goal
    goal = np.array([1, 0, 0, 0])  # goal state
    delta = 0.1  # tolerance of goal
    IT = np.array(
        [
            goal[0] + delta,
            goal[1] + delta,
            -(goal[0] - delta),
            -(goal[1] - delta),
            goal[2] + delta,
            goal[3] + delta,
            -(goal[2] - delta),
            -(goal[3] - delta),
        ]
    )  # final constraint thresholds (cons_T x 1)
    I = np.zeros((cons,))
    I[cons_k * (T - 1) :] = IT

    # Initial state
    x0 = np.array([0, 0, -1, -1])

    # Compute Q, b, E for the minmax problem
    Q = np.block([[B.T @ P @ B + R, B.T @ P @ C], [C.T @ P @ B, C.T @ P @ C - N]])
    b = 2 * np.concatenate([B.T @ P @ A @ x0, C.T @ P @ A @ x0])
    E = np.block([(G + F @ B), (H + F @ C)])
    Tp = I - F @ A @ x0

    def decoder(X):
        v = X[: 2 * T].reshape(-1, 2).T
        w = X[2 * T :].reshape(-1, 2).T
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 4).T
        return x, v, w

    def score(xf, vf, wf):
        return (
            xf @ P @ xf + vf @ N @ vf - wf @ R @ wf,
            xf @ P @ xf,
            vf.T @ N @ vf,
            wf.T @ R @ wf,
        )

    problem = {
        "x0": x0,
        "decoder": decoder,
        "score": score,
    }

    return Q, b, E, Tp


def mpc_problem_ieee118():
    """
    Generate a control problem instance based on the IEEE 118-bus case.

    max_w min_v (w v)^T Q (w v) + b^T (w v)
    s.t. E (w v) <= Tp

    with Q = (-R, 0; 0, N), b = (0 0),

    Then we can transform it into a minmax by flipping the signs
    min_w max_v -(w v)^T Q (w v) - b^T (w v)
    s.t. E (w v) <= Tp

    Here N  is the control cost, R is the disturbance cost
    E and T encodes:
    - Flow limit constraints
    f_max + S @ Pd <= S @ v + S @ w <= f_max + S @ Pd
    - Global conservation of power constraint
    sum(v) + sum(w) = sum(Pd)

    """

    # Load IEEE 118-bus case
    ppc = pp.ext2int(pp.case118())
    # B is the DC susceptance matrix (reduced system)
    B = pp.makeBdc(ppc["baseMVA"], ppc["bus"], ppc["branch"])[0]
    d = B.shape[0]  # number of buses
    # PTDF (Power Transfer Distribution Factors)
    S = pp.makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    # Pd (power demand)
    Pd = ppc["bus"][:, 2]
    # fmax (line flow limits)
    alpha = 50
    f_max = alpha * np.sum(np.abs(S), axis=1)

    # Costs
    N = 0.1 * np.eye(d)  # cost of generation
    R = 0.5 * np.eye(d)  # cost of disturbance
    Q = np.block(
        [
            [-R, np.zeros((d, d))],
            [np.zeros((d, d)), N],
        ]
    )
    b = np.zeros((2 * d,))

    # Constraints
    # Box constraints on generation and disturbance
    # G = np.block([[np.eye(d)], [-np.eye(d)]])
    # H = np.block([[np.eye(d)], [-np.eye(d)]])
    # Eb = np.block([[G, np.zeros((2 * d, d))], [np.zeros((2 * d, d)), H]])
    # lim = 150
    # Tb = np.concatenate([lim * np.ones(2 * d), -lim * np.ones(2 * d)])

    # Flow limit constraint
    Ef = np.block([[S, S], [-S, -S]])
    Tf = np.concatenate([f_max + S @ Pd, f_max - S @ Pd])

    # Global conservation of power constraint
    margin = 1e-2 * np.sum(Pd)
    Eg = np.block([[np.ones((1, 2 * d))], [-np.ones((1, 2 * d))]])
    Tg = np.array([np.sum(Pd) + margin, -np.sum(Pd) + margin])

    # Combine constraints
    E = np.block([[Ef], [Eg]])
    Tp = np.concatenate([Tf, Tg])

    # Flip signs to make it minmax
    Q = -Q
    b = -b

    def decoder(X):
        w = X[:d]
        v = X[d:]
        return v, w

    def score(vf, wf):
        return vf @ N @ vf - wf @ R @ wf, vf.T @ N @ vf, wf.T @ R @ wf

    problem = {
        "decoder": decoder,
        "ppc": ppc,
        "score": score,
    }

    return Q, b, E, Tp, problem


def mpc_problem_plot_ieee118(
    vars_op: np.ndarray, vars: np.ndarray, problem: dict, basepath: str
) -> None:
    """
    Plot the solution of the IEEE 118-bus case control problem.

    Parameters
    ----------
    vars_op : np.ndarray
        The optimal optimization variables (v, w) of shape (2b,).

    vars : np.ndarray
        The optimization variables from the SNN (v, w) of shape (2b,).

    problem : dict
        A dictionary containing the problem parameters including the decoder function.

    basepath : str
        Path where the plot file will be saved.

    Returns
    -------
    None

    """

    def subplot(ax, vars, title):

        # Plot trajectory
        decoder = problem["decoder"]
        ppc = problem["ppc"]
        v, w = decoder(vars)

        # Plot graph
        branches = ppc["branch"]
        from_bus = branches[:, 0].astype(int)
        to_bus = branches[:, 1].astype(int)
        G = nx.Graph()
        n_bus = ppc["bus"].shape[0]
        for i in range(n_bus):
            G.add_node(i)
        for i, j in zip(from_bus, to_bus):
            G.add_edge(i, j)
        pos = nx.kamada_kawai_layout(G, scale=10)

        # Color nodes according to power - demand
        Pd = ppc["bus"][:, 2]
        p = v + w - Pd
        p = (p - np.min(p)) / (np.max(p) - np.min(p))  # normalize to [0, 1]

        # Color connections according to flow
        S = pp.makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        f = S @ (v + w - Pd)
        alpha = 50
        f_max = alpha * np.sum(np.abs(S), axis=1)
        rho = np.abs(f) / (f_max + 1e-9)
        rho = np.clip(rho, 0.1, 1)
        edge_colors = plt.cm.get_cmap("Greys")(rho)

        # Plot graph
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=p,
            edgecolors="black",
            edge_color=edge_colors,
            width=1 + 3 * rho,
            cmap="gray",
            with_labels=False,
            node_size=25,
        )

        def add_triangle(ax, x, y, v, scale=1e-2, pos="right", color="blue"):
            # size scales with magnitude
            s = scale * np.abs(v)

            # direction: inward if v>0, outward if v<0
            direction = (
                -1 if (v > 0 and pos == "right") or (v < 0 and pos == "left") else 1
            )

            # base triangle (pointing right)
            tri = np.array([[0, 0], [-1, 0.25], [-1, -0.25]]) * s * direction

            # shift to node position
            marg = 0.1
            if v < 0 and pos == "right":
                xp = x + s + marg
            elif v < 0 and pos == "left":
                xp = x - s - marg
            elif v > 0 and pos == "right":
                xp = x + marg
            else:
                xp = x - marg

            tri[:, 0] += xp
            tri[:, 1] += y

            ax.fill(tri[:, 0], tri[:, 1], color=color, alpha=0.7, zorder=10)

        for i, (x, y) in pos.items():
            add_triangle(ax, x, y, v[i], pos="right", color="blue")
            add_triangle(ax, x, y, w[i], pos="left", color="red")

        # Write score
        score, control_cost, disturbance_cost = problem["score"](
            v.T.flatten(), w.T.flatten()
        )
        ax.text(
            0.05,
            0.95,
            f"Score: {score:.2f}\nControl cost: {control_cost:.2f}\nDisturbance cost: {disturbance_cost:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("IEEE118 - " + title)
        ax.grid()
        ax.legend()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    subplot(ax1, vars_op, "Solver")
    subplot(ax2, vars, "SNN")

    # Set axis of plot 2 to be the same as plot 1
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    fig.savefig(basepath + "-mpc.svg", dpi=300, bbox_inches="tight", pad_inches=0.1)


def mpc_problem_arm() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate a model predictive control problem instance with
    x_{t+1} = Ak x_t + Bk v_t + Ck w_t

    max_w min_v sum_t v_t^T Nk v_t - w_t^T Rk w_t
    s.t. Ft x_t + Gt v_t + Ht w_t <= It

    We can rewrite this as a maxmin problem by collapsing time.
    max_w min_v (w v)^T Q (w v) + b^T (w v)
    s.t. E (w v) <= Tp

    with Q = (-R, 0; 0, N), b = (0 0),
    E = (H + F @ C, G + F @ B), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, R, N
    and time-constraint matrices F, G, H, I

    Then we can transform it into a minmax by flipping the signs
    min_w max_v -(w v)^T Q (w v) - b^T (w v)
    s.t. E (w v) <= Tp

    In this case we model an arm system with (a_1, a_2, v_1, v_2)
    and force it to do a trajectory


    Parameters
    ----------

    Returns
    -------
    Q : np.ndarray
        Symmetric coupling matrix of shape (l, l).

    b : np.ndarray
        Bias vector of shape (l,).

    E : np.ndarray
        Constraint matrix of shape (N, l).

    Tp : np.ndarray
        Constraint thresholds of shape (N,).

    """
    dt = 0.1

    # Initial state
    # start at the first point of the trajectory with velocity towards the second point
    p0 = np.array(
        [np.pi / 2 - 0.1, 3 * np.pi / 2 + 0.1]
    )  # start at the first point of the trajectory
    x0 = np.array(
        [
            p0[0],  # a_1
            p0[1] - p0[0],  # a_2
            0.0,  # v_1
            0.0,  # v_2
        ]
    )

    # Dynamics of a 2-link arm system
    m1 = 5.0  # mass of first link
    m2 = 5.0  # mass of second link
    l1 = 1.0  # length of first link
    l2 = 1.0  # length of second link

    def M(x0):
        lc1 = l1 / 2
        lc2 = l2 / 2

        I1 = (1 / 12) * m1 * l1**2
        I2 = (1 / 12) * m2 * l2**2

        c2 = np.cos(x0[1])

        M11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2) + I1 + I2
        M12 = m2 * (lc2**2 + l1 * lc2 * c2) + I2
        M22 = m2 * lc2**2 + I2

        return np.array([[M11, M12], [M12, M22]])

    M_inv = np.linalg.inv(M(x0))

    # Define matrices Ak, Bk, Dk, Qk, Rk, Nk, Ek, Fk, Gk
    Ak = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
    )  # state transition matrix (position and velocity) (s x s)
    Bk = np.array(
        [[0, 0], [0, 0], [M_inv[0, 0], M_inv[0, 1]], [M_inv[1, 0], M_inv[1, 1]]]
    )  # control affects velocity (s x c)
    Ck = np.array(
        [[0, 0], [0, 0], [M_inv[0, 0], M_inv[0, 1]], [M_inv[1, 0], M_inv[1, 1]]]
    )  # disturbance affects velocity (s x d)
    Nk = np.array([[0.3]])  # control cost (c x c)
    Rk = np.array([[0.6]])  # disturbance cost (d x d)

    # Collapse time by transforming into Toeplitz matrices
    T = 5  # time horizon
    s = 4  # state dimension
    c = 2  # control dimension
    d = 2  # disturbance dimension
    # A = (Ak, Ak^2, Ak^3 ... Ak^T) (sT x s)
    A = np.zeros((s * T, s))
    for t in range(T):
        A[s * t : s * (t + 1), :] = np.linalg.matrix_power(Ak, t + 1)
    # B = (Bk, 0, 0, ... ; AkBk, B, 0, ... ; Ak^2Bk, AkBk, Bk, ... ; ...) (sT x cT)
    B = np.zeros((s * T, c * T))
    for t in range(T):
        for tau in range(t + 1):
            B[s * t : s * (t + 1), c * tau : c * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Bk
            )
    # C = (Ck, 0, 0, ... ; AkCk, Ck, 0, ... ; Ak^2Ck, AkCk, Ck, ... ; ...) (sT x dT)
    C = np.zeros((s * T, d * T))
    for t in range(T):
        for tau in range(t + 1):
            C[s * t : s * (t + 1), d * tau : d * (tau + 1)] = (
                np.linalg.matrix_power(Ak, t - tau) @ Ck
            )
    # N = (Nk, 0, 0, ... ; 0, Nk, 0, ... ; ... ) (cT x cT)
    N = Nk * np.eye(c * T)
    # R = (Rk, 0, 0, ... ; 0, Rk, 0, ... ; ... ) (dT x dT)
    R = Rk * np.eye(d * T)

    # Time evolution constraints
    cons_k = 4  # number of constraints at each time step
    Fk = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
        ]
    )  # state constraints during (cons_k x s)
    # Gk = np.array([[0], [0]])  # control constraint during (cons x c)
    # Hk = np.array([[0], [0]])  # disturbance constraint during (cons x d)
    # Ik = np.array([0, 0])  # constraint thresholds during (cons x 1)

    # Final constraints
    cons_T = 4  # number of constraints at final time step
    FT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
        ]
    )  # final state constraints (cons_T x s)
    cons = cons_k * (T - 1) + cons_T
    F = np.zeros((cons, s * T))
    for t in range(T):
        F[cons_k * t : cons_k * (t + 1), s * t : s * (t + 1)] = Fk
    F[(T - 1) * cons_k : (T - 1) * cons_k + cons_T, s * (T - 1) : s * T] = FT
    GT = np.zeros((cons, c))  # final control constraints (cons x c)
    G = np.zeros((cons, c * T))
    G[:, c * (T - 1) : c * T] = GT
    HT = np.zeros((cons, d))  # final disturbance constraints (cons x d)
    H = np.zeros((cons, d * T))
    H[:, d * (T - 1) : d * T] = HT

    # Trajectory
    # constraint thresholds during (cons_k * (T-1) x 1)
    thetas1 = np.linspace(p0[0], 0, T)
    thetas2 = np.linspace(p0[1], 2 * np.pi, T) - thetas1
    thetas1 = np.concatenate((p0[0:1], thetas1))
    thetas2 = np.concatenate((p0[1:2] - thetas1[0], thetas2))  # add initial point
    delta = 0.2
    Ik = np.zeros((cons_k * (T - 1),))
    for t in range(T - 1):
        Ik[cons_k * t : cons_k * (t + 1)] = [
            thetas1[t] + delta,
            thetas2[t] + delta,
            -(thetas1[t] - delta),
            -(thetas2[t] - delta),
        ]

    # Goal
    goal = np.array([thetas1[-1], thetas2[-1], 0, 0])  # goal state
    IT = np.array(
        [
            goal[0] + delta,
            goal[1] + delta,
            -(goal[0] - delta),
            -(goal[1] - delta),
        ]
    )  # final constraint thresholds (cons_T x 1)
    I = np.zeros((cons,))
    I[: cons_k * (T - 1)] = Ik
    I[cons_k * (T - 1) :] = IT

    # Compute Q, b, E for the maxmin problem
    Q = np.block([[-R, np.zeros((c * T, d * T))], [np.zeros((d * T, c * T)), N]])
    b = np.zeros((s * T,))
    E = np.block([(H + F @ C), (G + F @ B)])
    Tp = I - F @ A @ x0

    # Flip signs to make it minmax
    Q = -Q
    b = -b

    # Get rid of constraints that atuomatically are satisfied for the
    # first time-step
    E = E[4:, :]
    Tp = Tp[4:]

    def decoder(X):
        w = X[: 2 * T].reshape(-1, 2).T
        v = X[2 * T :].reshape(-1, 2).T
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 4).T
        # Add x0 to the trajectory
        x = np.concatenate((x0[:, None], x), axis=1)
        # Add zeros to the velocity and disturbance
        v = np.concatenate((v, np.zeros((2, 1))), axis=1)
        w = np.concatenate((w, np.zeros((2, 1))), axis=1)
        return x, v, w

    def score(vf, wf):
        return vf @ N @ vf - wf @ R @ wf, vf.T @ N @ vf, wf.T @ R @ wf

    problem = {
        "x0": x0,
        "goal": np.vstack([thetas1, thetas2, np.zeros(T + 1), np.zeros(T + 1)]),
        "delta": delta,
        "decoder": decoder,
        "score": score,
    }

    return Q, b, E, Tp, problem


def mpc_problem_plot_arm(
    vars_op: np.ndarray, vars: np.ndarray, problem: dict, basepath: str
) -> None:
    """
    Plot the trajectory of the MPC problem in state space.
    The system is (a_1, a_2, v_1, v_2)
    and we force it to do a trajectory

    Parameters
    ----------
    vars_op : np.ndarray
        The optimal optimization variables (v, w) of shape (2T,).

    vars : np.ndarray
        The optimization variables from the SNN (v, w) of shape (2T,).

    problem : dict
        A dictionary containing the problem parameters including the decoder function.

    basepath : str
        Path where the plot file will be saved.

    Returns
    -------
    None

    """

    def digest(vars):
        decoder = problem["decoder"]
        q, v, w = decoder(vars)

        # Transform the angles to position
        l1 = 1.0
        l2 = 1.0
        x = np.zeros_like(q)
        x[0, :] = l1 * np.cos(q[0, :]) + l2 * np.cos(q[0, :] + q[1, :])
        x[1, :] = l1 * np.sin(q[0, :]) + l2 * np.sin(q[0, :] + q[1, :])
        # Transform the angular velocities to linear velocities
        x[2, :] = -l1 * np.sin(q[0, :]) * q[2, :] - l2 * np.sin(q[0, :] + q[1, :]) * (
            q[2, :] + q[3, :]
        )
        x[3, :] = l1 * np.cos(q[0, :]) * q[2, :] + l2 * np.cos(q[0, :] + q[1, :]) * (
            q[2, :] + q[3, :]
        )
        # Transform the angular accelerations to linear accelerations
        v_lin = np.zeros_like(v)
        v_lin[0, :] = -l1 * np.sin(q[0, :]) * v[0, :] - l2 * np.sin(
            q[0, :] + q[1, :]
        ) * (v[0, :] + v[1, :])
        v_lin[1, :] = l1 * np.cos(q[0, :]) * v[0, :] + l2 * np.cos(
            q[0, :] + q[1, :]
        ) * (v[0, :] + v[1, :])
        w_lin = np.zeros_like(w)
        w_lin[0, :] = -l1 * np.sin(q[0, :]) * w[0, :] - l2 * np.sin(
            q[0, :] + q[1, :]
        ) * (w[0, :] + w[1, :])
        w_lin[1, :] = l1 * np.cos(q[0, :]) * w[0, :] + l2 * np.cos(
            q[0, :] + q[1, :]
        ) * (w[0, :] + w[1, :])
        return q, x, v, v_lin, w, w_lin

    def draw_circular_arrow(
        ax,
        center,
        radius,
        magnitude,
        color="blue",
        start_angle=0,
        lw=1,
        head_length=None,
        head_width=None,
    ):

        cx, cy = center

        end_angle = start_angle + magnitude

        # Arc
        arc = mpatches.Arc(
            (cx, cy),
            width=2 * radius,
            height=2 * radius,
            theta1=np.degrees(min(start_angle, end_angle)),
            theta2=np.degrees(max(start_angle, end_angle)),
            color=color,
            lw=lw,
        )
        ax.add_patch(arc)

        # Arrowhead location
        x = cx + radius * np.cos(end_angle)
        y = cy + radius * np.sin(end_angle)

        # Tangent direction
        sign = np.sign(magnitude) if magnitude != 0 else 1
        tx = -np.sin(end_angle) * sign
        ty = np.cos(end_angle) * sign

        if head_length is None:
            head_length = radius * 0.18
        if head_width is None:
            head_width = radius * 0.12

        if end_angle != 90:
            ax.arrow(
                x - tx * head_length,
                y - ty * head_length,
                tx * head_length,
                ty * head_length,
                head_width=head_width,
                head_length=head_length,
                fc=color,
                ec=color,
                length_includes_head=True,
            )

    def subplot(ax, vars, title):

        q, x, v, v_lin, w, w_lin = digest(vars)

        # Plot trajectory in position space (p_1, p_2)
        ax.plot(x[0, :], x[1, :], marker="o")

        vel = x[2:, :]
        for t in range(vel.shape[1]):
            # Plot an arrow of the velocity vector (v_1, v_2) at each point of the trajectory
            ax.arrow(
                x[0, t],
                x[1, t],
                vel[0, t],
                vel[1, t],
                head_width=0.01 * np.linalg.norm(vel[:, t]),
                head_length=0.2 * np.linalg.norm(vel[:, t]),
                fc="gray",
                ec="gray",
                label="Velocity" if t == 0 else None,
            )

        # Plot linear force vectors
        for t in range(v.shape[1]):
            # plot a vertical arrow with length v
            ax.arrow(
                x[0, t],
                x[1, t],
                v_lin[0, t],
                v_lin[1, t],
                head_width=0.01 * np.linalg.norm(v_lin[:, t]),
                head_length=0.2 * np.linalg.norm(v_lin[:, t]),
                fc="blue",
                ec="blue",
                label="Control" if t == 0 else None,
            )
            # plot a vertical arrow  with length w
            ax.arrow(
                x[0, t],
                x[1, t],
                w_lin[0, t],
                w_lin[1, t],
                head_width=0.01 * np.linalg.norm(w_lin[:, t]),
                head_length=0.2 * np.linalg.norm(w_lin[:, t]),
                fc="red",
                ec="red",
                label="Disturbance" if t == 0 else None,
            )

        # Plot goal region
        # goal = problem["goal"]
        # delta = problem["delta"]
        # square = mpatches.Rectangle(
        #     (goal[0] - delta, goal[1] - delta),
        #     2 * delta,
        #     2 * delta,
        #     color="green",
        #     alpha=0.5,
        # )
        # ax.add_patch(square)

        # Write score
        score, control_cost, disturbance_cost = problem["score"](
            v[:, :-1].T.flatten(), w[:, :-1].T.flatten()
        )
        ax.text(
            0.05,
            0.95,
            f"Score: {score:.2f}\nControl cost: {control_cost:.2f}\nDisturbance cost: {disturbance_cost:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("MPC Trajectory - " + title)
        ax.grid()
        ax.legend()

    def subplot2(axs, vars, title, linear=False):

        minx, maxx = np.inf, -np.inf
        miny, maxy = np.inf, -np.inf
        for i in range(3):
            for j in range(2):
                t = i * 2 + j
                axes = axs[i][j]
                axes.title.set_text(f"T = {t}")

                q, x, v, v_lin, w, w_lin = digest(vars)
                if t == 0:
                    minx = np.min(x[0, :]) + np.min(
                        np.concatenate((v_lin[0, :], w_lin[0, :]))
                    )
                    maxx = np.max(x[0, :]) + np.max(
                        np.concatenate((v_lin[0, :], w_lin[0, :]))
                    )
                    miny = np.min(x[1, :]) + np.min(
                        np.concatenate((v_lin[1, :], w_lin[1, :]))
                    )
                    maxy = np.max(x[1, :]) + np.max(
                        np.concatenate((v_lin[1, :], w_lin[1, :]))
                    )
                if not np.isnan(minx):
                    if linear:
                        axes.set_xlim(np.min([minx, miny]), np.max([maxx, maxy]))
                        axes.set_ylim(np.min([minx, miny]), np.max([maxx, maxy]))
                        axes.set_aspect("equal", adjustable="box")
                    else:
                        axes.set_xlim(-0.25, 2.1)
                        axes.set_ylim(-0.25, 1.1)
                        axes.set_aspect("equal", adjustable="box")
                T = v.shape[1]

                if t % 2 == 1:
                    axes.tick_params(labelleft=False)  # hide labels, keep ticks
                if t < T - 2:
                    axes.tick_params(labelbottom=False)

                q = q[:, t]
                x = x[:, t]
                v = v[:, t]
                v_lin = v_lin[:, t]
                w = w[:, t]
                w_lin = w_lin[:, t]

                # Plot trajectory in position space (p_1, p_2)
                axes.plot(x[0], x[1], marker="o")

                if linear:
                    vel = x[2:]

                    # Plot an arrow of the velocity vector (v_1, v_2) at each point of the trajectory
                    axes.arrow(
                        x[0],
                        x[1],
                        vel[0],
                        vel[1],
                        head_width=0.01 * np.linalg.norm(vel),
                        head_length=0.2 * np.linalg.norm(vel),
                        fc="gray",
                        ec="gray",
                        label="Velocity" if t == 0 else None,
                    )

                    # Plot linear force vectors
                    axes.arrow(
                        x[0],
                        x[1],
                        v_lin[0],
                        v_lin[1],
                        head_width=0.01 * np.linalg.norm(v_lin),
                        head_length=0.2 * np.linalg.norm(v_lin),
                        fc="blue",
                        ec="blue",
                        label="Control" if t == 0 else None,
                    )
                    # plot a  arrow  with length w
                    axes.arrow(
                        x[0],
                        x[1],
                        w_lin[0],
                        w_lin[1],
                        head_width=0.01 * np.linalg.norm(w_lin),
                        head_length=0.2 * np.linalg.norm(w_lin),
                        fc="red",
                        ec="red",
                        label="Disturbance" if t == 0 else None,
                    )

                # Plot goal region

                # goal = problem["goal"]
                # goal = goal[:, t]
                # goal_lin = np.zeros((2,))
                # goal_lin[0] = l1 * np.cos(goal[0]) + l2 * np.cos(goal[0] + goal[1])
                # goal_lin[1] = l1 * np.sin(goal[0]) + l2 * np.sin(goal[0] + goal[1])
                # delta = problem["delta"]
                # square = mpatches.Rectangle(
                #     (goal_lin[0] - delta, goal_lin[1] - delta),
                #     2 * delta,
                #     2 * delta,
                #     color="green",
                #     alpha=0.5,
                # )
                # axes.add_patch(square)

                # Plot arm
                l1 = 1.0
                l2 = 1.0
                # Circle at joint 1
                base = mpatches.Circle((0, 0), radius=0.05, color="black", zorder=10)
                axes.add_patch(base)
                # Torque at joint 1
                draw_circular_arrow(
                    axes,
                    (0, 0),
                    radius=0.15,
                    magnitude=v[0],
                    color="blue",
                    start_angle=90,
                )
                draw_circular_arrow(
                    axes,
                    (0, 0),
                    radius=0.12,
                    magnitude=w[0],
                    color="red",
                    start_angle=90,
                )
                draw_circular_arrow(
                    axes,
                    (0, 0),
                    radius=0.09,
                    magnitude=q[2],
                    color="grey",
                    start_angle=90,
                )
                # Circle at joint 2
                joint2pos = (
                    x[0] - l2 * np.cos(q[0] + q[1]),
                    x[1] - l2 * np.sin(q[0] + q[1]),
                )
                joint2 = mpatches.Circle(
                    joint2pos,
                    radius=0.05,
                    color="black",
                    zorder=10,
                )
                axes.add_patch(joint2)
                # Torque at joint 2
                draw_circular_arrow(
                    axes,
                    joint2pos,
                    radius=0.15,
                    magnitude=v[1],
                    color="blue",
                    start_angle=90,
                )
                draw_circular_arrow(
                    axes,
                    joint2pos,
                    radius=0.12,
                    magnitude=w[1],
                    color="red",
                    start_angle=90,
                )
                draw_circular_arrow(
                    axes,
                    joint2pos,
                    radius=0.09,
                    magnitude=q[3],
                    color="grey",
                    start_angle=90,
                )
                # Segment connecting joint 1 to joint 2
                axes.plot(
                    [0, joint2pos[0]],
                    [0, joint2pos[1]],
                    color="black",
                    linewidth=2,
                )
                # Segment connecting joint 2 to end effector
                axes.plot(
                    [joint2pos[0], x[0]],
                    [joint2pos[1], x[1]],
                    color="black",
                    linewidth=2,
                )

                if t == T - 1:
                    axes.legend()

                # axes.set_title("MPC Trajectory - " + title)
                axes.grid()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    subplot(ax1, vars_op, "Solver")
    subplot(ax2, vars, "SNN")

    # Set axis of plot 2 to be the same as plot 1
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    fig.savefig(basepath + "-mpc.svg", dpi=300, bbox_inches="tight", pad_inches=0.1)

    fig = plt.figure(figsize=(12, 6))
    outer = fig.add_gridspec(1, 2, wspace=0.25)
    # Left panel: 3x2
    left = outer[0].subgridspec(3, 2, hspace=-0.15, wspace=0.05)
    # Right panel: 3x2
    right = outer[1].subgridspec(3, 2, hspace=-0.15, wspace=0.05)
    # Create axes
    axs_left = [[fig.add_subplot(left[i, j]) for j in range(2)] for i in range(3)]
    axs_right = [[fig.add_subplot(right[i, j]) for j in range(2)] for i in range(3)]

    subplot2(axs_left, vars_op, "Solver")
    fig.text(0.27, 0.98, "Solver", ha="center", va="top", fontsize=14)
    # Write score
    q, x, v, v_lin, w, w_lin = digest(vars_op)
    score, control_cost, disturbance_cost = problem["score"](
        v[:, :-1].T.flatten(), w[:, :-1].T.flatten()
    )
    fig.text(
        0.22,
        0.95,
        f"Score: {score:.2f}\nControl cost: {control_cost:.2f}\nDisturbance cost: {disturbance_cost:.2f}",
        transform=fig.transFigure,
        verticalalignment="top",
    )

    subplot2(axs_right, vars, "SNN")
    fig.text(0.73, 0.98, "SNN", ha="center", va="top", fontsize=14)
    # Write score
    q, x, v, v_lin, w, w_lin = digest(vars)
    score, control_cost, disturbance_cost = problem["score"](
        v[:, :-1].T.flatten(), w[:, :-1].T.flatten()
    )
    fig.text(
        0.68,
        0.95,
        f"Score: {score:.2f}\nControl cost: {control_cost:.2f}\nDisturbance cost: {disturbance_cost:.2f}",
        transform=fig.transFigure,
        verticalalignment="top",
    )

    fig.savefig(
        basepath + "-mpc_comic.svg", dpi=300, bbox_inches="tight", pad_inches=0.1
    )
