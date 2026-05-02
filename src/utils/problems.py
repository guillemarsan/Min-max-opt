import os
from typing import Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


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
        pos_eigvecs = eigvecs[:, eigvals > 0]
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
    Q = np.block([[Q_bb, Q_ba], [Q_ba.T, Q_aa]])

    # Generate random b
    b = np.random.randn(lat_dim)

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

    We can rewrite this as a maxmin problem by collapsing time.
    max_w min_v (v w)^T Q (v w) + b^T (v w)
    s.t. E (v w) <= Tp

    with Q = (N, 0; 0, -R), b = (0 0),
    E = (G + FB, H + FC), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, R, N
    and time-constraint matrices F, G, H, I

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

    # Compute Q, b, E for the minmax problem
    Q = np.block([[N, np.zeros((c * T, d * T))], [np.zeros((d * T, c * T)), -R]])
    b = np.zeros((s * T,))
    E = np.block([(G + F @ B), (H + F @ C)])
    Tp = I - F @ A @ x0

    def decoder(X):
        v = X[:T].reshape(1, -1)
        w = X[T:].reshape(1, -1)
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 2).T
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
    max_w min_v (v w)^T Q (v w) + b^T (v w)
    s.t. E (v w) <= Tp

    with Q = (N, 0; 0, -R), b = (0 0),
    E = (G + FB, H + FC), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, R, N
    and time-constraint matrices F, G, H, I

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

    # Compute Q, b, E for the minmax problem
    Q = np.block([[N, np.zeros((c * T, d * T))], [np.zeros((d * T, c * T)), -R]])
    b = np.zeros((s * T,))
    E = np.block([(G + F @ B), (H + F @ C)])
    Tp = I - F @ A @ x0

    def decoder(X):
        v = X[: 2 * T].reshape(-1, 2).T
        w = X[2 * T :].reshape(-1, 2).T
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 4).T
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
    max_w min_v (v w)^T Q (v w) + b^T (v w)
    s.t. E (v w) <= Tp

    with Q = (N, 0; 0, -R), b = (0 0),
    E = (G + FB, H + FC), Tp = I - F A x_0
    for some initial state x_0 and Toeplitz matrices A, B, C, R, N
    and time-constraint matrices F, G, H, I

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

    # Compute Q, b, E for the minmax problem
    Q = np.block([[N, np.zeros((c * T, d * T))], [np.zeros((d * T, c * T)), -R]])
    b = np.zeros((s * T,))
    E = np.block([(G + F @ B), (H + F @ C)])
    Tp = I - F @ A @ x0

    # Get rid of constraints that atuomatically are satisfied for the
    # first time-step
    E = E[4:, :]
    Tp = Tp[4:]

    def decoder(X):
        v = X[: 2 * T].reshape(-1, 2).T
        w = X[2 * T :].reshape(-1, 2).T
        x = A @ x0 + B @ v.T.flatten() + C @ w.T.flatten()
        x = x.reshape(-1, 4).T
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
        # Add x0 to the trajectory
        x = np.concatenate((problem["x0"][:, None], x), axis=1)
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
        # Add x0 to the trajectory
        x = np.concatenate((problem["x0"][:, None], x), axis=1)

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
        # Add x0 to the trajectory
        x = np.concatenate((problem["x0"][:, None], x), axis=1)

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
