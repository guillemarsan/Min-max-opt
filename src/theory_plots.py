import time as time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from SCN import Low_rank_LIF, Simulation
from SCN.utils_neuro import _canon_symmetric

# Type of optimization to plot
# cos = ["minmax", "canminmax", "QP", "canQP", "dalian"]
cos = ["minmax"]
dir = "./data/theory_plots/"


def gradient_line(y: np.ndarray, forget: bool = False) -> LineCollection:

    fsteps = 1000
    if y.shape[0] == 2:
        points = np.array([y[0, :], y[1, :]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(list(segments), colors="black", norm=Normalize(0, 1))
    else:
        points = np.array([y[0, :], y[1, :], y[2, :]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(list(segments), linewidths=2, norm=Normalize(0, 1))
        lc._segments3d = segments  # type: ignore

    alphas = (
        np.concatenate([np.zeros(y.shape[1] - fsteps), np.linspace(0, 1, fsteps)])
        if forget and y.shape[1] > fsteps
        else np.linspace(0, 1, y.shape[1])
    )

    if y.shape[0] == 2:
        lc.set_alpha(list(alphas))
        lc.set_linewidth(2)
    elif y.shape[0] == 3:
        colors = np.zeros((len(segments), 4))  # Create an array for RGBA colors
        colors[:, :3] = 0  # Set RGB to black
        for i in range(len(segments)):
            colors[i, -1] = alphas[i]  # Set the alpha channel
        lc.set_color([tuple(color) for color in colors])

    return lc


def plot_traj(
    ax: Axes,
    traj: np.ndarray,
    gradient: bool = True,
) -> None:

    if gradient:
        final = ax.scatter(
            traj[0, -1], traj[1, -1], edgecolor="black", facecolors="white", zorder=10
        )
        middle = gradient_line(traj)
        ax.add_collection(middle)
        ini = ax.scatter(traj[0, 0], traj[1, 0], edgecolor="grey", facecolors="none")

    else:

        final = ax.scatter(
            traj[0, -1], traj[1, -1], edgecolor="blue", facecolors="white", zorder=10
        )
        middle = ax.plot(traj[0, :], traj[1, :], color="blue", alpha=0.5)[0]
        ini = ax.scatter(traj[0, 0], traj[1, 0], edgecolor="blue", facecolors="none")


def plot2D(Q, b, E, Tp, pts, D, y, y_op, y_op_lim, extrapoint=None, name="test"):

    fig = plt.figure()
    ax = fig.gca()

    vleng = 2

    # Create a grid of x and y values
    y1 = np.linspace(-vleng, vleng, 1000)
    y2 = np.linspace(-vleng, vleng, 1000)
    Y1, Y2 = np.meshgrid(y1, y2)

    # Evaluate and contours
    Z = (
        1 / 2 * (Q[0, 0] * Y1**2 + 2 * Q[0, 1] * Y1 * Y2 + Q[1, 1] * Y2**2)
        + b[0] * Y1
        + b[1] * Y2
    )
    opt = (
        1
        / 2
        * (
            Q[0, 0] * y_op_lim[0] ** 2
            + 2 * Q[0, 1] * y_op_lim[0] * y_op_lim[1]
            + Q[1, 1] * y_op_lim[1] ** 2
        )
        + b[0] * y_op_lim[0]
        + b[1] * y_op_lim[1]
    )
    contour_levels = np.linspace(-2, 2, 11)  # Adjust the number of contour levels here
    contour_levels = np.append(contour_levels, [opt])
    contour_levels = np.sort(contour_levels)
    cont = ax.contour(Y1, Y2, Z, levels=contour_levels, cmap="coolwarm", linewidths=1)

    # Unfeasible region
    # Region where both inequalities hold
    slack = 0
    mask = (
        lambda y1, y2: (E[0, 0] * y1 + E[0, 1] * y2 <= Tp[0] + slack)
        & (E[1, 0] * y1 + E[1, 1] * y2 <= Tp[1] + slack)
        & (E[2, 0] * y1 + E[2, 1] * y2 <= Tp[2] + slack)
    )
    feasible = mask(Y1, Y2).astype(float)
    feasible[np.where(feasible == 1)] = np.nan
    ax.contourf(Y1, Y2, feasible, levels=0, colors=["lightgrey"], alpha=0.5)

    mask1 = lambda y1, y2: (E[1, 0] * y1 + E[1, 1] * y2 <= Tp[1] + slack) & (
        E[2, 0] * y1 + E[2, 1] * y2 <= Tp[2] + slack
    )
    mask2 = lambda y1, y2: (E[0, 0] * y1 + E[0, 1] * y2 <= Tp[0] + slack) & (
        E[2, 0] * y1 + E[2, 1] * y2 <= Tp[2] + slack
    )
    mask3 = lambda y1, y2: (E[0, 0] * y1 + E[0, 1] * y2 <= Tp[0] + slack) & (
        E[1, 0] * y1 + E[1, 1] * y2 <= Tp[1] + slack
    )
    masks = [mask1, mask2, mask3]
    # Constraints
    colors = ["g", "b", "r"]
    for i in range(E.shape[0]):
        # Compute the line corresponding to the constraint
        y2_constr = (Tp[i] - E[i, 0] * y1) / E[i, 1]
        mask_vec = masks[i](y1, y2_constr)
        y1_masked = y1[mask_vec]
        y2_constr_masked = y2_constr[mask_vec]
        ax.plot(y1_masked, y2_constr_masked, label=f"Constraint {i+1}", color=colors[i])

    # Decoder arrows
    for i in range(D.shape[1]):
        ax.quiver(
            pts[i, 0],
            pts[i, 1],
            D[0, i],
            D[1, i],
            angles="xy",
            scale_units="xy",
            scale=1,
            zorder=10,
            color=colors[i],
        )

    # Plot the traj
    plot_traj(ax, y, gradient=True)
    ax.quiver(
        y[0, -1],
        y[1, -1],
        (-y[0, -1] - (np.linalg.inv(Q) @ b)[0]) / 2,
        (-y[1, -1] - (np.linalg.inv(Q) @ b)[1]) / 2,
        scale=2,
        scale_units="xy",
        angles="xy",
        color="grey",
        alpha=1,
        zorder=10,
    )

    # Plot the optimal points
    # Black contour with white filling
    ax.scatter(
        y_op_lim[0],
        y_op_lim[1],
        c="white",
        edgecolors="black",
        marker="o",
        label="Optimal point",
        zorder=8,
    )
    ax.scatter(
        y_op[0],
        y_op[1],
        c="white",
        edgecolors="black",
        marker="D",
        label="Optimal point without constraints",
        zorder=8,
    )

    if extrapoint is not None:
        ax.scatter(
            extrapoint[0],
            extrapoint[1],
            c="white",
            edgecolors="grey",
            marker="o",
            label="Extrapolated point",
            zorder=8,
        )
        ax.vlines(
            extrapoint[0], -vleng, vleng, color="grey", linestyle="--", linewidth=1
        )
        ax.hlines(
            extrapoint[1], -vleng, vleng, color="grey", linestyle="--", linewidth=1
        )
        ax.vlines(
            y_op_lim[0], -vleng, vleng, color="black", linestyle="--", linewidth=1
        )
        ax.hlines(
            y_op_lim[1], -vleng, vleng, color="black", linestyle="--", linewidth=1
        )

    # Set the axes to pass through (0, 0)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Remove the top and right spines
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Add arrows to the spines by drawing triangle shaped points over them
    ax.plot(1, 0.5, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0.5, 1, "^k", transform=ax.transAxes, clip_on=False)
    ax.set_aspect("equal")
    ax.set_xlim(-vleng - 0.1, vleng + 0.1)
    ax.set_ylim(-vleng - 0.1, vleng + 0.1)
    tick_vals = np.arange(-vleng, vleng + 1, 1)
    tick_vals = np.delete(tick_vals, np.where(tick_vals == 0))
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    ax.grid(True)

    fig.savefig(name + ".svg", dpi=300, bbox_inches="tight", pad_inches=0.1)

    return cont


def plot_subproblem(Q, b, opta, optb, cont, name):

    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    axes = [ax1, ax2]

    y2valarr_l = lambda y2: (
        1 / 2 * (Q[0, 0] * y1valarr**2 + Q[1, 1] * y2**2 + 2 * Q[0, 1] * y1valarr * y2)
        + b[0] * y1valarr
        + b[1] * y2
    )

    for i in range(2):
        vleng = 2
        y1valarr = np.linspace(-vleng, vleng, 200)
        y2valarr = y2valarr_l(opta[1] if i == 0 else optb[1])
        points = np.array([y1valarr, y2valarr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(list(segments), cmap=cont.cmap, norm=cont.norm)
        lc.set_array(y2valarr)  # color by y-values
        lc.set_linewidth(2)
        axes[i].add_collection(lc)

        cut1 = (E[0, 1] * opta[1] - Tp[0]) / -E[0, 0]
        axes[i].vlines(cut1, -1, 4, color="g", linestyle="-", linewidth=2)

        axes[i].axvspan(xmin=-vleng, xmax=cut1, facecolor="grey", alpha=0.2, zorder=0)
        axes[i].set_xlim(-vleng, vleng)
        axes[i].set_ylim(-1, 4)

        axes[i].set_aspect("equal")

    fig.savefig(name + "_sub.svg", dpi=300, bbox_inches="tight", pad_inches=0.1)


for co in cos:

    # Parameters of the constraints
    E = np.array([[-1, -1 / 3], [-1, -0.1], [0, -1]])
    pts = np.array([[0.35, 0.5], [0.15, 1.5], [1.5, -1.1]])
    Tp = np.array([-0.5, -0.3, 1.1])
    opta = None

    # Parameters of the problem
    if co == "canQP":
        Q = np.array([[1, 0], [0, 1]])
        b = np.array([0, 0])
        y0 = np.array([1, 1])
        centergeom = np.array([0.5, 0.5])
    elif co == "QP":
        Q = np.array([[1 / 2, -1 / 2], [-1 / 2, 2]])
        b = np.array([-1, 3])
        A, S = _canon_symmetric(Q)
        fact = S @ np.linalg.inv(A.T) @ b
        Tp = Tp - E @ fact
        E = E @ A
        pts = (np.linalg.inv(A) @ (pts - fact).T).T
        y0 = np.linalg.inv(A) @ (np.array([1, 1]) - fact)
        centergeom = np.array([0, -1])
    elif co == "canminmax":
        Q = np.array([[1, 0], [0, -1]])
        b = np.array([0, 0])
        y0 = np.array([0.3, 1.5])
        centergeom = np.array([0.5, 0.5])
        xa = 0.25
        ya = (-E[0, 0] * xa + Tp[0]) / E[0, 1]
        opta = np.array([xa, ya])
    elif co == "minmax":
        Q = np.array([[1 / 2, 2], [2, -2]])
        b = np.array([-1, -1])
        A, S = _canon_symmetric(Q)
        fact = S @ np.linalg.inv(A.T) @ b
        Tp = Tp - E @ fact
        E = E @ A
        pts = (np.linalg.inv(A) @ (pts - fact).T).T
        y0 = np.linalg.inv(A) @ (np.array([0.3, 1.5]) - fact)
        centergeom = np.array([0, 0])
    elif co == "dalian":
        E = np.array([[-3 / 4, -1 / 4], [0, -1 / np.sqrt(2)]])
        Tp = np.array([0, -0.5])  # np.array([-0.5, 0])
        Q = np.array([[1, 0], [0, -1]])
        b = np.array([0, 0])
        y0 = np.array([0.3, 1.5])
        centergeom = np.array([0.5, 0.5])
    elif co == "dalianinv":
        E = np.array([[-3 / 4, 1 / 4], [0, -1 / np.sqrt(2)]])
        Tp = np.array([0, -0.5])  # np.array([-0.5, 0])
        Q = np.array([[1, 0], [0, -1]])
        b = np.array([0, 0])
        y0 = np.array([0.3, 1.5])
        centergeom = np.array([0.5, 0.5])
    elif co == "notsufficient":
        E = np.array([[-1 / 4, -3 / 4], [0, -1 / np.sqrt(2)]])
        Tp = np.array([-0.2, -0.1])  # np.array([-0.5, 0])
        Q = np.array([[1, 0], [0, -1]])
        b = np.array([0, 0])
        y0 = np.array([0.3, 1.5])
        centergeom = np.array([0.5, 0.5])
    else:
        E = np.array([[-3 / 4, -1 / 4 - 0.5], [0, -1 / np.sqrt(2)]])
        Tp = np.array([0, -0.5])  # np.array([-0.5, 0])
        Q = np.array([[1, 0], [0, -1]])
        b = np.array([0, 0])
        y0 = np.array([0.3, 1.5])
        centergeom = np.array([0.5, 0.5])

    # initialize simulation
    sim = Simulation.init_optim(
        Q=Q,
        b=b,
        E=E,
        Tp=Tp,
        spike_scale=1,
        y0=y0,
        Tmax=5,
        tag=co,
    )

    # run simulation
    sim.run(
        draw_break="one",
        criterion="inh_max",
    )

    # optimize simulation
    sim.optimize(Q=Q)

    # plot
    sim.plot(centergeom=centergeom)

    # plot latent space pretty
    cont = plot2D(
        Q,
        b,
        E,
        Tp,
        pts,
        sim.net.D,
        sim.y,
        sim.y_op[:, -1],
        sim.y_op_lim[:, -1],
        extrapoint=opta,
        name=dir + co,
    )

    if co == "canminmax":
        plot_subproblem(
            Q,
            b,
            opta,
            sim.y_op_lim[:, -1],
            cont,
            name=dir + co,
        )
