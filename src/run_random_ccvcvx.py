import argparse
import json
import random
import string
import time

import numpy as np
from scipy.signal import lfilter
from SCN import Simulation

from utils.problems import random_problem_ccvcvx

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument(
        "--lat_dim", type=int, default=5, help="Dimensionality of latents"
    )
    parser.add_argument(
        "--num_neur", type=int, default=20, help="Number of constraints/neurons"
    )
    parser.add_argument(
        "--num_ncoup",
        type=int,
        default=8,
        help="Number of non-coupled constraints/neurons",
    )
    parser.add_argument(
        "--spike_scale",
        type=float,
        default=0.01,
        help="Spike scale for the simulation",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=113,
        help="Random seed for the sampling of parameters",
    )
    parser.add_argument(
        "--SNN_init",
        type=str,
        default="Solver",
        help="SNN initialization mode: 'SNN' for SNN-initialized, 'Solver' for solver-initialized, 'Indep' for independent",
    )

    parser.add_argument(
        "--dir", type=str, default="./data/test/", help="Directory to dump output"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Plot the results"
    )
    parser.add_argument(
        "--save_sim", action="store_true", default=False, help="Save the simulation"
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=True,
        help="Save the simulation outputs",
    )
    parser.add_argument(
        "--save_config",
        action="store_true",
        default=False,
        help="Save the script configuration",
    )

    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    code = "".join(random.choice(string.ascii_letters) for i in range(5))
    name = (
        timestr
        + "-"
        + code
        + "-l-"
        + str(args.lat_dim)
        + "-n-"
        + str(args.num_neur)
        + "-nc-"
        + str(args.num_ncoup)
        + "-sp-"
        + str(args.spike_scale)
        + "-s-"
        + str(args.rand_seed)
    )
    basepath = args.dir + name

    # PROBLEM PARAMETERS
    Q, b, E, Tp, Q_diag, b_diag, E_diag, Tp_diag, decoder = random_problem_ccvcvx(
        args.lat_dim, args.num_neur, args.num_ncoup, args.rand_seed, sphere=True
    )

    if args.SNN_init == "SNN":
        # SNN first and then solver where SNN ends
        # Initialize simulation
        sim = Simulation.init_optim(
            Q=Q_diag,
            b=b_diag,
            E=E_diag,
            Tp=Tp_diag,
            spike_scale=args.spike_scale,
            Tmax=10,
            y0=np.zeros(args.lat_dim),
        )

        # RUN SIMULATION
        # run simulation
        sim.run()

        # optimize simulation
        sim.optimize(Q=Q_diag)
    elif args.SNN_init == "Solver":
        # Solver first and then SNN where solver ends
        # Initialize simulation
        sim_opt = Simulation.init_optim(
            Q=Q_diag,
            b=b_diag,
            E=E_diag,
            Tp=Tp_diag,
            spike_scale=args.spike_scale,
            Tmax=10,
            y0=np.zeros(args.lat_dim),
        )

        # optimize simulation
        sim_opt.optimize(Q=Q_diag)
        solver_fail = bool(np.any(np.isnan(sim_opt.y_op_lim)))

        # RUN SIMULATION
        # Initialize simulation with correct initial point
        sim = Simulation.init_optim(
            Q=Q_diag,
            b=b_diag,
            E=E_diag,
            Tp=Tp_diag,
            spike_scale=args.spike_scale,
            Tmax=10,
            y0=(
                0.999 * sim_opt.y_op_lim[:, -1]
                if not solver_fail
                else np.zeros(args.lat_dim)
            ),
        )
        # run simulation
        sim.run()

        # transfer data
        sim.y_op = sim_opt.y_op
        sim.y_op_lim = sim_opt.y_op_lim
        sim.r_op = sim_opt.r_op
        sim.r_op_lim = sim_opt.r_op_lim
    else:
        # Solver first and then SNN independently
        # Initialize simulation
        sim_opt = Simulation.init_optim(
            Q=Q_diag,
            b=b_diag,
            E=E_diag,
            Tp=Tp_diag,
            spike_scale=args.spike_scale,
            Tmax=10,
            y0=np.zeros(args.lat_dim),
        )

        # optimize simulation
        sim_opt.optimize(Q=Q_diag)

        # RUN SIMULATION
        # Initialize simulation
        sim = Simulation.init_optim(
            Q=Q_diag,
            b=b_diag,
            E=E_diag,
            Tp=Tp_diag,
            spike_scale=args.spike_scale,
            Tmax=10,
            y0=np.zeros(args.lat_dim),
        )
        # run simulation
        sim.run()

        # transfer data
        sim.y_op = sim_opt.y_op
        sim.y_op_lim = sim_opt.y_op_lim
        sim.r_op = sim_opt.r_op
        sim.r_op_lim = sim_opt.r_op_lim

    # match check
    y_op_lim_from_r_op_lim = (
        sim.net.D @ sim.r_op_lim[:, -1] - np.linalg.inv(Q_diag) @ b_diag
    )
    match = bool(np.linalg.norm(sim.y_op_lim[:, -1] - y_op_lim_from_r_op_lim) < 5e-1)

    # Plot
    if args.plot:
        sim.plot()

        def estimate_energy(nspikes, neurs):
            energy_per_syn = 30  # pJ/synapse
            energy_est = nspikes * neurs * energy_per_syn
            print("Estimated energy consumption (pJ):", energy_est)

        def estimate_convergence(x):
            running_mean = np.cumsum(x, axis=1) / np.arange(1, x.shape[1] + 1)
            delta = np.diff(running_mean, axis=1)
            rel_change = np.linalg.norm(delta, axis=0) / np.maximum(
                np.linalg.norm(running_mean[:, :-1], axis=0), 1e-8
            )
            tol = 1e-4
            window_size = 10
            good = rel_change < tol
            conv = (
                np.convolve(good, np.ones(window_size, dtype=int), "valid")
                == window_size
            )
            if np.any(conv):
                idx = np.argmax(conv) + window_size
                print("Convergence achieved at index:", idx)
            else:
                print("No convergence achieved.")

        estimate_energy(sim.stimes.shape[0], sim.r.shape[0])
        estimate_convergence(sim.y)

    # Transform back to original space
    sim.y = decoder(sim.y)

    sim.y_op = decoder(sim.y_op)
    sim.y_op_lim = decoder(sim.y_op_lim)

    ## RESULTS
    results = {}

    if args.save_sim:
        # Save simulation
        print("Saving simulation...")
        sim.save(args.dir, name)

    if args.save_outputs:
        # Save results
        if hasattr(sim, "y"):
            np.savetxt("%s-y.csv" % basepath, sim.y, fmt="%.3e")
        if hasattr(sim, "y_op"):
            np.savetxt("%s-y_op.csv" % basepath, sim.y_op[:, -1], fmt="%.3e")
        if hasattr(sim, "y_op_lim"):
            np.savetxt("%s-y_op_lim.csv" % basepath, sim.y_op_lim[:, -1], fmt="%.3e")

        if hasattr(sim, "r"):
            np.savetxt("%s-r.csv" % basepath, sim.r, fmt="%.3e")
        if hasattr(sim, "r_op"):
            np.savetxt("%s-r_op.csv" % basepath, sim.r_op[:, -1], fmt="%.3e")
        if hasattr(sim, "r_op_lim"):
            np.savetxt("%s-r_op_lim.csv" % basepath, sim.r_op_lim[:, -1], fmt="%.3e")

        if hasattr(sim, "stimes"):
            np.savetxt("%s-stimes.csv" % basepath, sim.stimes, fmt="%.3e")
        if hasattr(sim, "V"):
            np.savetxt("%s-V.csv" % basepath, sim.V, fmt="%.3e")

    if args.save_config:
        results["name"] = name
        results["dir"] = args.dir
        results["basepath"] = basepath
        results["args"] = vars(args)
        results["Q"] = Q.tolist()
        results["Q_diag"] = Q_diag.tolist()
        results["b"] = b.tolist()
        results["b_diag"] = b_diag.tolist()
        results["E"] = E.tolist()
        results["E_diag"] = E_diag.tolist()
        results["D"] = sim.net.D.tolist()
        results["Tp"] = Tp.tolist()
        results["Tp_diag"] = Tp_diag.tolist()
        results["spike_scale"] = args.spike_scale
        results["Tmax"] = sim.Tmax
        results["dt"] = sim.dt

        if hasattr(sim, "r"):
            case = (
                "silent"
                if len(sim.stimes) == 0
                else ("exploding" if np.max(sim.r[:, -1]) > 5e2 else "normal")
            )
            results["spiking_case"] = case

        results["solver_match"] = match
        results["y_op_nan"] = bool(np.any(np.isnan(sim.y_op)))
        results["y_op_lim_nan"] = bool(np.any(np.isnan(sim.y_op_lim)))
        results["r_op_nan"] = bool(np.any(np.isnan(sim.r_op)))
        results["r_op_lim_nan"] = bool(np.any(np.isnan(sim.r_op_lim)))

        filepath = "%s.json" % basepath
        with open(filepath, "w") as file_handle:
            json.dump(results, file_handle, indent=4)
