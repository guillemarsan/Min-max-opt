import argparse
import json
import random
import string
import time

import numpy as np
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
        default=0.1,
        help="Spike scale for the simulation",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=3,
        help="Random seed for the sampling of parameters",
    )

    parser.add_argument(
        "--dir", type=str, default="./data/test/", help="Directory to dump output"
    )
    parser.add_argument(
        "--plot", action="store_true", default=True, help="Plot the results"
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

    # Initialize simulation
    sim = Simulation.init_optim(
        Q=Q_diag,
        b=b_diag,
        E=E_diag,
        Tp=Tp_diag,
        spike_scale=args.spike_scale,
        Tmax=5,
        y0=np.zeros(args.lat_dim),
    )

    # RUN SIMULATION
    # run simulation
    sim.run(draw_break="no")

    # optimize simulation
    sim.optimize(Q=Q_diag)

    # Plot
    if args.plot:
        sim.plot()

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
        results["b"] = b.tolist()
        results["E"] = E.tolist()
        results["D"] = sim.net.D.tolist()
        results["Tp"] = Tp.tolist()
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

        filepath = "%s.json" % basepath
        with open(filepath, "w") as file_handle:
            json.dump(results, file_handle, indent=4)
