import json
import pathlib
import subprocess

import numpy as np

if __name__ == "__main__":

    cases = ["low_dim"]
    dir_loc = "./data/test/low_dim"

    compute = True
    analyse = True
    plot = True

    for case in cases:
        print("##### CASE " + case + " ######")

        basepath = dir_loc + "/"
        path = pathlib.Path(basepath)
        path.mkdir(parents=True, exist_ok=True)

        if case == "low_dim":
            dims, neurs, ncoups, spike_scale, seeds = (
                [5],
                [20],
                [8],
                [0.75, 0.5, 0.25, 0.05, 0.02],
                np.arange(60) + 0,
            )
        else:
            raise ValueError("Case not recognized")

        script_args = " --dir " + basepath + " --save_outputs --save_config"

        if compute:
            compute_script = "./src/run_random_ccvcvx.py"
            print("## RUNNING SIMULATIONS ##")
            # Run trials
            for dim in dims:
                print("Latent dimensions:" + str(dim))
                for n in neurs:
                    print("Constraints:" + str(n))
                    for ncoup in ncoups:
                        print("Non-coupled neurons:" + str(ncoup))
                        for spike in spike_scale:
                            print("Spike scale:" + str(spike))
                            for s in seeds:
                                print("Seed:" + str(s))
                                args = (
                                    script_args
                                    + " --lat_dim "
                                    + str(dim)
                                    + " --num_neur "
                                    + str(n)
                                    + " --num_ncoup "
                                    + str(ncoup)
                                    + " --rand_seed "
                                    + str(s)
                                    + " --spike_scale "
                                    + str(spike)
                                )

                                command = "python " + compute_script + " " + args
                                subprocess.run(command)

        if analyse:
            print("## ANALYZING DATA ##")
            # Do analysis
            analyse_script = "python ./src/analyse_results.py --dir " + basepath
            for s in ["database", "database_filter", "prediction"]:
                print("Analyse " + s)
                command = analyse_script + " --compute " + s
                subprocess.run(command)

        if plot:
            print("## PLOTTING RESULTS ##")
            # Do plots
            plot_script = "python ./src/plot_analysis.py --dir " + basepath
            array = []
            subarray = []
            if case == "low_dim":
                array = []
                for s in [
                    "error_y",
                    "error_r",
                    "error_y_lim",
                    "error_r_lim",
                    "error_y_s_ext",
                    "error_r_s_ext",
                    "error_y_lim_s_ext",
                    "error_r_lim_s_ext",
                ]:
                    array.append(["error_cont", s])
                    array.append(["error_cont_one", s])
                for s in ["error_y_s", "error_r_s", "error_y_lim_s", "error_r_lim_s"]:
                    array.append(["error_disc", s])
                    array.append(["error_disc_one", s])

            for case_plot in array:
                print("Plot " + case_plot[0] + " with " + case_plot[1])
                command = (
                    plot_script
                    + " --plot "
                    + case_plot[0]
                    + " --sub_case "
                    + case_plot[1]
                )
                subprocess.run(command)
