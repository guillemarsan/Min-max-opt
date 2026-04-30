import argparse
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

############# PLOTTING ##########################################


def make_plot(
    ptype,
    data,
    title,
    axis_labels,
    basepath,
    legends=None,
):

    plt.figure(figsize=(4, 4))
    ax = plt.gca()

    if ptype == "cont":
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        i = 0
        xaxis = None
        for _, point in data.iterrows():
            mean = point["mean"]
            err = point["sem"]
            xaxis = point["xaxis"]
            legend = point["legend"] if legends is None else legends[i % len(legends)]

            plt.plot(xaxis, mean, color=c[i], label=legend)
            plt.fill_between(
                xaxis,
                mean - err,
                mean + err,
                color=c[i],
                alpha=0.3,
            )
            i += 1

    elif ptype == "disc":
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        i = 0
        xaxis = None
        for _, point in data.iterrows():
            mean = point["mean"]
            err = point["sem"]
            xaxis = point["xaxis"]
            if len(xaxis) > 10:
                xaxis = xaxis[:10]
                mean = mean[:10]
                err = err[:10]

            legend = point["legend"] if legends is None else legends[i % len(legends)]

            plt.errorbar(
                xaxis, mean, yerr=err, fmt="o", color=c[i], label=legend, capsize=5
            )
            plt.plot(xaxis, mean, color=c[i])
            i += 1

    plt.title(title, fontsize=10)
    plt.xlabel(axis_labels[0], fontsize=10)
    plt.ylabel(axis_labels[1], fontsize=10)
    plt.tick_params(axis="both", labelsize=10)
    plt.yscale("log")
    # plt.ylim([-1e-3, 0.01])
    plt.legend(frameon=False, prop={"size": 10})
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    filepath = "{0}-{1}_plot.svg".format(basepath, ptype)
    plt.savefig(filepath, dpi=600, bbox_inches="tight")


############# LOADING AND PREPARING DATA ########################
def load_dataframe(keyname, basepath):

    patt = "*-%s_df.csv" % keyname
    path = pathlib.Path(basepath)
    results_files = sorted(path.rglob(patt), reverse=True)
    no_load = False
    file = None
    if len(results_files) > 0:
        file = results_files[0]
    else:
        no_load = True

    if no_load:
        df = pd.DataFrame()
    else:
        old_file_name = str(file)
        with open(old_file_name, "rb") as f:
            df = pd.read_csv(f, index_col=[0])

    return df


def filter(df, visualize, params_sweep, tags):

    full_keys = []
    full_keys.extend(visualize)
    for t in tags:
        full_keys.append(t)

    filteredf = pd.DataFrame()
    for tuple in params_sweep:
        points = df[(df[visualize] == tuple).all(1)]
        if len(points.index) > 0:
            filteredf = pd.concat([filteredf, points[full_keys]])
    return filteredf


def prepare_single(df, labelsin, labelsout, labelsparams, visualize):

    newdf = pd.DataFrame([])
    for _, point in df.iterrows():
        newpoint = {}
        for i in np.arange(len(labelsin)):
            value = point[labelsin[i]]
            if type(value) is str and value.startswith("["):
                newpoint[labelsout[i]] = [np.array(eval(point[labelsin[i]]))]
            else:
                newpoint[labelsout[i]] = value
        legend = ""
        i = 0
        for l_idx, l in enumerate(labelsparams):
            legend += l + "=" + str(point[visualize[i]])
            legend += ", " if l_idx != len(labelsparams) - 1 else ""
            i += 1
        newpoint["legend"] = legend
        newdf = pd.concat([newdf, pd.DataFrame(newpoint)])
    return newdf


def prepare_combine(df, across, data_label, xaxis_label, labels, visualize):
    def combinelambda(set):
        points = pd.DataFrame([])
        for dlabel in data_label:
            newpoint = {}
            list = set[dlabel]
            list = list[~list.str.contains("nan")]
            list = list.apply(lambda x: np.array(eval(x)))

            if across:
                # If all elements have the same legnth
                lengths = list.apply(lambda x: len(x))
                if len(lengths.unique()) == 1:
                    means = np.mean(list._values)
                    sems = np.std(list._values) / np.sqrt(len(list))
                    xaxis = np.arange(lengths.iloc[0])
                # if not crop to the minimum length
                else:
                    min_length = lengths.min()
                    list_cropped = list.apply(lambda x: x[:min_length])
                    means = np.mean(list_cropped._values)
                    sems = np.std(list_cropped._values) / np.sqrt(len(list_cropped))
                    xaxis = np.arange(min_length)
            else:
                means = []
                sems = []
                xaxis = []
                i = 0
                for elem in list._values:
                    means.append(np.mean(elem))
                    sems.append(np.std(elem) / np.sqrt(elem.shape[0]))
                    xaxis.append(set.iloc[i][xaxis_label])
                    i += 1
                newpoint["values"] = [list._values]
            newpoint["mean"] = [means]
            newpoint["sem"] = [sems]
            newpoint["xaxis"] = [xaxis]

            legend = ""
            i = 0
            for l_idx, l in enumerate(labels):
                legend += l + "=" + str(set.iloc[0][visualize[i]])
                legend += ", " if l_idx != len(labels) - 1 else ""
                i += 1
            legend += (", " + dlabel) if len(data_label) > 1 else ""
            legend += "(" + str(len(list)) + ")"
            newpoint["legend"] = legend

            points = pd.concat([points, pd.DataFrame(newpoint)])
        return points

    # Mean and std within points and combine
    gb = df.groupby(visualize, as_index=False)
    newdf = gb.apply(lambda x: combinelambda(x))
    return newdf


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument(
        "--sub_case",
        type=str,
        default="error_y_lim",
        help="Sub-case to plot (e.g. error_y_lim, error_y, etc.)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/test/low_dim/",
        help="Directory to read and write files",
    )
    parser.add_argument(
        "--plot", type=str, default="error_cont", help="Which plot to make"
    )

    args = parser.parse_args()

    plot = args.plot

    timestr = time.strftime("%Y%m%d-%H%M%S")
    basepath = args.dir
    name = basepath + timestr

    print("Loading results...")

    # Load DataFrame
    df = pd.DataFrame()
    if plot in {""}:
        df = load_dataframe("database", basepath)
    elif plot in {"error_cont", "error_disc", "error_cont_one", "error_disc_one"}:
        df = load_dataframe("prediction", basepath)

    # Plotting
    print("Plotting results...")

    #### General plots ################

    if plot == "error_cont":

        visualize = ["spike_scale", "all_normal"]
        labels = ["c", "s"]
        params_sweep = [(sp, True) for sp in [0.75, 0.5, 0.25, 0.1, 0.05, 0.02]]
        tags = [args.sub_case, "time_vector"]

        fdf = filter(df, visualize, params_sweep, tags)

        newdf = prepare_combine(
            fdf,
            across=True,
            data_label=[args.sub_case],
            xaxis_label="time_vector",
            labels=labels,
            visualize=visualize,
        )

        title = "Error of the latent with respect to optimum"
        axis_labels = ["Time (s)", "Error (" + args.sub_case + ")"]
        print("Plotting results...")
        make_plot("cont", newdf, title, axis_labels, name)

    elif plot == "error_disc":

        visualize = ["spike_scale", "all_normal"]
        labels = ["c", "s"]
        params_sweep = [(sp, True) for sp in [0.75, 0.5, 0.25, 0.1, 0.05, 0.02]]
        tags = [args.sub_case, "time_vector"]

        fdf = filter(df, visualize, params_sweep, tags)

        newdf = prepare_combine(
            fdf,
            across=True,
            data_label=[args.sub_case],
            xaxis_label="time_vector",
            labels=labels,
            visualize=visualize,
        )

        title = "Error of the latent with respect to optimum"
        axis_labels = ["Spikes ", "Error (" + args.sub_case + ")"]
        print("Plotting results...")
        make_plot("disc", newdf, title, axis_labels, name)

    elif plot == "error_cont_one":

        visualize = ["spike_scale", "all_normal", "arg_rand_seed"]
        labels = ["c", "s", "seed"]

        params_sweep = [(sp, True, 0) for sp in [0.75, 0.5, 0.25, 0.1, 0.05, 0.02]]
        tags = [args.sub_case, "time_vector"]

        fdf = filter(df, visualize, params_sweep, tags)

        newdf = prepare_combine(
            fdf,
            across=True,
            data_label=[args.sub_case],
            xaxis_label="time_vector",
            labels=labels,
            visualize=visualize,
        )

        title = "Error of the latent with respect to optimum (1 example)"
        axis_labels = ["Time (s)", "Error (" + args.sub_case + ")"]
        print("Plotting results...")
        make_plot("cont", newdf, title, axis_labels, name)

    elif plot == "error_disc_one":

        visualize = ["spike_scale", "all_normal", "arg_rand_seed"]
        labels = ["c", "s", "seed"]
        params_sweep = [(sp, True, 0) for sp in [0.75, 0.5, 0.25, 0.1, 0.05, 0.02]]
        tags = [args.sub_case, "time_vector"]

        fdf = filter(df, visualize, params_sweep, tags)

        newdf = prepare_combine(
            fdf,
            across=True,
            data_label=[args.sub_case],
            xaxis_label="time_vector",
            labels=labels,
            visualize=visualize,
        )

        title = "Error of the latent with respect to optimum (1 example)"
        axis_labels = ["Spikes ", "Error (" + args.sub_case + ")"]
        print("Plotting results...")
        make_plot("disc", newdf, title, axis_labels, name)
