import argparse
import pathlib
import time
from unittest import case

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
    xlog=True,
    secondary=False,
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
            xaxis = point["xaxis"] / 1e3
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

        if "vol_KKT" in data.columns:
            # Voltage feasibility KKT residual analysis
            vol_KKT = data["vol_KKT"]
            mean_vol_KKT = np.mean(vol_KKT)
            SEM_vol_KKT = 2 * (np.std(vol_KKT) / np.sqrt(len(vol_KKT)))
            min_vol_KKT = np.min(vol_KKT)
            max_vol_KKT = np.max(vol_KKT)
            plt.text(
                0.05,
                0.80,
                rf"$\mathrm{{vol\ KKT}} = {mean_vol_KKT:.2e} \pm {SEM_vol_KKT:.2e}, [{min_vol_KKT:.2e}, {max_vol_KKT:.2e}]$",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

        if "compl_KKT" in data.columns:
            # Complementarity KKT residual analysis
            compl_KKT = data["compl_KKT"]
            mean_compl_KKT = np.mean(compl_KKT)
            SEM_compl_KKT = 2 * (np.std(compl_KKT) / np.sqrt(len(compl_KKT)))
            min_compl_KKT = np.min(compl_KKT)
            max_compl_KKT = np.max(compl_KKT)
            plt.text(
                0.05,
                0.70,
                rf"$\mathrm{{compl\ KKT}} = {mean_compl_KKT:.2e} \pm {SEM_compl_KKT:.2e}, [{min_compl_KKT:.2e}, {max_compl_KKT:.2e}]$",
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
            )

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

    elif ptype == "scatter":
        seeds = data["arg_rand_seed"].unique()
        spike_scales = data["spike_scale"].unique()
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(spike_scales)][::-1]
        neurs = 0
        if secondary:
            neurs = data.iloc[0]["arg_num_neur"]
            data = data.drop(columns=["arg_num_neur"])
        full_data = np.ones((len(spike_scales), len(seeds), 2)) * -1
        for _, point in data.iterrows():
            valuex = point.iloc[-1] if ~np.isnan(point.iloc[-1]) else np.nan
            valuey = (
                eval(point.iloc[-2])[0]
                if (isinstance(point.iloc[-2], str) and (point.iloc[-2] != "[nan]"))
                else np.nan
            )
            seed = point["arg_rand_seed"]
            seed_idx = seeds.tolist().index(seed)
            spike_scale = point["spike_scale"]
            spike_scale_idx = spike_scales.tolist().index(spike_scale)
            if np.isnan(valuex) or np.isnan(valuey):
                continue
            full_data[spike_scale_idx, seed_idx, 0] = valuex
            full_data[spike_scale_idx, seed_idx, 1] = valuey
            plt.scatter(valuex, valuey, color=c[spike_scale_idx], alpha=0.5)
        for s in range(len(spike_scales)):
            valid_seeds = np.where(full_data[s, :, 0] != -1)[0]
            mean_xaxis = np.mean(full_data[s, valid_seeds, 0])
            mean_err = np.mean(full_data[s, valid_seeds, 1])
            sem_xaxis = np.std(full_data[s, valid_seeds, 0]) / np.sqrt(len(valid_seeds))
            sem_err = np.std(full_data[s, valid_seeds, 1]) / np.sqrt(len(valid_seeds))
            legend = "k = " + str(spike_scales[s]) + "(" + str(len(valid_seeds)) + ")"
            plt.scatter(
                mean_xaxis,
                mean_err,
                color=c[s],
                label=legend,
                edgecolor="k",
                s=150,
                marker="^",
            )
            # plot error bar without marker
            plt.errorbar(
                mean_xaxis,
                mean_err,
                xerr=sem_xaxis,
                yerr=sem_err,
                fmt="none",
                color=c[s],
                alpha=1,
                capsize=3,
            )

        if xlog:
            plt.xscale("log")
        plt.yscale("log")

        if secondary:
            energy_per_syn = 0.03  # nJ/synapse
            secax = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda x: x * neurs * energy_per_syn,  # type: ignore
                    lambda x: x / (neurs * energy_per_syn),  # type: ignore
                ),
            )
            secax.set_xlabel("Energy estimation (nJ)")
            secax.set_xscale("log")

        m = []
        b = []
        r2 = []
        for i in np.arange(len(seeds)):
            if np.where(~np.isin(full_data[:, i, 0], [0, -1]))[0].size < 3:
                continue
            x = full_data[np.where(~np.isin(full_data[:, i, 0], [0, -1]))[0], i, 0]
            y = full_data[np.where(~np.isin(full_data[:, i, 0], [0, -1]))[0], i, 1]
            xs = np.log(x) if xlog else x
            ys = np.log(np.maximum(y, 1e-10))
            p, residuals, *_ = np.polyfit(xs, ys, 1, full=True)
            m.append(p[0])
            b.append(p[1])
            rss = residuals[0] if residuals.size > 0 else 0
            tss = np.sum((ys - np.mean(ys)) ** 2)
            r2.append(1 - rss / tss if tss > 0 else 0)
        mean_m = np.mean(m)
        sem_m = np.std(m) / np.sqrt(len(m))
        mean_b = np.mean(b)
        sem_b = np.std(b) / np.sqrt(len(b))
        mean_r2 = np.mean(r2)
        sem_r2 = np.std(r2) / np.sqrt(len(r2))
        # x = full_data[:, 0]
        # x[x == 0] = 1e-10
        # y = full_data[:, 1]
        # y[y == 0] = 1e-10
        # xs = np.log(x)
        # ys = np.log(y)
        # p, residuals, *_ = np.polyfit(xs, ys, 1, full=True)
        # m, b = p
        # x_vec = np.linspace(np.min(x), np.max(x))
        # plt.plot(
        #     x_vec,
        #     np.exp(b) * x_vec**m,
        #     linestyle="--",
        #     color="grey",
        #     alpha=0.5,
        #     linewidth=2,
        # )

        # rss = residuals[0]
        # tss = np.sum((ys - np.mean(ys)) ** 2)
        # r2 = 1 - rss / tss
        strm = "{:.2f}".format(mean_m) + " ± " + "{:.2f}".format(2 * sem_m)
        strb = "{:.2f}".format(mean_b) + " ± " + "{:.2f}".format(2 * sem_b)
        strr2 = "{:.2f}".format(mean_r2) + " ± " + "{:.2f}".format(2 * sem_r2)
        strper = "{:.2f}".format(np.sum(np.array(m) < 0) / len(m) * 100) + "\\%"
        # plot it in bottom left corner
        main = (
            rf"$\mathrm{{err}} = e^{{{strb}}} \# s^{{{strm}}}$"
            if xlog
            else rf"$\mathrm{{err}} = e^{{{strb}}} e^{{{strm} \# s}}$"
        )
        plt.text(
            0.05,
            0.05,
            # break in three lines
            (main + "\n" rf"$R^2 = {strr2}$" "\n" rf"$neg\% = {strper}$"),
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )

    elif ptype == "scatter_uniform":
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        i = 0
        seeds = data["values"][0][0].shape[0]
        full_data = np.zeros((data.shape[0], seeds, 2))
        for _, point in data.iterrows():
            values = point["values"]
            mean_err = np.mean(values)
            sem_err = np.std(values) / np.sqrt(len(values))
            xaxis = point["xaxis"]
            mean_xaxis = np.mean(xaxis)
            sem_xaxis = np.std(xaxis) / np.sqrt(len(xaxis))
            legend = point["legend"] if legends is None else legends[i % len(legends)]
            for j in range(len(values)):
                full_data[i, j, 0] = xaxis[j]
                full_data[i, j, 1] = values[j][0]
                plt.scatter(xaxis[j], values[j], color=c[i], label=legend, alpha=0.5)
            plt.scatter(
                mean_xaxis,
                mean_err,
                color=c[i],
                label=legend,
                edgecolor="k",
                s=150,
                marker="^",
            )
            # plot error bar without marker
            plt.errorbar(
                mean_xaxis,
                mean_err,
                xerr=sem_xaxis,
                yerr=sem_err,
                fmt="none",
                color=c[i],
                alpha=1,
                capsize=3,
            )
            i += 1
        if xlog:
            plt.xscale("log")
        plt.yscale("log")
        m = []
        b = []
        r2 = []
        for i in np.arange(seeds):
            if np.where(full_data[:, i, 0] != -1)[0].size < 3:
                continue
            x = full_data[:, i, 0]
            y = full_data[:, i, 1]
            xs = np.log(x) if xlog else x
            ys = np.log(np.maximum(y, 1e-10))
            p, residuals, *_ = np.polyfit(xs, ys, 1, full=True)
            m.append(p[0])
            b.append(p[1])
            rss = residuals[0] if residuals.size > 0 else 0
            tss = np.sum((ys - np.mean(ys)) ** 2)
            r2.append(1 - rss / tss if tss > 0 else 0)
        mean_m = np.mean(m)
        sem_m = np.std(m) / np.sqrt(len(m))
        mean_b = np.mean(b)
        sem_b = np.std(b) / np.sqrt(len(b))
        mean_r2 = np.mean(r2)
        sem_r2 = np.std(r2) / np.sqrt(len(r2))
        # x = full_data[:, 0]
        # x[x == 0] = 1e-10
        # y = full_data[:, 1]
        # y[y == 0] = 1e-10
        # xs = np.log(x)
        # ys = np.log(y)
        # p, residuals, *_ = np.polyfit(xs, ys, 1, full=True)
        # m, b = p
        # x_vec = np.linspace(np.min(x), np.max(x))
        # plt.plot(
        #     x_vec,
        #     np.exp(b) * x_vec**m,
        #     linestyle="--",
        #     color="grey",
        #     alpha=0.5,
        #     linewidth=2,
        # )

        # rss = residuals[0]
        # tss = np.sum((ys - np.mean(ys)) ** 2)
        # r2 = 1 - rss / tss
        strm = "{:.2f}".format(mean_m) + " ± " + "{:.2f}".format(2 * sem_m)
        strb = "{:.2f}".format(mean_b) + " ± " + "{:.2f}".format(2 * sem_b)
        strr2 = "{:.2f}".format(mean_r2) + " ± " + "{:.2f}".format(2 * sem_r2)
        strper = "{:.2f}".format(np.sum(np.array(m) < 0) / len(m) * 100) + "\\%"
        main = (
            rf"$\mathrm{{err}} = e^{{{strb}}} \# s^{{{strm}}}$"
            if xlog
            else rf"$\mathrm{{err}} = e^{{{strb}}} e^{{{strm} \# s}}$"
        )
        plt.text(
            0.05,
            0.95,
            # break in three lines
            (main + "\n" rf"$R^2 = {strr2}$" "\n" rf"$neg\% = {strper}$"),
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )

    elif ptype == "bars":

        idx = data[data["Wacac_neg"].notna()].index
        Wacac_neg_notnan = data.loc[idx, "Wacac_neg"]
        spiking_case_notnan = data.loc[idx, "spiking_case"]
        idx_nan = data[data["Wacac_neg"].isna()].index
        spiking_case_nan = data.loc[idx_nan, "spiking_case"]
        eaneg = sum(Wacac_neg_notnan & (spiking_case_notnan == "exploding"))
        naneg = sum(Wacac_neg_notnan & (spiking_case_notnan == "normal"))
        epos = sum(~Wacac_neg_notnan & (spiking_case_notnan == "exploding"))
        npos = sum(~Wacac_neg_notnan & (spiking_case_notnan == "normal"))
        enan = sum(spiking_case_nan == "exploding")
        nnan = sum(spiking_case_nan == "normal")

        if "Wcc_neg" in data.columns:
            labels = labels = [
                r"$\boldsymbol{W}^{cc} \prec 0$",
                r"$\boldsymbol{W}^{cc}_{\mathcal{A}_c\mathcal{A}_c} \prec 0$",
                r"$\boldsymbol{W}^{cc}_{\mathcal{A}_c\mathcal{A}_c} \nprec 0$",
                "Solver failed \n to converge",
            ]
            eneg = sum(data["Wcc_neg"] & (data["spiking_case"] == "exploding"))
            nneg = sum(data["Wcc_neg"] & (data["spiking_case"] == "normal"))
            e_vals = [np.sum(eneg), np.sum(eaneg), np.sum(epos), np.sum(enan)]
            n_vals = [np.sum(nneg), np.sum(naneg), np.sum(npos), np.sum(nnan)]
        else:
            labels = labels = [
                r"$\boldsymbol{W}^{cc}_{\mathcal{A}_c\mathcal{A}_c} \prec 0$",
                r"$\boldsymbol{W}^{cc}_{\mathcal{A}_c\mathcal{A}_c} \nprec 0$",
                "Solver failed \n to converge",
            ]
            e_vals = [np.sum(eaneg), np.sum(epos), np.sum(enan)]
            n_vals = [np.sum(naneg), np.sum(npos), np.sum(nnan)]

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width / 2, e_vals, width, color="r", alpha=0.5, label="exploding")

        plt.bar(x + width / 2, n_vals, width, color="b", alpha=0.5, label="normal")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon=False, prop={"size": 10}, loc="upper right")

        # Active set condition analysis
        active_set_cond = data["active_set_cond"]
        active_set_cond_notna = active_set_cond[active_set_cond.notna()]
        asc_per = (
            np.sum(active_set_cond_notna == True) / len(active_set_cond_notna)
        ) * 100
        plt.text(
            0.05,
            0.90,
            rf"$\mathrm{{active\ set\ condition}} = {asc_per:.2f}\%$",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )

    elif ptype == "cont_count":
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        i = 0
        xaxis = None
        scales = data["spike_scale"].unique()
        classes = data["spiking_case"].unique()
        normal_per_scale = []
        silent_per_scale = []
        exploding_per_scale = []
        for s in scales:
            normal_per_scale.append(
                len(
                    data[
                        (data["spike_scale"] == s) & (data["spiking_case"] == "normal")
                    ]
                )
            )
            silent_per_scale.append(
                len(
                    data[
                        (data["spike_scale"] == s) & (data["spiking_case"] == "silent")
                    ]
                )
            )
            exploding_per_scale.append(
                len(
                    data[
                        (data["spike_scale"] == s)
                        & (data["spiking_case"] == "exploding")
                    ]
                )
            )
        total = normal_per_scale[0] + silent_per_scale[0] + exploding_per_scale[0]
        per_normal = np.array(normal_per_scale) / total * 100
        per_silent = np.array(silent_per_scale) / total * 100
        per_exploding = np.array(exploding_per_scale) / total * 100
        # Plotting as shaded areas under the curves
        plt.fill_between(
            scales,
            per_silent,
            color=c[0],
            alpha=0.5,
            label="silent",
        )
        plt.fill_between(
            scales,
            per_normal + per_silent,
            per_silent,
            color=c[2],
            alpha=0.5,
            label="normal",
        )
        plt.fill_between(
            scales,
            per_exploding + per_normal + per_silent,
            per_normal + per_silent,
            color=c[1],
            alpha=0.5,
            label="exploding",
        )
        plt.plot(
            scales,
            per_silent,
            color="grey",
            marker="o",
            markersize=5,
        )
        plt.plot(
            scales,
            per_normal + per_silent,
            color="grey",
            marker="o",
            markersize=5,
        )
        plt.plot(
            scales,
            per_normal + per_silent + per_exploding,
            color="grey",
            marker="o",
            markersize=5,
        )
        plt.ylim(bottom=0)

    plt.title(title, fontsize=10)
    plt.xlabel(axis_labels[0], fontsize=10)
    plt.ylabel(axis_labels[1], fontsize=10)
    plt.tick_params(axis="both", labelsize=10)
    if ptype not in {"bars", "cont_count"}:
        plt.yscale("log")
    # plt.ylim([-1e-3, 0.01])
    (
        plt.legend(frameon=False, prop={"size": 10})
        if ptype in {"cont", "disc", "cont_count"}
        else None
    )
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
        default="error_y_lim_final",
        help="Sub-case to plot (e.g. error_y_lim, error_y, etc.)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/test/low_dim/",
        help="Directory to read and write files",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="error_scatter_nspikes",
        help="Which plot to make",
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
    elif plot in {
        "error_cont",
        "error_disc",
        "error_scatter_nspikes",
        "error_scatter_ctime",
        "error_scatter_ctime_uniform",
        "error_cont_one",
        "error_disc_one",
        "conv_exploding",
        "sim_type",
        "KKT_residuals",
    }:
        df = load_dataframe("prediction", basepath)

    # Plotting
    print("Plotting results...")

    #### General plots ################

    if plot == "error_cont":

        visualize = [
            "spike_scale",
            "all_normal",
        ]
        if "_y_lim" in args.sub_case:
            visualize.append("any_y_op_lim_nan")
        elif "_y" in args.sub_case:
            visualize.append("any_y_op_nan")
        elif "_r_lim" in args.sub_case:
            visualize.append("any_r_op_lim_nan")
        elif "_r" in args.sub_case:
            visualize.append("any_r_op_nan")
        labels = ["k", "s"]
        params_sweep = [(sp, True, False) for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]]
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

        title = (
            "Error of the latent with respect to optimum"
            if "_y" in args.sub_case
            else "Error of the rates with respect to optimum"
        )
        axis_labels = [
            "Time (s)",
            (
                " Primal Error (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "Dual Error (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("cont", newdf, title, axis_labels, name)

    elif plot == "error_disc":

        visualize = ["spike_scale", "all_normal"]
        labels = ["k", "s"]
        params_sweep = [(sp, True) for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]]
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

        title = (
            "Error of the latent with respect to optimum"
            if "_y" in args.sub_case
            else "Error of the rates with respect to optimum"
        )
        axis_labels = [
            "Spikes ",
            (
                " Primal Error (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "DualError (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("disc", newdf, title, axis_labels, name)

    elif plot == "KKT_residuals":

        visualize = ["spiking_case"]
        labels = ["c"]
        params_sweep = ["normal"]
        tags = ["vol_KKT", "compl_KKT", "spike_scale"]

        fdf = filter(df, visualize, params_sweep, tags)

        newdf = prepare_combine(
            fdf,
            across=True,
            data_label=["vol_KKT", "compl_KKT"],
            xaxis_label="spike_scale",
            labels=labels,
            visualize=visualize,
        )

        title = (
            "KKT residuals of the latent with respect to optimum"
            if "_y" in args.sub_case
            else "KKT residuals of the rates with respect to optimum"
        )
        axis_labels = [
            "Spike Scale (k)",
            (
                " KKT Residuals (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "KKT Residuals (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("cont", newdf, title, axis_labels, name)

    elif plot == "error_scatter_nspikes":

        visualize = [
            "spike_scale",
            "arg_rand_seed",
            # "all_normal",
        ]
        # visualize = [
        #     "spike_scale",
        #     "all_normal",
        # ]
        # if "_y_lim" in args.sub_case:
        #     visualize.append("any_y_op_lim_nan")
        # elif "_y" in args.sub_case:
        #     visualize.append("any_y_op_nan")
        # elif "_r_lim" in args.sub_case:
        #     visualize.append("any_r_op_lim_nan")
        # elif "_r" in args.sub_case:
        #     visualize.append("any_r_op_nan")
        labels = ["k", "s"]
        # params_sweep = [(sp, True, False) for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]]
        params_sweep = [
            (sp, s + 480)
            for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]
            for s in range(120)
        ]
        tags = (
            [args.sub_case, "nspikes", "arg_num_neur"]
            if "conv" not in args.sub_case
            else [args.sub_case, "nspikes_conv", "arg_num_neur"]
        )

        fdf = filter(df, visualize, params_sweep, tags)

        # newdf = prepare_combine(
        #     fdf,
        #     across=False,
        #     data_label=(
        #         [args.sub_case]
        #         # if "_y_lim" not in args.sub_case
        #         # else [args.sub_case, "vol_KKT", "compl_KKT"]
        #     ),
        #     xaxis_label="nspikes",
        #     labels=labels,
        #     visualize=visualize,
        # )

        title = (
            "Error of the latent with respect to optimum"
            if "_y" in args.sub_case
            else "Error of the rates with respect to optimum"
        )
        axis_labels = [
            "Number of Spikes",
            (
                "Final Primal Error (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "Final Dual Error (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("scatter", fdf, title, axis_labels, name, secondary=True)

    elif plot == "error_scatter_ctime":

        visualize = [
            "spike_scale",
            "arg_rand_seed",
        ]
        labels = ["k", "s"]
        params_sweep = [
            (sp, s) for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01] for s in range(120)
        ]
        tags = [args.sub_case, "conv_time"]

        fdf = filter(df, visualize, params_sweep, tags)

        title = (
            "Error of the latent with respect to optimum"
            if "_y" in args.sub_case
            else "Error of the rates with respect to optimum"
        )
        axis_labels = [
            "Convergence Time (s)",
            (
                "Primal Error at Convergence (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "Dual Error at Convergence (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("scatter", fdf, title, axis_labels, name, xlog=False)

    elif plot == "error_scatter_ctime_uniform":

        visualize = [
            "spike_scale",
            "all_normal",
        ]
        if "_y_lim" in args.sub_case:
            visualize.append("any_y_op_lim_nan")
        elif "_y" in args.sub_case:
            visualize.append("any_y_op_nan")
        elif "_r_lim" in args.sub_case:
            visualize.append("any_r_op_lim_nan")
        elif "_r" in args.sub_case:
            visualize.append("any_r_op_nan")
        labels = ["k", "s"]
        params_sweep = [(sp, True, False) for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]]
        tags = [args.sub_case, "conv_time"]

        fdf = filter(df, visualize, params_sweep, tags)

        newdf = prepare_combine(
            fdf,
            across=False,
            data_label=[args.sub_case],
            xaxis_label="conv_time",
            labels=labels,
            visualize=visualize,
        )

        title = (
            "Error of the latent with respect to optimum"
            if "_y" in args.sub_case
            else "Error of the rates with respect to optimum"
        )
        axis_labels = [
            "Convergence Time (s)",
            (
                "Primal Error at Convergence (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "Dual Error at Convergence (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("scatter_uniform", newdf, title, axis_labels, name, xlog=False)

    elif plot == "conv_exploding":

        visualize = ["spike_scale"]
        labels = ["k", "s"]
        params_sweep = [(sp) for sp in [0.4]]
        # tags = ["Wcc_neg", "Wacac_neg", "spiking_case"]
        tags = ["Wacac_neg", "active_set_cond", "spiking_case"]

        fdf = filter(df, visualize, params_sweep, tags)

        title = "Convergence analysis"
        axis_labels = ["Conditions", "Number of simulations"]
        print("Plotting results...")
        make_plot("bars", fdf, title, axis_labels, name)

    elif plot == "error_cont_one":

        visualize = ["spike_scale", "all_normal", "arg_rand_seed"]
        labels = ["k", "s", "seed"]

        params_sweep = [(sp, True, 0) for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]]
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

        title = (
            "Error of the latent with respect to optimum (1 example)"
            if "_y" in args.sub_case
            else "Error of the rates with respect to optimum (1 example)"
        )
        axis_labels = [
            "Time (s)",
            (
                " Primal Error (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "DualError (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("cont", newdf, title, axis_labels, name)

    elif plot == "sim_type":

        visualize = ["spike_scale"]
        labels = ["s"]

        params_sweep = [sp for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]]
        tags = ["spiking_case"]

        newdf = filter(df, visualize, params_sweep, tags)

        title = "Types of networks depending on spike magnitude"
        axis_labels = [
            "Spike magnitude (k)",
            "Percentage of networks (%)",
        ]
        print("Plotting results...")
        make_plot("cont_count", newdf, title, axis_labels, name)

    elif plot == "error_disc_one":

        visualize = ["spike_scale", "all_normal", "arg_rand_seed"]
        labels = ["k", "s", "seed"]
        params_sweep = [(sp, True, 0) for sp in [0.4, 0.25, 0.1, 0.05, 0.02, 0.01]]
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

        title = (
            "Error of the latent with respect to optimum (1 example)"
            if "_y  " in args.sub_case
            else "Error of the rates with respect to optimum (1 example)"
        )
        axis_labels = [
            "Spikes ",
            (
                " Primal Error (" + args.sub_case + ")"
                if "_y" in args.sub_case
                else "DualError (" + args.sub_case + ")"
            ),
        ]
        print("Plotting results...")
        make_plot("disc", newdf, title, axis_labels, name)
