import argparse
import json
import pathlib
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.signal import convolve

tostore = lambda array: np.array2string(
    array, separator=",", suppress_small=True, threshold=np.inf, max_line_width=np.inf  # type: ignore
)


def compute_database(basepath: str) -> pd.DataFrame:
    """
    Compute a database from all the results stored in basepath

    Parameters
    ----------
    basepath : str
        Base path where to look for results

    Returns
    -------
    df: pd.DataFrame
        Dataframe with all the results
    """

    patt = "*.json"
    path = pathlib.Path(basepath)
    results_files = path.rglob(patt)

    newdata = []
    for f in results_files:
        with open(f) as res_file:
            f = json.load(res_file)

            # Unroll args
            for args, value in f["args"].items():
                f["arg_" + args] = value
            del f["args"]

            delete_list = [
                "arg_dir",
                "arg_plot",
                "arg_save_sim",
                "arg_save_outputs",
                "arg_save_config",
            ]
            for d in delete_list:
                del f[d]

            f["time_vector"] = tostore(
                np.linspace(0, f["Tmax"], int(f["Tmax"] / f["dt"]))
            )

            newdata.append(f)

    return pd.DataFrame(newdata)


def load_dataframe(keyname: str, basepath: str) -> pd.DataFrame:
    """
    Load a dataframe

    Parameters
    ----------
    keyname : str
        Keyname of the dataframe

    basepath : str
        Basepath where to look for the dataframe

    Returns
    -------
    df: pd.DataFrame
        Loaded dataframe
    """

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


def load_data(basepath: str, varname: str) -> np.ndarray:
    """
    Load data from a csv file

    Parameters
    ----------
    basepath : str
        Basepath where to look for the data

    varname : str
        Variable name to load

    Returns
    -------
    data: np.ndarray
        Loaded data
    """

    filename = "{0}-{1}.csv".format(basepath, varname)
    with open(filename) as data_file:
        data = np.genfromtxt(data_file)

    return data


def compute_per_simulation(
    dbase: pd.DataFrame, params: list, func: Callable[[pd.DataFrame, list], Any]
) -> pd.Series:
    """
    Compute a metric per simulation

    Parameters
    ----------
    dbase : pd.DataFrame
        Database to use
    params : list
        List of parameters to use
    func : callable
        Function to apply

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the results
    """

    return dbase.apply(lambda x: func(x, params), axis=1)


def compute_across(
    dbase: pd.DataFrame,
    across: list,
    params: list,
    func: Callable[[pd.DataFrame, list], Any],
) -> pd.Series:
    """
    Compute a metric across simulations

    Parameters
    ----------
    dbase : pd.DataFrame
        Database to use
    across : list
        List of parameters to group by
    params : list
        List of parameters to use
    func : callable
        Function to apply

    Returns
    -------
    df: pd.Series
        Dataframe with the results
    """
    groupby_pars = params
    for a in across:
        groupby_pars.remove(a)

    gb = dbase.groupby(groupby_pars, as_index=False)
    df = gb.apply(lambda x: func(x, groupby_pars))
    return df


###### ANALYSIS ACROSS SIMULATIONS ######


def all_normal(set: pd.DataFrame):
    """
    Check if all simulations in the set are normal

    Parameters
    ----------
    set : pd.DataFrame
        Set of simulations to check

    Returns
    -------
    set : pd.DataFrame
        Set of simulations to check with added column "all_normal" which is True if all simulations are normal else False
    """

    all_normal = set["spiking_case"].isin(["normal"]).all()
    set["all_normal"] = bool(all_normal)
    return set


###### ANALYSIS PER SIMULATION ######


def analyse_prediction(point: pd.Series):
    """
    Analyse the prediction performance of one simulation

    Parameters
    ----------
    point : pd.Series
        Series with the simulation data

    Returns
    -------
    point: pd.Series
        Series with added analysis results
    """

    print("Analysing prediction for " + point["basepath"])
    y = load_data(point["basepath"], "y")
    y_op = load_data(point["basepath"], "y_op")
    y_op_lim = load_data(point["basepath"], "y_op_lim")
    r = load_data(point["basepath"], "r")
    r_op = load_data(point["basepath"], "r_op")
    r_op_lim = load_data(point["basepath"], "r_op_lim")
    data = [y, y_op, y_op_lim, r, r_op, r_op_lim]

    label_solver_sols(data, point)

    if point["spiking_case"] in ["normal", "silent"]:
        compute_time_errors(data, point)

    if point["spiking_case"] == "normal":
        stimes = load_data(point["basepath"], "stimes")
        data.append(stimes)
        compute_spike_errors(data, point)

    return point


def label_solver_sols(data: list, point: pd.Series):
    """
    Label the solutions of the solver for one simulation

    nan : True if it contains NaN else False
    match : True if y* = Dr* + b else False
    KKT : True if it is a KKT point else False

    Parameters
    ----------
    point : pd.Series
        Series with the simulation data
    """

    y_op = data[1]
    y_op_lim = data[2]
    r_op = data[4]
    r_op_lim = data[5]

    D = np.array(eval(point["D"]))
    Q = np.array(eval(point["Q"]))
    E = np.array(eval(point["E"]))
    b = np.array(eval(point["b"]))
    Tp = np.array(eval(point["Tp"]))

    point["y_lim_nan"] = np.isnan(y_op_lim).any()
    point["r_lim_nan"] = np.isnan(r_op_lim).any()
    y_op_lim_from_r_op_lim = np.array([])
    if not point["r_lim_nan"]:
        y_op_lim_from_r_op_lim = D @ r_op_lim - np.linalg.inv(Q) @ b

    if not point["y_lim_nan"] and not point["r_lim_nan"]:
        point["y_lim_match"] = np.allclose(y_op_lim, y_op_lim_from_r_op_lim, atol=1e-3)
    else:
        point["y_lim_match"] = False

    if not point["r_lim_nan"]:
        vol = E @ y_op_lim_from_r_op_lim - Tp
        feasr = np.all(r_op_lim >= 0)
        feasV = np.all(vol <= 0)
        compl = np.allclose(r_op_lim * vol, 0)
        point["r_lim_KKT"] = feasr and feasV and compl
    else:
        point["r_lim_KKT"] = False


def compute_spike_errors(data, point: pd.Series):
    """
    Compute spike errors for one simulation

    Parameters
    ----------
    data : list
        List with the simulation data

    point : pd.Series
        Series with the simulation data

    """

    def extend(array: np.ndarray, idx: np.ndarray, length: int) -> np.ndarray:
        """
        New array z fulfills z[i] = array[j] for idx[j] <= i < idx[j+1]

        Parameters
        ----------
        array : np.ndarray
            Array to extend
        idx : np.ndarray
            Indices to use for extension
        length : int
            Length of the extended array

        Returns
        -------
        z: np.ndarray
            Extended array
        """
        idx = np.concatenate((idx, [length]))
        z = np.zeros((array.shape[0], length))
        for start, end, val in zip(idx[:-1], idx[1:], array.T):
            z[:, start:end] = val[:, np.newaxis]
        return z

    basepath = point["basepath"]
    dt = point["dt"]
    y = data[0]
    r = data[3]
    stimes = data[6]
    if stimes.ndim == 1:
        stimes = stimes.reshape(-1, 2)
    stimes_idx = (stimes[:, 1] / dt).astype(int)

    # Predictions
    y_pred = (y[:, stimes_idx] + y[:, stimes_idx + 1]) / 2
    y_pred_ext = extend(y_pred, stimes_idx, y.shape[1])
    y_lim_pred = y[:, stimes_idx]
    y_lim_pred_ext = extend(y_lim_pred, stimes_idx, y.shape[1])
    r_pred = (r[:, stimes_idx] + r[:, stimes_idx + 1]) / 2
    r_pred_ext = extend(r_pred, stimes_idx, r.shape[1])
    r_lim_pred = r[:, stimes_idx]
    r_lim_pred_ext = extend(r_lim_pred, stimes_idx, r.shape[1])

    point["y_pred_1"] = tostore(y_pred)
    point["y_pred_1_ext"] = tostore(y_pred_ext)
    point["y_lim_pred_1"] = tostore(y_lim_pred)
    point["y_lim_pred_1_ext"] = tostore(y_lim_pred_ext)
    point["r_pred_1"] = tostore(r_pred)
    point["r_pred_1_ext"] = tostore(r_pred_ext)
    point["r_lim_pred_1"] = tostore(r_lim_pred)
    point["r_lim_pred_1_ext"] = tostore(r_lim_pred_ext)

    # Errors
    y_op = data[1]
    y_op_lim = data[2]
    r_op = data[4]
    r_op_lim = data[5]

    point["error_y_s"] = tostore(
        np.linalg.norm((y_op[:, np.newaxis] - y_pred) ** 2, axis=0)
    )
    point["error_y_s_ext"] = tostore(
        np.linalg.norm((y_op[:, np.newaxis] - y_pred_ext) ** 2, axis=0)
    )
    point["error_y_lim_s"] = tostore(
        np.linalg.norm((y_op_lim[:, np.newaxis] - y_lim_pred) ** 2, axis=0)
    )
    point["error_y_lim_s_ext"] = tostore(
        np.linalg.norm((y_op_lim[:, np.newaxis] - y_lim_pred_ext) ** 2, axis=0)
    )
    point["error_r_s"] = tostore(
        np.linalg.norm((r_op[:, np.newaxis] - r_pred) ** 2, axis=0)
    )
    point["error_r_s_ext"] = tostore(
        np.linalg.norm((r_op[:, np.newaxis] - r_pred_ext) ** 2, axis=0)
    )
    point["error_r_lim_s"] = tostore(
        np.linalg.norm((r_op_lim[:, np.newaxis] - r_lim_pred) ** 2, axis=0)
    )
    point["error_r_lim_s_ext"] = tostore(
        np.linalg.norm((r_op_lim[:, np.newaxis] - r_lim_pred_ext) ** 2, axis=0)
    )

    # convs = [2, 3, 5]
    # for c in convs:
    #     # Compute the average with a sliding window of size c.
    #     # If i < c, we average over the last i values, otherwise we average over the last c values.
    #     y_pred_sum = convolve(y_pred, np.ones((1, c)), mode="full")[
    #         :, : y_pred.shape[1]
    #     ]
    #     y_pred_avg = y_pred_sum / np.minimum(np.arange(1, y_pred.shape[1] + 1), c)
    #     y_lim_pred_sum = convolve(y_lim_pred, np.ones((1, c)), mode="full")[
    #         :, : y_lim_pred.shape[1]
    #     ]
    #     y_lim_pred_avg = y_lim_pred_sum / np.minimum(
    #         np.arange(1, y_lim_pred.shape[1] + 1), c
    #     )
    #     r_pred_sum = convolve(r_pred, np.ones((1, c)), mode="full")[
    #         :, : r_pred.shape[1]
    #     ]
    #     r_pred_avg = r_pred_sum / np.minimum(np.arange(1, r_pred.shape[1] + 1), c)
    #     r_lim_pred_sum = convolve(r_lim_pred, np.ones((1, c)), mode="full")[
    #         :, : r_lim_pred.shape[1]
    #     ]
    #     r_lim_pred_avg = r_lim_pred_sum / np.minimum(
    #         np.arange(1, r_lim_pred.shape[1] + 1), c
    #     )

    #     point["y_pred_%d" % c] = tostore(y_pred_avg)
    #     point["y_pred_%d_ext" % c] = tostore(extend(y_pred_avg, stimes_idx, y.shape[1]))
    #     point["r_pred_%d" % c] = tostore(r_pred_avg)
    #     point["r_pred_%d_ext" % c] = tostore(extend(r_pred_avg, stimes_idx, r.shape[1]))
    #     point["y_lim_pred_%d" % c] = tostore(y_lim_pred_avg)
    #     point["y_lim_pred_%d_ext" % c] = tostore(
    #         extend(y_lim_pred_avg, stimes_idx, y.shape[1])
    #     )
    #     point["r_lim_pred_%d" % c] = tostore(r_lim_pred_avg)
    #     point["r_lim_pred_%d_ext" % c] = tostore(
    #         extend(r_lim_pred_avg, stimes_idx, r.shape[1])
    #     )


def compute_time_errors(data: list, point: pd.Series):
    """
    Compute error for one simulation

    Parameters
    ----------
    data : list
        List with the simulation data

    point : pd.Series
        Series with the simulation data

    """
    y = data[0]
    y_op = data[1]
    y_op_lim = data[2]
    r = data[3]
    r_op = data[4]
    r_op_lim = data[5]

    point["error_y"] = tostore(np.linalg.norm((y_op[:, np.newaxis] - y) ** 2, axis=0))
    point["error_y_lim"] = tostore(
        np.linalg.norm((y_op_lim[:, np.newaxis] - y) ** 2, axis=0)
    )
    point["error_r"] = tostore(np.linalg.norm((r_op[:, np.newaxis] - r) ** 2, axis=0))
    point["error_r_lim"] = tostore(
        np.linalg.norm((r_op_lim[:, np.newaxis] - r) ** 2, axis=0)
    )


###### MAIN SCRIPT #####

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/test/low_dim/",
        help="Directory to read and write files",
    )
    parser.add_argument(
        "--compute",
        type=str,
        default="prediction",
        help="Which thing to analyse to make",
    )

    args = parser.parse_args()

    compute = args.compute

    timestr = time.strftime("%Y%m%d-%H%M%S")
    basepath = args.dir
    filename = basepath + timestr + "-" + compute + "_df.csv"

    dbase = pd.DataFrame()
    params = []
    df = pd.DataFrame()
    if compute == "database":
        df = compute_database(basepath)
    elif compute == "database_filter":
        dbase = load_dataframe("database", basepath)
        cols = list(dbase.columns)
        params = [c for c in cols if c.startswith("arg_")]
    else:
        dbase = load_dataframe("database_filter", basepath)
        cols = list(dbase.columns)
        params = [c for c in cols if c.startswith("arg_")]

    # On top of the main database
    if compute == "database_filter":
        lambdafunc = lambda x, params: all_normal(x)
        df = compute_across(dbase, ["arg_spike_scale"], params, lambdafunc)
    elif compute == "prediction":
        lambdafunc = lambda x, params: analyse_prediction(x)
        df = compute_per_simulation(dbase, params, lambdafunc)

    print("Saving results...")
    with open(filename, "wb") as f:
        df.to_csv(f)
