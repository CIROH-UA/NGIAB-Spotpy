import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import spotpy
import spotpy.objectivefunctions as objf
import xarray as xr
from dataretrieval import nwis
from spotpy.parameter import Uniform

from plots import (
    create_interactive_plots,
    plot_bestmodelrun,
    plot_parameter_correlation,
    plot_parameterInteraction,
    plot_parametertrace,
)

sys.path.append("/ngen/pyngiab")


def update_parameters(file_path, param_updates, model_type_name):
    """
    Update selected NOAH LSM parameters in the MPTABLE.TBL file.

    Parameters:
        directory_path (str): Path to the 'noah_om/parameters' directory.
        param_updates (dict): Keys are parameter names (e.g., 'MFSNO'), values are strings to insert.
    """
    with open(file_path, "r") as f:
        realization = json.load(f)
    models = realization["global"]["formulations"][0]["params"]["modules"]
    for model in models:
        if model["params"]["model_type_name"] == model_type_name:
            model["params"]["model_params"] = param_updates
            break
    with open(file_path, "w") as f:
        json.dump(realization, f, indent=4)


# === Utility Function to Retrieve and Preprocess USGS Streamflow ===
def process_usgs_streamflow(site, start, end, output_path=None):
    start = pd.to_datetime(start) - pd.Timedelta(days=1)
    end = pd.to_datetime(end) + pd.Timedelta(days=1)
    adjusted_start = start.strftime("%Y-%m-%d")
    adjusted_end = end.strftime("%Y-%m-%d")

    dfo_usgs = nwis.get_record(sites=site, service="iv", start=adjusted_start, end=adjusted_end)
    dfo_usgs.index = pd.to_datetime(dfo_usgs.index)
    dfo_usgs["Time"] = dfo_usgs.index.floor("h")
    dfo_usgs["00060"] = pd.to_numeric(dfo_usgs["00060"], errors="coerce")
    dfo_usgs_hr = dfo_usgs.groupby("Time")["00060"].mean().reset_index()
    dfo_usgs_hr["values"] = dfo_usgs_hr["00060"] / 35.3147
    dfo_usgs_hr = dfo_usgs_hr[["Time", "values"]]
    if output_path:
        dfo_usgs_hr.to_pickle(output_path)
    return dfo_usgs_hr


# === Wrapper to Set Up NextGen Model Execution ===
class NextGenSetup:
    def __init__(
        self,
        gage_id,
        start_date,
        end_date,
        training_start_date,
        observed_flow_path,
        troute_output_path,
        data_dir,
    ):
        self.gage_id = gage_id
        self.training_start_date = pd.to_datetime(training_start_date)
        self.end_date = pd.to_datetime(end_date)
        self.observed = pd.read_pickle(observed_flow_path)
        self.observed["Time"] = pd.to_datetime(self.observed["Time"]).dt.tz_localize(None)
        self.observed = self.observed[
            (self.observed["Time"] >= self.training_start_date)
            & (self.observed["Time"] <= self.end_date)
        ]
        self.observed = self.observed.set_index("Time")
        self.troute_output_path = troute_output_path
        self.realization_path = Path(data_dir) / "config" / "realization.json"

    def write_config(self, params):
        param_map = {
            "b": params[0],
            "satpsi": params[1],
            "satdk": params[2],
            "maxsmc": params[3],
            "expon": params[4],
            "slope": params[5],
            "Kn": params[6],
            "Klf": params[7],
        }

        update_parameters(self.realization_path, param_map, "CFE")

        # Create updated NOAH parameters dictionary
        noah_param_updates = {
            "MFSNO": params[8],  # Pass float directly
            "MP": params[9],
            "RSURF_EXP": params[10],
            # "SNOW_EMIS": params[11],
            "CWP": params[11],
            "VCMX25": params[12],
            "RSURF_SNOW": params[13],
            "SCAMAX": params[14],
        }

        update_parameters(self.realization_path, noah_param_updates, "NoahOWP")

        # Append run info to summary_df

    def run_model(self, data_dir):
        troute_output_folder = Path(data_dir) / "outputs" / "troute"
        for file in troute_output_folder.glob("*.nc"):
            file.unlink()
            print("T-route has been removed from previous run")
        try:
            # model = PyNGIAB(data_dir, serial_execution_mode=True)
            # model.run()
            command = f'docker run --rm -it -v "{data_dir}:/ngen/ngen/data" joshcu/ngiab:fast_cal /ngen/ngen/data/ auto 100 local'
            # command = f'docker run --rm -it -v "{data_dir}:/ngen/ngen/data" joshcu/ngiab:fast /ngen/ngen/data/ auto 100 local'
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)
            print("Next Gen run complete.")
            # route_command = f"route_rs {data_dir}"
            # subprocess.run(route_command, shell=True, stdout=subprocess.DEVNULL)
            # print("routing complete.")
        except:
            print("Next Gen run failed.")

    def evaluate(self, feature_id):
        ds = xr.open_dataset(self.troute_output_path)
        simulated = ds["flow"].sel(feature_id=feature_id).values
        simulated = simulated[ds["time"] >= self.training_start_date]
        simulated = simulated[: len(self.observed) - 1]
        return simulated


# === SPOTPY Setup Class for Calibration ===
class SpotpySetup:
    # CFE model parmaters
    soil_params_b = Uniform(2.0, 15.0)
    satpsi = Uniform(0.03, 0.955)
    satdk = Uniform(0.0000001, 0.000726)
    maxsmc = Uniform(0.5, 0.8)
    expon = Uniform(1.0, 8.0)
    slope = Uniform(0.0, 1.0)
    K_nash_subsurface = Uniform(0.01, 1.0)
    K_lf = Uniform(0.005, 1.0)

    # Additional NOAH OWP Modular parameters
    MFSNO = Uniform(low=1.0, high=5.0)  # multiplier on snowfall melt factor
    MP = Uniform(3.6, 12.6)
    RSURF_EXP = Uniform(4.5, 8.5)
    # SNOW_EMIS = Uniform(low=0.90, high=1.0)  # snow emissivity
    CWP = Uniform(0.09, 0.36)
    VCMX25 = Uniform(24.0, 112.0)
    RSURF_SNOW = Uniform(0.001, 50.0)
    SCAMAX = Uniform(0.7, 1.0)

    def __init__(self, model_setup, data_dir, feature_id, invert_objective, objective_function):
        self.obj_func = objective_function
        self.invert_objective = invert_objective
        self.model = model_setup
        self.data_dir = data_dir
        self.feature_id = feature_id
        self.run_id = 0

        # Ensure spotpy directory exists
        self.output_dir = f"{data_dir}/spotpy"
        os.makedirs(f"{self.output_dir}/plots/iterations", exist_ok=True)

    def simulation(self, vector):
        self.current_params = vector
        self.model.write_config(vector)
        self.model.run_model(self.data_dir)
        return self.model.evaluate(self.feature_id)

    def evaluation(self):
        return self.model.observed.values.squeeze()[1:]

    def objectivefunction(self, simulation, evaluation):
        if len(simulation) != len(evaluation):
            raise ValueError("simulation and observation are not equal length")

        if not self.obj_func:
            # This is used if not overwritten by user
            like = objf.rmse(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)

        # Plot each calibration iteration
        plt.figure(figsize=(10, 4))
        plt.plot(evaluation, label="Observed", color="black")
        plt.plot(simulation, label="Simulated", linestyle="--")
        plt.text(
            0.5, 0.9, f"objective_function: {like:.2f}", transform=plt.gca().transAxes, fontsize=12
        )
        plt.legend()
        plt.title(f"CFE-NOM-TR Streamflow for Gage {self.model.gage_id} - Run {self.run_id}")
        plt.xlabel("Time step")
        plt.ylabel("Streamflow [m3/sec]")
        plt.savefig(
            f"{self.output_dir}/plots/iterations/spotpy_run_{str(self.run_id).zfill(4)}.png"
        )
        plt.close()

        self.run_id += 1
        if self.invert_objective:
            return -like
        else:
            return like


def plot_results(results, observation_data, output_dir):
    plot_parametertrace(results, output_dir)
    plot_parameterInteraction(results, output_dir)
    plot_bestmodelrun(results, observation_data, output_dir)
    plot_parameter_correlation(results, output_dir)
    create_interactive_plots(results, observation_data, output_dir)


# === Function to Run SPOTPY Calibration ===
def run_spotpy(
    gage_id,
    start_date,
    end_date,
    training_start_date,
    observed_flow_path,
    troute_output_path,
    data_dir,
    feature_id,
    algorithm,
    objective_function,
    repetitions=25,
    dds_trials=5,
):
    # Model setup
    model_setup = NextGenSetup(
        gage_id,
        start_date,
        end_date,
        training_start_date,
        observed_flow_path,
        troute_output_path,
        data_dir,
    )

    if objective_function == "KGE":
        best_is_higher = True
        obj_func = spotpy.objectivefunctions.kge
    elif objective_function == "RMSE":
        best_is_higher = False
        obj_func = spotpy.objectivefunctions.rmse

    if algorithm == "DDS":
        algorithm_maximizes = True
    elif algorithm == "SCE":
        algorithm_maximizes = False

    invert_objective = best_is_higher != algorithm_maximizes

    optimizer = SpotpySetup(model_setup, data_dir, feature_id, invert_objective, obj_func)
    db_name = f"{optimizer.output_dir}/spotpy_results_{algorithm}"
    # SCE hyperparameters
    if algorithm == "SCE":
        sampler = spotpy.algorithms.sceua(optimizer, dbname=db_name, dbformat="csv")
        sampler.sample(repetitions, ngs=20)

    ## ADD hyperparameters
    elif algorithm == "DDS":
        sampler = spotpy.algorithms.dds(optimizer, dbname=db_name, dbformat="csv")
        sampler.sample(repetitions, trials=int(dds_trials))
    # results = spotpy.analyser.load_csv_results(db_name)

    results = sampler.getdata()
    plot_results(results, optimizer.evaluation(), f"{data_dir}/spotpy/plots")

    best_params = spotpy.analyser.get_best_parameterset(results, maximize=best_is_higher)

    return best_params
