import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotpy
import xarray as xr
from dataretrieval import nwis
from spotpy.parameter import Uniform
from tensorboardX import SummaryWriter
import tempfile
import yaml
import shutil
from mpi4py import MPI

from plots import (
    create_interactive_plots,
    plot_bestmodelrun,
    plot_parameter_correlation,
    plot_parameterInteraction,
    plot_parametertrace,
)

sys.path.append("/ngen/pyngiab")

def update_output_path(realization_path_name, troute_config_file_name, temp_ngen_output_dir, temp_troute_output_dir):

    #updating troute and ngen output path in realization
    with open(realization_path_name, 'r') as f:
        data = json.load(f)
    data['output_root'] = os.path.join("outputs/ngen",os.path.basename(temp_ngen_output_dir))
    data['routing']['t_route_config_file_with_path'] = os.path.join("config",os.path.basename(troute_config_file_name))
    with open(realization_path_name, 'w') as f:
        json.dump(data, f, indent=4)

    #updating lateral input path (that comes from temp_ngen_output_dir) and stream output path in troute yaml file
    with open(troute_config_file_name, 'r') as f:
        data = yaml.safe_load(f)
    data['output_parameters']['stream_output']['stream_output_directory'] = os.path.join("outputs/troute",os.path.basename(temp_troute_output_dir))
    data['compute_parameters']['forcing_parameters']['qlat_input_folder'] = os.path.join("outputs/ngen",os.path.basename(temp_ngen_output_dir))
    with open(troute_config_file_name, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=100)


def update_parameters(file_path, param_updates, model_type_name):
    with open(file_path, "r") as f:
        realization = json.load(f)
    models = realization["global"]["formulations"][0]["params"]["modules"]
    for model in models:
        if model["params"]["model_type_name"] == model_type_name:
            model["params"]["model_params"] = param_updates
            break
    with open(file_path, "w") as f:
        json.dump(realization, f, indent=4)


def update_snow_emis(data_dir, value):
    """
    Update selected NOAH LSM parameters in the MPTABLE.TBL file.

    Parameters:
        directory_path (str): Path to the 'noah_om/parameters' directory.
        param_updates (dict): Keys are parameter names (e.g., 'MFSNO'), values are strings to insert.
    """
    file_path = Path(os.path.join(data_dir, "config/MPTABLE.TBL"))
    if not file_path.exists():
        os.system(f"touch {str(file_path)}")
        # raise FileNotFoundError(f"MPTABLE.TBL not found at {file_path}")

    with open(file_path, "r") as file:
        lines = file.readlines()
        # print(f"Updating parameters in {file_path}...")

        for i, line in enumerate(lines):
            if line.strip().startswith("SNOW_EMIS"):
                lines[i] = f"  SNOW_EMIS     = {value}\n"

    with open(file_path, "w") as file:
        file.writelines(lines)


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
        execution_mode="parallel",
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
        self.data_dir = data_dir
        self.execution_mode = execution_mode

    def write_config(self, realization_path_name, params):
        realization_path = Path(self.data_dir)/ "config" / realization_path_name

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

        update_parameters(realization_path, param_map, "CFE")

        # Create updated NOAH parameters dictionary
        noah_param_updates = {
            "MFSNO": params[8],  # Pass float directly
            "MP": params[9],
            "RSURF_EXP": params[10],
            # "SNOW_EMIS": params[11],
            "CWP": params[12],
            "VCMX25": params[13],
            "RSURF_SNOW": params[14],
            "SCAMAX": params[15],
        }

        update_parameters(realization_path, noah_param_updates, "NoahOWP")
        # update_snow_emis(self.data_dir, params[11])


    def run_model(self, gage_id, realization, troute_yaml, temp_ngen_output_dir, temp_troute_output_dir):
        try:
            if self.execution_mode == "serial":
                cmd_base = f"docker run --entrypoint /ngen/Sonam_NGEN.sh -v /home/slama/Documents/hf3_remap/hf3_remap/output/gage-10109001:/ngen/ngen/data slama07/ngen_parallel_realization:0.1 /ngen/ngen/data/ auto 100 local config/{os.path.basename(realization)}"
                subprocess.call(cmd_base, shell=True)

            else:
                gpkg_path = "/ngen/ngen/data/config/" + f"gage-{str(gage_id)}_subset.gpkg"
                cmd_base = f"docker run --entrypoint /dmod/bin/ngen-serial -w /ngen/ngen/data -v {self.data_dir}:/ngen/ngen/data awiciroh/ciroh-ngen-image"
                ngen_cmd = f" {gpkg_path} all {gpkg_path} all /ngen/ngen/data/config/{os.path.basename(realization)}"               
                subprocess.call(cmd_base + ngen_cmd, shell=True)
        except:
            raise RuntimeError("Next Gen run failed.")

        self.troute_output_path = os.path.join(temp_troute_output_dir, os.path.basename(self.troute_output_path))
        if not os.path.exists(self.troute_output_path):
            raise RuntimeError("Nextgen Run failed. Couldn't find troute file.")
        else:
            print("Nextgen run complete.")
            #remove realization file
            shutil.rmtree(temp_ngen_output_dir)
            os.remove(realization)
            os.remove(troute_yaml)  

    def evaluate(self, temp_troute_output_dir, feature_id):
        ds = xr.open_dataset(self.troute_output_path)
        simulated = ds["flow"].sel(feature_id=feature_id).values
        actual_start = min(self.training_start_date, self.observed.index[0])
        simulated = simulated[ds["time"] >= actual_start]
        simulated = simulated[: len(self.observed) - 1]
        shutil.rmtree(temp_troute_output_dir)
        return simulated


# === SPOTPY Setup Class for Calibration with TensorBoard ===
class SpotpySetup:
    # CFE model parameters
    soil_params_b = Uniform(2.0, 15.0)
    satpsi = Uniform(0.03, 0.955)
    satdk = Uniform(0.0000001, 0.000726)  # hit min
    maxsmc = Uniform(0.16, 1.0)  # hit max set to 0.8
    expon = Uniform(1.0, 8.0)
    slope = Uniform(0.0, 1.0)
    K_nash_subsurface = Uniform(0.01, 1.0)
    K_lf = Uniform(0.005, 1.0)

    # Additional NOAH OWP Modular parameters
    MFSNO = Uniform(0.5, 4.0)  # multiplier on snowfall melt factor
    MP = Uniform(3.6, 14.6)  # hit max
    RSURF_EXP = Uniform(1.0, 15.0)  # hit max
    SNOW_EMIS = Uniform(0.90, 1.0)  # snow emissivity
    CWP = Uniform(0.09, 0.36)
    VCMX25 = Uniform(24.0, 152.0)
    RSURF_SNOW = Uniform(0.0, 100.0)  # hit min
    SCAMAX = Uniform(0.7, 1.0)

    def __init__(
        self,
        model_setup,
        data_dir,
        feature_id,
        invert_objective,
        objective_function,
        writer=None,
        objective_function_name=None,
        execution_mode="parallel",
    ):
        self.obj_func = objective_function
        self.objective_function_name = objective_function_name
        self.invert_objective = invert_objective
        self.model = model_setup
        self.data_dir = data_dir
        self.feature_id = feature_id
        self.run_id = 0
        self.writer = writer
        self.execution_mode = execution_mode
        self.best_objective = float("inf") if not invert_objective else float("-inf")

        # Get parameter names for logging
        self.param_names = [
            "soil_params_b",
            "satpsi",
            "satdk",
            "maxsmc",
            "expon",
            "slope",
            "K_nash_subsurface",
            "K_lf",
            "MFSNO",
            "MP",
            "RSURF_EXP",
            "SNOW_EMIS",
            "CWP",
            "VCMX25",
            "RSURF_SNOW",
            "SCAMAX",
        ]

        # Ensure spotpy directory exists
        self.output_dir = f"{data_dir}/spotpy"
        os.makedirs(f"{self.output_dir}/plots/iterations", exist_ok=True)

    def simulation(self, vector):
        self.current_params = vector
        #cerate a temporary copy of realization file and yaml file for each process
        realization_path = Path(self.data_dir)/ "config" / "realization.json"
        with open(realization_path, 'r') as f:
            data = json.load(f)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir = os.path.join(self.data_dir, "config")) as temp_file_realization:
            json.dump(data, temp_file_realization, indent=4, ensure_ascii=False)
            temp_file_realization_name = temp_file_realization.name
            print(f"Temporary file created: {temp_file_realization_name}")

        troute_config_path = Path(self.data_dir) / "config" / "troute.yaml"
        with open(troute_config_path, 'r') as f:
            data = yaml.safe_load(f)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir = os.path.join(self.data_dir, "config")) as temp_file_yaml:
            yaml.dump(data, temp_file_yaml)
            temp_file_yaml_name = temp_file_yaml.name
            print(f"Temporary YAML file created: {temp_file_yaml_name}")

        #create temporary output directories for ngen and troute for each process
        temp_ngen_output_dir = tempfile.mkdtemp(dir = os.path.join(self.data_dir, "outputs/ngen"))
        temp_troute_output_dir = tempfile.mkdtemp(dir = os.path.join(self.data_dir, "outputs/troute"))

        print(f"Temporary Nextgen output directory: {temp_ngen_output_dir}")
        print(f"Temporary T-route output directory: {temp_troute_output_dir}")

        update_output_path(temp_file_realization_name, temp_file_yaml_name, temp_ngen_output_dir, temp_troute_output_dir)


        self.model.write_config(temp_file_realization_name, vector)
        self.model.run_model(self.model.gage_id, temp_file_realization_name, temp_file_yaml_name, temp_ngen_output_dir, temp_troute_output_dir)
        return self.model.evaluate(temp_troute_output_dir,self.feature_id)

    def evaluation(self):
        return self.model.observed.values.squeeze()[1:]

    def objectivefunction(self, simulation, evaluation):
        if len(simulation) != len(evaluation):
            raise ValueError("simulation and observation are not equal length")

        objective_metric = self.obj_func(evaluation, simulation)
        if self.invert_objective:
            if self.objective_function_name == "KGE":
                objective_metric = 1 - objective_metric
            else:
                objective_metric = -objective_metric
        else:
            if self.objective_function_name == "KGE":
                objective_metric = objective_metric - 1

        # Calculate additional metrics for TensorBoard
        rmse = spotpy.objectivefunctions.rmse(evaluation, simulation)
        kge = spotpy.objectivefunctions.kge(evaluation, simulation)
        mae = np.mean(np.abs(evaluation - simulation))
        nse = 1 - (
            np.sum((evaluation - simulation) ** 2) / np.sum((evaluation - np.mean(evaluation)) ** 2)
        )
        correlation = np.corrcoef(evaluation, simulation)[0, 1]



        # Log to TensorBoard if writer is available
        if self.writer:
            # Log objective function value
            self.writer.add_scalar("Metrics/Objective_Function", objective_metric, self.run_id)
            self.writer.add_scalar("Metrics/MAE", mae, self.run_id)
            self.writer.add_scalar("Metrics/KGE", kge, self.run_id)
            self.writer.add_scalar("Metrics/NSE", nse, self.run_id)
            self.writer.add_scalar("Metrics/RMSE", rmse, self.run_id)
            self.writer.add_scalar("Metrics/Correlation", correlation, self.run_id)


            # # Log parameters
            # for i, param_name in enumerate(self.param_names):
            #     if i < len(self.current_params):
            #         self.writer.add_scalar(
            #             f"Parameters/{param_name}", self.current_params[i], self.run_id
            #         )

            # Log hydrographs periodically (every 10 iterations)
            if self.run_id % 2 == 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(evaluation, label="Observed", color="black", linewidth=1.5)
                ax.plot(simulation, label="Simulated", linestyle="--", alpha=0.8)
                ax.legend()
                ax.set_title(f"Iteration {self.run_id} - Objective: {objective_metric:.3f}")
                ax.set_xlabel("Time step")
                ax.set_ylabel("Streamflow [m3/sec]")
                ax.grid(True, alpha=0.3)
                self.writer.add_figure("Hydrographs/Comparison", fig, self.run_id)
                plt.close(fig)

                # Log residuals
                residuals = evaluation - simulation
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.plot(residuals)
                ax1.set_title("Residuals Over Time")
                ax1.set_xlabel("Time step")
                ax1.set_ylabel("Residual [m3/sec]")
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

                ax2.hist(residuals, bins=30, edgecolor="black")
                ax2.set_title("Residual Distribution")
                ax2.set_xlabel("Residual [m3/sec]")
                ax2.set_ylabel("Frequency")
                ax2.grid(True, alpha=0.3)

                self.writer.add_figure("Residuals/Analysis", fig, self.run_id)
                plt.close(fig)

        self.run_id += 1
        return objective_metric


def plot_results(results, observation_data, output_dir):
    plot_parametertrace(results, output_dir)
    plot_parameterInteraction(results, output_dir)
    plot_bestmodelrun(results, observation_data, output_dir)
    plot_parameter_correlation(results, output_dir)
    create_interactive_plots(results, observation_data, output_dir)


# === Function to Run SPOTPY Calibration with TensorBoard ===
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
    execution_mode="parallel",
    number_of_cores=4,
    tensorboard_logdir=None,
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
        execution_mode=execution_mode,
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

    # Set up TensorBoard writer
    if tensorboard_logdir is None:
        tensorboard_logdir = f"{data_dir}/tensorboard_logs"

    run_name = f"{algorithm}_{objective_function}_{gage_id}_2017_10_02"
    writer = SummaryWriter(log_dir=f"{tensorboard_logdir}/{run_name}")

    # Log hyperparameters
    hparams = {
        "algorithm": algorithm,
        "objective_function": objective_function,
        "repetitions": repetitions,
        "gage_id": gage_id,
        "start_date": str(start_date),
        "end_date": str(end_date),
    }
    if algorithm == "DDS":
        hparams["dds_trials"] = dds_trials

    # writer.add_hparams(hparams, {"dummy": 0})  # TensorBoard requires at least one metric
    optimizer = SpotpySetup(
        model_setup, data_dir, feature_id, invert_objective, obj_func, writer, objective_function, execution_mode
    )
    db_name = f"{optimizer.output_dir}/spotpy_results_{algorithm}_{objective_function}"

    # SCE hyperparameters
    if algorithm == "SCE":
        if execution_mode == "serial":
            sampler = spotpy.algorithms.sceua(optimizer, dbname=db_name, dbformat="csv")
            sampler.sample(repetitions, ngs=5)
        else:
            sampler = spotpy.algorithms.sceua(optimizer, dbname=db_name, dbformat="csv", parallel="mpi")
            sampler.sample(repetitions, ngs=max((number_of_cores - 1), 5))

    elif algorithm == "DDS":
        if execution_mode == "serial":
            sampler = spotpy.algorithms.dds(optimizer, dbname=db_name, dbformat="csv")
        else:
            sampler = spotpy.algorithms.dds(optimizer, dbname=db_name, dbformat="csv", parallel="mpi")
        sampler.sample(repetitions, trials=int(dds_trials))

    results = sampler.getdata()
    # Final results to TensorBoard
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=best_is_higher)

    print("*********CALIBRATION COMPLETE**********")
    print("***************************************")
    print("***************************************")
    print(f"*******BEST PARAMETERS**********: {best_params}")


    best_params_value = best_params[0]
    realization_path = Path(data_dir)/ "config" / "realization.json"

    param_map = {
        "b": best_params_value[0],
        "satpsi": best_params_value[1],
        "satdk": best_params_value[2],
        "maxsmc": best_params_value[3],
        "expon": best_params_value[4],
        "slope": best_params_value[5],
        "Kn": best_params_value[6],
        "Klf": best_params_value[7],
        }
    update_parameters(realization_path, param_map, "CFE")
    # Create updated NOAH parameters dictionary
    noah_param_updates = {
        "MFSNO": best_params_value[8],  # Pass float directly
        "MP": best_params_value[9],
        "RSURF_EXP": best_params_value[10],
        # "SNOW_EMIS": best_params_value[11],
        "CWP": best_params_value[12],
        "VCMX25": best_params_value[13],
        "RSURF_SNOW": best_params_value[14],
        "SCAMAX": best_params_value[15],
        }
    update_parameters(realization_path, noah_param_updates, "NoahOWP")

    # Log final best parameters
    for i, param_name in enumerate(optimizer.param_names):
        if i < len(best_params[0]):
            writer.add_scalar(f"FinalBestParameters/{param_name}", best_params[0][i], 0)
    # Close TensorBoard writer
    writer.close()

    # Generate standard plots
    # plot_results(results, optimizer.evaluation(), f"{data_dir}/spotpy/plots")

    print(f"\nTensorBoard logs saved to: {tensorboard_logdir}/{run_name}")
    print(f"Run 'tensorboard --logdir={tensorboard_logdir}' to view results")

    return best_params