import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mpi4py.MPI as MPI
import numpy as np
import pandas as pd
import spotpy
import xarray as xr
import yaml
from spotpy.parameter import Uniform
from tensorboardX import SummaryWriter

from plots import (
    plot_bestmodelrun,
    plot_parameter_correlation,
    plot_parameterInteraction,
    plot_parametertrace,
)

sys.path.append("/ngen/pyngiab")

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


def parameters_available_bool(realization_path):
    """If parameters already exist in the realization file, use them
    as initial parameters. Only available for DDS algorithm."""
    with open(realization_path, "r") as f:
        config = json.load(f)

    models_list = ["CFE", "NoahOWP"]
    parameters_available = False
    models_config = config["global"]["formulations"][0]["params"]["modules"]
    parameters = {}
    for model_type_name in models_list:
        for model in models_config:
            if model["params"]["model_type_name"] == model_type_name:
                if "model_params" in model["params"].keys():
                    parameters_available = True
                    parameters.update(model["params"]["model_params"])
                    print(f"Model: {model_type_name} has parameters available.")
                else:
                    parameters_available = False
                    break

    if parameters_available:
        param_names = [
            "b",
            "satpsi",
            "satdk",
            "maxsmc",
            "refkdt",
            "expon",
            "slope",
            "max_gw_storage",
            "Kn",
            "Klf",
            "Cgw",
            "MFSNO",
            "MP",
            "RSURF_EXP",
            "SNOW_EMIS",
            "CWP",
            "VCMX25",
            "RSURF_SNOW",
            "SCAMAX",
        ]
        # just taking the parameters that are in param_names
        parameters = [v for k, v in parameters.items() if k in param_names]

    return parameters_available, parameters


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
        groups,
        merge_catchment,
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
        self.realization_path = data_dir / "config" / "realization.json"
        self.data_dir = data_dir
        self.groups = groups
        self.merge_catchment = merge_catchment
        self.execution_mode = execution_mode

    def write_config(self, realization_path_name, params):
        param_map = {
            "b": params[0],
            "satpsi": params[1],
            "satdk": params[2],
            "maxsmc": params[3],
            "refkdt": params[4],
            "expon": params[5],
            "slope": params[6],
            "max_gw_storage": params[7],
            "Kn": params[8],
            "Klf": params[9],
            "Cgw": params[10],
        }

        update_parameters(realization_path_name, param_map, "CFE")

        # Create updated NOAH parameters dictionary
        noah_param_updates = {
            "MFSNO": params[11],  # Pass float directly
            "MP": params[12],
            "RSURF_EXP": params[13],
            "CWP": params[14],
            "VCMX25": params[15],
            "RSURF_SNOW": params[16],
            "SCAMAX": params[17],
        }

        update_parameters(realization_path_name, noah_param_updates, "NoahOWP")


    
    def run_model(
        self, tmp_root, realization, troute_yaml, temp_ngen_output_dir, temp_troute_output_dir, groups
    ):
        # running nextgen simulation ro get lateral flows
        if self.merge_catchment:
            gpkg_path = Path("/ngen/ngen/data/config/merged.gpkg")
        else:
            gpkg_path = Path("/ngen/ngen/data/config") / f"{self.data_dir.name}_subset.gpkg"
        try:
            if self.execution_mode == "serial":
                # important note: number of cores exposed should be less than or equal to number of partitions.
                partition_file = next(self.data_dir.glob("*.json")).name
                cpu_count = partition_file.split(".")[0].split("_")[-1]
                cmd_base = f"docker run --rm --entrypoint mpirun -w /ngen/ngen/data -v {tmp_root}:/ngen/ngen/data awiciroh/ciroh-ngen-image -n {cpu_count} /dmod/bin/ngen-parallel"
                ngen_cmd = (
                    f" {gpkg_path} all {gpkg_path} all "
                    f"/ngen/ngen/data/config/{realization.name} /ngen/ngen/data/{partition_file} "
                )
                cmd = cmd_base + ngen_cmd
                print(cmd)
                # cmd = f"bmi-driver {self.data_dir} --hf {self.data_dir / 'config' / gpkg_path.name} --config {self.data_dir / 'config' / realization.name}"
                subprocess.call(
                    cmd, shell=True
                )

            else:
                cmd_base = f"docker run --rm --entrypoint /dmod/bin/ngen-serial -w /ngen/ngen/data -v {tmp_root}:/ngen/ngen/data awiciroh/ciroh-ngen-image"
                ngen_cmd = (
                    f" {gpkg_path} all {gpkg_path} all /ngen/ngen/data/config/{realization.name}"
                )
                cmd = cmd_base + ngen_cmd
                print(cmd)
                # cmd = f"bmi-driver {self.data_dir} -j 1 --hf {self.data_dir / 'config' / gpkg_path.name} --config {self.data_dir / 'config' / realization.name}"

                subprocess.call(
                    cmd, shell=True
                )
        except:
            raise RuntimeError("Next Gen Simulation failed.")

        if self.merge_catchment:
            # create symbolic link for actual lateral files to merged lateral files if merged catchment is true
            # create a merged directory inside temp_ngen_output_dir
            merged_lateral_dir = temp_ngen_output_dir / "merged"
            merged_lateral_dir.mkdir(exist_ok=True)
            # mv lat files from temp_ngen_directory to merged directory
            os.system(f"mv {temp_ngen_output_dir}/cat-*.csv {merged_lateral_dir}/")

            # groups[i] maps to merged/cat-i.csv by construction.
            for i, cat_ids in enumerate(groups):
                merged_file_name = f"cat-{i}.csv"
                for cat_id in cat_ids:
                    os.symlink(
                        temp_ngen_output_dir / "merged" / merged_file_name,
                        temp_ngen_output_dir / f"cat-{cat_id}.csv",
                    )

        # running troute simulation to get streamflow
        try:
            subset_gpkg = tmp_root / "config" / f"{self.data_dir.name}_subset.gpkg"
            cmd = (
                f"route_rs {tmp_root} {subset_gpkg} "
                f"{temp_ngen_output_dir} {temp_troute_output_dir} --num-threads 31"
            )
            subprocess.call(cmd, shell=True)
        except:
            raise RuntimeError("T-route run failed.")
        rank = MPI.COMM_WORLD.rank

        print(f"Rank {rank} completed troute simulation.")

        self.troute_output_path = temp_troute_output_dir / self.troute_output_path.name
        if not self.troute_output_path.exists():
            print(f"Rank {rank} doesn't have troute output file. ####")
            raise RuntimeError("Nextgen Run failed. Couldn't find troute file.")

    def evaluate(self, tmp_root, feature_id):
        ds = xr.open_dataset(self.troute_output_path)
        simulated = ds["flow"].sel(feature_id=feature_id).values
        actual_start = min(self.training_start_date, self.observed.index[0])
        simulated = simulated[ds["time"] >= actual_start]
        simulated = simulated[: len(self.observed) - 1]
        shutil.rmtree(tmp_root, ignore_errors=True)
        return simulated


# === SPOTPY Setup Class for Calibration with TensorBoard ===
class SpotpySetup:
    # CFE model parameters
    soil_params_b = Uniform(2.0, 15.0, optguess=4.05)
    satpsi = Uniform(0.03, 0.955, optguess=0.355)
    satdk = Uniform(0.0000001, 0.000726, optguess=0.00000338)  # hit min
    maxsmc = Uniform(0.16, 0.59, optguess=0.439)  # hit max set to 0.8
    refkdt = Uniform(0.1, 4.0, optguess=1.0)  ######new
    expon = Uniform(1.0, 8.0, optguess=3.0)
    slope = Uniform(0.0, 1.0, optguess=0.1)
    max_gw_storage = Uniform(0.01, 0.25, optguess=0.05)  ######### new
    K_nash_subsurface = Uniform(0.0, 1.0, optguess=0.03)
    K_lf = Uniform(0.0, 1.0, optguess=0.01)
    Cgw = Uniform(0.0000018, 0.0018, optguess=0.000018)

    # # Additional NOAH OWP Modular parameters
    MFSNO = Uniform(0.5, 4.0, optguess=2.0)  # multiplier on snowfall melt factor
    MP = Uniform(3.6, 12.6, optguess=9.0)  # hit max
    RSURF_EXP = Uniform(1.0, 6.0, optguess=5.0)  # hit max
    # SNOW_EMIS = Uniform(0.90, 1.0)  # snow emissivity
    CWP = Uniform(0.09, 0.36, optguess=0.18)
    VCMX25 = Uniform(24.0, 112.0, optguess=52.2)
    RSURF_SNOW = Uniform(0.136, 100.0, optguess=50.0)  # hit min
    SCAMAX = Uniform(0.7, 1.0, optguess=0.9)

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
        self.calibration_dir = data_dir / "Calibration"
        self.calibration_dir.mkdir(exist_ok=True)
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
            "refkdt",
            "expon",
            "slope",
            "max_gw_storage",
            "K_nash_subsurface",
            "K_lf",
            "Cgw",
            "MFSNO",
            "MP",
            "RSURF_EXP",
            # "SNOW_EMIS",
            "CWP",
            "VCMX25",
            "RSURF_SNOW",
            "SCAMAX",
        ]

        # Ensure spotpy directory exists
        self.output_dir = Path(data_dir) / "spotpy"
        (self.output_dir / "plots" / "iterations").mkdir(parents=True, exist_ok=True)

    def _create_process_temp_dir(self) -> Path:
        """
        Create a temporary directory that mirrors data_dir for an individual MPI process.

        Directory structure created:
            <tmpdir>/
                config/          <- files copied from data_dir/config
                forcings/        <- forcings files hard-linked from data_dir/forcings
                metadata/        <- files copied from data_dir/metadata
                outputs/
                    ngen/
                    troute/

        Returns
        -------
        Path
            Root of the temporary mirror directory.
        """
        tmp_root = Path(tempfile.mkdtemp(dir=self.calibration_dir))

        # --- config: full copy so each process can mutate its own files freely ---
        shutil.copytree(self.data_dir / "config", tmp_root / "config")

        # --- metadata: full copy ---
        metadata_src = self.data_dir / "metadata"
        shutil.copytree(metadata_src, tmp_root / "metadata")

        # --- forcings: hard-link the two large NetCDF files to avoid duplication ---
        forcings_dst = tmp_root / "forcings"
        forcings_dst.mkdir(parents=True)
        for nc_file in (self.data_dir / "forcings").iterdir():
            os.link(nc_file, forcings_dst / nc_file.name)

        # --- outputs: empty dirs ready for ngen / troute ---
        (tmp_root / "outputs" / "ngen").mkdir(parents=True)
        (tmp_root / "outputs" / "troute").mkdir(parents=True)

        #if a partition file exist, link them as well
        partition_file = next(self.data_dir.glob("*.json"), None)
        if partition_file:
            os.link(partition_file, tmp_root / partition_file.name)

        return tmp_root


    def simulation(self, vector):
        self.current_params = vector

        tmp_root = self._create_process_temp_dir()
        realization_path   = tmp_root / "config" / "realization.json"
        troute_config_path = tmp_root / "config" / "troute.yaml"
        ngen_output_dir    = tmp_root / "outputs" / "ngen"
        troute_output_dir  = tmp_root / "outputs" / "troute"
        self.model.write_config(realization_path, vector)
        self.model.run_model(
            tmp_root,
            realization_path,
            troute_config_path,
            ngen_output_dir,
            troute_output_dir,
            self.model.groups,
        )
        return self.model.evaluate(tmp_root, self.feature_id)

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

        # # Log to TensorBoard if writer is available
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

            # Log hydrographs periodically (every 2 iterations)
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
    plot_parametertrace(results=results, output_folder=output_dir)
    plot_parameterInteraction(results=results, output_folder=output_dir)
    plot_bestmodelrun(results=results, evaluation=observation_data, output_folder=output_dir)
    plot_parameter_correlation(results=results, output_folder=output_dir)


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
    rank,
    algorithm,
    objective_function,
    groups,
    merge_catchment,
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
        groups,
        merge_catchment=merge_catchment,
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

    if tensorboard_logdir is None:
        tensorboard_logdir = data_dir / "tensorboard_logs"
    run_name = f"{algorithm}_{objective_function}_{gage_id}_2017_10_02"
    run_log_dir = tensorboard_logdir / run_name
    writer = None
    # Only let rank 0 create and own the TensorBoard writer.
    if rank == 0:
        os.makedirs(run_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_log_dir)

    # Ensure rank 0 creates the run directory before workers proceed.
    MPI.COMM_WORLD.Barrier()

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
        model_setup,
        data_dir,
        feature_id,
        invert_objective,
        obj_func,
        writer,
        objective_function,
        execution_mode,
    )
    db_name = f"{str(optimizer.output_dir)}/spotpy_results_{algorithm}_{objective_function}"

    realization_path = data_dir / "config" / "realization.json"
    parameters_available, parameters = parameters_available_bool(realization_path)

    parameters_available = False
    # FIX ME: there are some issues with initial parameters (even with the case of calibrated parameters) not being in the range
    # so for now, parameters_available is set to false to avoid using them as initial parameters for DDS algorithm. This needs to
    # be fixed in the future to fully utilize the benefits of DDS algorithm.
    # parameters_available = False
    # SCE hyperparameters
    if algorithm == "SCE":
        if execution_mode == "serial":
            sampler = spotpy.algorithms.sceua(optimizer, dbname=db_name, dbformat="csv")
            sampler.sample(repetitions, ngs=5)
        else:
            sampler = spotpy.algorithms.sceua(
                optimizer, dbname=db_name, dbformat="csv", parallel="mpi"
            )
            sampler.sample(repetitions, ngs=max((number_of_cores - 1), 5))

    elif algorithm == "DDS":
        if execution_mode == "serial":
            sampler = spotpy.algorithms.dds(optimizer, dbname=db_name, dbformat="csv")
        else:
            sampler = spotpy.algorithms.dds(
                optimizer, dbname=db_name, dbformat="csv", parallel="mpi"
            )

        if parameters_available:
            parameters = np.array(parameters)
            sampler.sample(repetitions, trials=int(dds_trials), x_initial=parameters)
        else:
            sampler.sample(repetitions, trials=int(dds_trials))

    results = sampler.getdata()
    # Final results to TensorBoard
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=best_is_higher)

    print("*********CALIBRATION COMPLETE**********")
    print("***************************************")
    print("***************************************")
    print(f"*******BEST PARAMETERS**********: {best_params}")

    best_params_value = best_params[0]
    print(f"Updating the best parameters in the realization file: {realization_path}")
    param_map = {
        "b": best_params_value[0],
        "satpsi": best_params_value[1],
        "satdk": best_params_value[2],
        "maxsmc": best_params_value[3],
        "refkdt": best_params_value[4],
        "expon": best_params_value[5],
        "slope": best_params_value[6],
        "max_gw_storage": best_params_value[7],
        "Kn": best_params_value[8],
        "Klf": best_params_value[9],
        "Cgw": best_params_value[10],
    }
    update_parameters(realization_path, param_map, "CFE")
    # Create updated NOAH parameters dictionary
    noah_param_updates = {
        "MFSNO": best_params_value[11],  # Pass float directly
        "MP": best_params_value[12],
        "RSURF_EXP": best_params_value[13],
        "CWP": best_params_value[14],
        "VCMX25": best_params_value[15],
        "RSURF_SNOW": best_params_value[16],
        "SCAMAX": best_params_value[17],
    }
    update_parameters(realization_path, noah_param_updates, "NoahOWP")

    # # Log final best parameters
    if writer:
        for i, param_name in enumerate(optimizer.param_names):
            if i < len(best_params[0]):
                writer.add_scalar(f"FinalBestParameters/{param_name}", best_params[0][i], 0)
        writer.close()

    # # Generate standard plots
    plot_results(results, optimizer.evaluation(), data_dir / "spotpy" / "plots")

    print(f"\nTensorBoard logs saved to: {run_log_dir}")
    print(f"Run 'tensorboard --logdir={tensorboard_logdir}' to view results")

    return best_params
