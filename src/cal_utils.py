from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI
import numpy as np
import pandas as pd
from flush_output import spotpy_stdout_control, suppress_spotpy_syntax_warnings
from datetime import datetime
suppress_spotpy_syntax_warnings()
import spotpy
import xarray as xr
from tensorboardX import SummaryWriter
from helper import *

# === Wrapper to Set Up NextGen Model Execution ===
class NextGenSetup:
    def __init__(
        self,
        gage_id: str,
        start_date: str,
        end_date: str,
        training_start_date: str,
        observed_flow_path: str | Path,
        troute_output_path: Path,
        data_dir: Path,
        groups: Any,
        param_to_model: dict[str, str],
        merge_catchment: bool,
        execution_mode: str = "parallel",
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
        self.param_to_model = param_to_model
        self.merge_catchment = merge_catchment
        self.execution_mode = execution_mode
    
    def run_model(
        self,
        tmp_root: Path,
        realization: Path,
        troute_yaml: Path,
        temp_ngen_output_dir: Path,
        temp_troute_output_dir: Path,
        groups: Any,
    ) -> None:
        # running nextgen simulation ro get lateral flows
        rank = MPI.COMM_WORLD.rank
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
                subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

            else:
                cmd_base = f"docker run --rm --entrypoint /dmod/bin/ngen-serial -w /ngen/ngen/data -v {tmp_root}:/ngen/ngen/data awiciroh/ciroh-ngen-image"
                ngen_cmd = (
                    f" {gpkg_path} all {gpkg_path} all /ngen/ngen/data/config/{realization.name}"
                )
                cmd = cmd_base + ngen_cmd
                # cmd = f"bmi-driver {self.data_dir} -j 1 --hf {self.data_dir / 'config' / gpkg_path.name} --config {self.data_dir / 'config' / realization.name}"

                subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Rank {rank} failed to run ngen simulation.")
            restore_data_dir(data_dir=self.data_dir)
            MPI.COMM_WORLD.Abort(rank)

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
            subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Rank {rank} failed to run troute simulation.")
            restore_data_dir(data_dir=self.data_dir)
            MPI.COMM_WORLD.Abort(rank)
        
        self.troute_output_path = temp_troute_output_dir / self.troute_output_path.name
        if not self.troute_output_path.exists():
            print(f"Rank {rank} doesn't have troute output file. ####\n\n")
            restore_data_dir(data_dir=self.data_dir)
            MPI.COMM_WORLD.Abort(rank)

    def evaluate(self, tmp_root: Path, feature_id: int) -> np.ndarray:
        ds = xr.open_dataset(self.troute_output_path)
        simulated = ds["flow"].sel(feature_id=feature_id).values
        actual_start = min(self.training_start_date, self.observed.index[0])
        simulated = simulated[ds["time"] >= actual_start]
        simulated = simulated[: len(self.observed) - 1]
        shutil.rmtree(tmp_root, ignore_errors=True)
        return simulated


# === SPOTPY Setup Class for Calibration with TensorBoard ===
class SpotpySetup:
    def __init__(
        self,
        model_setup: NextGenSetup,
        data_dir: Path,
        feature_id: int,
        invert_objective: bool,
        objective_function: Any,
        calibration_dir: Path,
        writer: Any = None,
        objective_function_name: str | None = None,
        execution_mode: str = "parallel",
    ):
        self.obj_func = objective_function
        self.objective_function_name = objective_function_name
        self.invert_objective = invert_objective
        self.model = model_setup
        self.data_dir = data_dir
        self.calibration_dir = calibration_dir
        self.temp_runs = self.calibration_dir / "temp_runs"
        self.feature_id = feature_id
        self.run_id = 0
        self.writer = writer
        self.execution_mode = execution_mode
        self.best_objective = float("inf") if not invert_objective else float("-inf")

        # Ensure spotpy directory exists
        self.output_dir = calibration_dir / "spotpy"

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
        tmp_root = Path(tempfile.mkdtemp(dir=self.temp_runs))

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
        
        #do a cp instead
        if partition_file:
            shutil.copy2(partition_file, tmp_root / partition_file.name)

        return tmp_root


    def simulation(self, vector: Sequence[float]) -> np.ndarray:
        self.current_params = vector

        tmp_root = self._create_process_temp_dir()
        realization_path   = tmp_root / "config" / "realization.json"
        troute_config_path = tmp_root / "config" / "troute.yaml"
        ngen_output_dir    = tmp_root / "outputs" / "ngen"
        troute_output_dir  = tmp_root / "outputs" / "troute"
        write_config(realization_path, vector, self.model.param_to_model)
        self.model.run_model(
            tmp_root,
            realization_path,
            troute_config_path,
            ngen_output_dir,
            troute_output_dir,
            self.model.groups,
        )
        return self.model.evaluate(tmp_root, self.feature_id)

    def evaluation(self) -> np.ndarray:
        return self.model.observed.values.squeeze()[1:]

    def objectivefunction(self, simulation: np.ndarray, evaluation: np.ndarray) -> float:
        if len(simulation) != len(evaluation):
            raise ValueError("simulation and observation are not equal length")

        if np.sum(evaluation) == 0:
            #since streamflow cant be negative, this means all streamflow value here is 0
            evaluation = evaluation + np.float64(1e-10)
        
        objective_metric = self.obj_func(evaluation, simulation)

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

            # Log hydrographs periodically (every 10 iterations)
            if self.run_id % 10 == 0:
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
                self.writer.flush()

        if self.invert_objective:
            if self.objective_function_name == "KGE":
                objective_metric = 1 - objective_metric
            else:
                objective_metric = -objective_metric
        else:
            if self.objective_function_name == "KGE":
                objective_metric = objective_metric - 1

        self.run_id += 1
        return objective_metric


# === Function to Run SPOTPY Calibration with TensorBoard ===
def run_spotpy(
    gage_id: str,
    start_date: str,
    end_date: str,
    training_start_date: str,
    observed_flow_path: str | Path,
    troute_output_path: Path,
    data_dir: Path,
    feature_id: int,
    rank: int,
    algorithm: str,
    objective_function: str,
    groups: Any,
    merge_catchment: bool,
    calibration_params: dict,
    tensorboard_logdir: Path,
    repetitions: int = 25,
    dds_trials: int = 5,
    execution_mode: str = "parallel",
    number_of_cores: int = 4
) -> Any:
    
    param_to_model = {name: model for model, names in calibration_params.items() for name in names}
    params_names_list = []
    # Add spotpy parameters to the optimizer so spotpy can sample them.
    # Doing it like this makes it easier to change and log parameter values.
    for _, params in calibration_params.items():
        for _name, _param in params.items():
            setattr(SpotpySetup, _name, _param)
            params_names_list.append(_name)

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
        param_to_model,
        merge_catchment=merge_catchment,
        execution_mode=execution_mode,
    )  


    if objective_function == "RMSE":
        best_is_higher = False
        obj_func = spotpy.objectivefunctions.rmse
    else:
        best_is_higher = True
        obj_func = spotpy.objectivefunctions.kge

    # if objective_function == "KGE":
    #     best_is_higher = True
    #     obj_func = spotpy.objectivefunctions.kge
    # elif objective_function == "RMSE":
    #     best_is_higher = False
    #     obj_func = spotpy.objectivefunctions.rmse

    if algorithm == "SCE":
        algorithm_maximizes = False
    else:
        algorithm_maximizes = True
    # if algorithm == "DDS":
    #     algorithm_maximizes = True
    # elif algorithm == "SCE":
    #     algorithm_maximizes = False

    invert_objective = best_is_higher != algorithm_maximizes

    calibration_dir = data_dir.parent.parent

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_name = f"{algorithm}_{objective_function}_{gage_id}_{timestamp}"
    run_log_dir = tensorboard_logdir / run_name
    writer = None
    # Only let rank 0 create and own the TensorBoard writer.
    if rank == 0:
        os.makedirs(run_log_dir, exist_ok=True)
        # Use aggressive flushing to reduce the chance of "missing" figures due to buffering.
        try:
            writer = SummaryWriter(log_dir=str(run_log_dir), max_queue=1, flush_secs=1)
        except TypeError:
            writer = SummaryWriter(log_dir=str(run_log_dir))

    # Ensure rank 0 creates the run directory before workers proceed.
    MPI.COMM_WORLD.Barrier()

    optimizer = SpotpySetup(
        model_setup,
        data_dir,
        feature_id,
        invert_objective,
        obj_func,
        calibration_dir,
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
    # SCE hyperparameters
    if algorithm == "SCE":
        if execution_mode == "serial":
            sampler = spotpy.algorithms.sceua(optimizer, dbname=db_name, dbformat="csv")
            sampler.sample(repetitions, ngs=5)
        else:
            sampler = spotpy.algorithms.sceua(
                optimizer, dbname=db_name, dbformat="csv", parallel="mpi"
            )
            with spotpy_stdout_control(rank=rank, execution_mode=execution_mode):
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
            with spotpy_stdout_control(rank=rank, execution_mode=execution_mode):
                sampler.sample(repetitions, trials=int(dds_trials), x_initial=parameters)
        else:
            with spotpy_stdout_control(rank=rank, execution_mode=execution_mode):
                sampler.sample(repetitions, trials=int(dds_trials))

    results = sampler.getdata()
    # Final results to TensorBoard
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=best_is_higher)

    best_params_value = best_params[0]

    #redefine realization path to the main data directory
    realization_path = calibration_dir.parent / "config" / "realization.json"
    write_config(realization_path, best_params_value, param_to_model)

    if writer:
        # Log the parameter traces for all iterations from the SPOTPY CSV database.
        csv_path = Path(f"{db_name}.csv")
        log_parameters_from_spotpy_csv(writer, csv_path, params_names_list)
        writer.close()

    # # Generate standard plots
    plot_results(results, optimizer.evaluation(), calibration_dir / "spotpy" / "plots", objective_function, invert_objective)

    print(f"\nTensorBoard logs saved to: {run_log_dir}")
    print(f"Run 'tensorboard --logdir={tensorboard_logdir}' to view results\n\n")

    return best_params
