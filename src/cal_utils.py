from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence
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
import glob
import sqlite3

# === Wrapper to Set Up NextGen Model Execution ===
class NextGenSetup:
    def __init__(
        self,
        gage_id: str,
        start_date: str,
        end_date: str,
        training_start_date: str,
        target_variables: dict,
        troute_output_path: Path,
        data_dir: Path,
        groups: Any,
        param_to_model: dict[str, str],
        merge_catchment: bool,
        execution_mode: str = "parallel",
    ):
        self.gage_id = gage_id
        self.training_start_date = pd.to_datetime(training_start_date)
        #works properly with the ET and SWE
        self.end_date = pd.to_datetime(end_date) - pd.Timedelta(days=1)
        self.target_variables = target_variables
        self.troute_output_path = troute_output_path
        self.realization_path = data_dir / "config" / "realization.json"
        self.data_dir = data_dir
        self.groups = groups
        self.param_to_model = param_to_model
        self.merge_catchment = merge_catchment
        self.execution_mode = execution_mode

        for var_name, var_info in target_variables.items():
            if var_name == "streamflow":
                self.observed_streamflow = pd.read_csv(var_info["observed_data_path"])
                self.observed_streamflow = adjust_date_index(self.observed_streamflow, self.training_start_date, self.end_date, "streamflow")
            elif var_name == "ET":
                self.observed_ET = pd.read_csv(var_info["observed_data_path"])
                self.observed_ET = adjust_date_index(self.observed_ET, self.training_start_date, self.end_date, "ET")
            elif var_name == "SWE":
                self.observed_SWE = pd.read_csv(var_info["observed_data_path"])
                self.observed_SWE = adjust_date_index(self.observed_SWE, self.training_start_date, self.end_date, "SWE")
            else:
                destroy_all_processes(f"Unsupported target variable: {var_name}")
                return
    
    def run_model(
        self,
        tmp_root: Path,
        realization: Path,
        troute_yaml: Path,
        temp_ngen_output_dir: Path,
        temp_troute_output_dir: Path,
        groups: Any,
    ) -> None:
        #running nextgen simulation ro get lateral flows
        if self.merge_catchment:
            gpkg_path = Path("/ngen/ngen/data/config/merged.gpkg")
        else:
            gpkg_path = Path("/ngen/ngen/data/config") / f"{self.data_dir.name}_subset.gpkg"

        for failure_counter in range(10):
            try:
                if self.execution_mode == "serial":
                    partition_file = next(self.data_dir.glob("*.json")).name
                    cpu_count = partition_file.split(".")[0].split("_")[-1]

                    cmd_base = (
                        f"docker run --rm --entrypoint mpirun "
                        f"-w /ngen/ngen/data "
                        f"-v {tmp_root}:/ngen/ngen/data "
                        f"awiciroh/ciroh-ngen-image -n {cpu_count} "
                        f"/dmod/bin/ngen-parallel"
                    )

                    ngen_cmd = (
                        f" {gpkg_path} all {gpkg_path} all "
                        f"/ngen/ngen/data/config/{realization.name} "
                        f"/ngen/ngen/data/{partition_file} "
                    )

                else:
                    cmd_base = (
                        f"docker run --rm --entrypoint /dmod/bin/ngen-serial "
                        f"-w /ngen/ngen/data "
                        f"-v {tmp_root}:/ngen/ngen/data "
                        f"awiciroh/ciroh-ngen-image"
                    )

                    ngen_cmd = (
                        f" {gpkg_path} all {gpkg_path} all "
                        f"/ngen/ngen/data/config/{realization.name}"
                    )

                cmd = cmd_base + ngen_cmd
                subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

            except subprocess.CalledProcessError:
                if failure_counter < 9:
                    print(f"Failed ngen simulation: {failure_counter+1}/10. Rerunning again\n")
                    continue
                else:
                    restore_data_dir(data_dir=self.data_dir)
                    destroy_all_processes(f"Failed to run ngen simulation.")

            if self.merge_catchment:
                merged_lateral_dir = temp_ngen_output_dir / "merged"
                merged_lateral_dir.mkdir(exist_ok=True)

                os.system(f"mv {temp_ngen_output_dir}/cat-*.csv {merged_lateral_dir}/")

                for group_idx, cat_ids in enumerate(groups):
                    merged_file_name = f"cat-{group_idx}.csv"

                    for cat_id in cat_ids:
                        os.symlink(
                            temp_ngen_output_dir / "merged" / merged_file_name,
                            temp_ngen_output_dir / f"cat-{cat_id}.csv",
                        )

            if "streamflow" in self.target_variables:
                try:
                    subset_gpkg = tmp_root / "config" / f"{self.data_dir.name}_subset.gpkg"

                    cmd = (
                        f"rs-route {self.data_dir} --hf {subset_gpkg} -k route-rs "
                        f"-i {temp_ngen_output_dir} "
                        f"-o {temp_troute_output_dir}"
                    )

                    subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)

                except subprocess.CalledProcessError:
                    if failure_counter < 9:
                        print(f"Failed routing simulation: {failure_counter+1}/10. Rerunning again\n")
                        continue
                    else:
                        restore_data_dir(data_dir=self.data_dir)
                        destroy_all_processes(f"Failed to run troute simulation.")

                self.troute_output_path = temp_troute_output_dir / self.troute_output_path.name

                if not self.troute_output_path.exists():
                    if failure_counter < 9:
                        print(f"Failed routing simulation: {failure_counter+1}/10. Rerunning again\n")
                        continue
                    else:
                        restore_data_dir(data_dir=self.data_dir)
                        destroy_all_processes(f"Doesn't have troute output file. ####\n\n")
                        
            break



    def evaluate_streamflow(self, tmp_root: Path, feature_id: int) -> np.ndarray:
        ds = xr.open_dataset(self.troute_output_path)
        sim_df = pd.Series(
            ds["flow"].sel(feature_id=feature_id).values,
            index=pd.DatetimeIndex(ds["time"].values)
        )
        simulated = sim_df.reindex(self.observed_streamflow.index).values
        simulated = np.array(simulated)
        return simulated
    
    def evaluate_ET_SWE(self, tmp_root: Path, column: str) -> np.ndarray:
        if self.merge_catchment:
            gpkg_path = tmp_root / "config" / "merged.gpkg"
        else:
            gpkg_path = tmp_root / "config" / f"{self.data_dir.name}_subset.gpkg"
        with sqlite3.connect(gpkg_path) as conn:
            cmd = (
                f"SELECT divide_id, areasqkm FROM 'divides'"
            )
            results = conn.execute(cmd).fetchall()
            area_lookup = {str(divide_id): areasqkm for divide_id, areasqkm in results}
            weighted_sum = None

        total_area = 0.0

        if self.merge_catchment:
            merged_lateral_dir = tmp_root / "outputs" / "ngen" / "merged"
            files = glob.glob(os.path.join(merged_lateral_dir, "cat-*.csv"))
        else:
            files = glob.glob(os.path.join(tmp_root / "outputs" / "ngen", "cat-*.csv"))

        for _, file in enumerate(files, start=1):

            cat_id = os.path.basename(file)
            # cat_id = cat_id.replace("cat-", "")
            cat_id = cat_id.replace(".csv", "")

            if cat_id not in area_lookup:
                print(f"Skipping {cat_id}: no area found")
                continue

            area = area_lookup[cat_id]

            df = pd.read_csv(
                file,
                usecols=["Time", column]
            )
            df = df[:len(df)-1]

            df["Time"] = pd.to_datetime(df["Time"])

            if column == "ACTUAL_ET":
                daily = (
                    df.groupby(df["Time"].dt.floor("D"))["ACTUAL_ET"]
                    .sum()
                )
            else:
                daily = (
                    df.groupby(df["Time"].dt.floor("D"))["SNEQV"]
                    .mean()
                )
            weighted_daily = daily * area

            if weighted_sum is None:
                weighted_sum = weighted_daily
            else:
                weighted_sum = weighted_sum.add(
                    weighted_daily,
                    fill_value=0
                )

            total_area += area

        weighted_mean = weighted_sum / total_area

        result = pd.DataFrame({
            "Time": weighted_mean.index,
            "values": weighted_mean.values
        })
        result = result.set_index("Time")

        if column == "ACTUAL_ET":
            simulated = result.reindex(self.observed_ET.index)["values"].to_numpy() * 1000
        else:
            simulated = result.reindex(self.observed_SWE.index)["values"].to_numpy()

        return simulated
    def evaluate(self, tmp_root: Path, feature_id: int) -> list[np.ndarray]:
        simulated_list = []
        for var_name in self.target_variables:
            if var_name == "streamflow":
                simulated_streamflow = self.evaluate_streamflow(tmp_root, feature_id)
                simulated_list.append(simulated_streamflow)
            elif var_name == "ET":
                simulated_et = self.evaluate_ET_SWE(tmp_root, "ACTUAL_ET")
                simulated_list.append(simulated_et)
            elif var_name == "SWE":
                simulated_swe = self.evaluate_ET_SWE(tmp_root, "SNEQV")
                simulated_list.append(simulated_swe)
        # shutil.rmtree(tmp_root, ignore_errors=True)
        return simulated_list


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
        norm: bool = False,
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
        self.norm = norm
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


    def simulation(self, vector: Sequence[float]) -> list[np.ndarray]:
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

    def evaluation(self) -> list[np.ndarray]:
        evaluation_list = []
        for var_name in self.model.target_variables:
            if var_name == "streamflow":
                evaluation_list.append(self.model.observed_streamflow.values.squeeze())
            elif var_name == "ET":
                evaluation_list.append(self.model.observed_ET.values.squeeze())
            elif var_name == "SWE":
                evaluation_list.append(self.model.observed_SWE.values.squeeze())
        return evaluation_list

    def objectivefunction(self, simulation: list[np.ndarray], evaluation: list[np.ndarray]) -> float:

        def calculate_metrics(eval_values: np.ndarray, sim_values: np.ndarray) -> dict:
            rmse = spotpy.objectivefunctions.rmse(eval_values, sim_values)
            kge = spotpy.objectivefunctions.kge(eval_values, sim_values)
            mae = np.mean(np.abs(eval_values - sim_values))
            nse_denominator = np.sum((eval_values - np.mean(eval_values)) ** 2)
            nse = np.nan
            if nse_denominator != 0:
                nse = 1 - (
                    np.sum((eval_values - sim_values) ** 2) / nse_denominator
                )
            correlation = np.nan
            if len(eval_values) > 1:
                correlation = np.corrcoef(eval_values, sim_values)[0, 1]
            return {
                "RMSE": rmse,
                "KGE": kge,
                "MAE": mae,
                "NSE": nse,
                "Correlation": correlation,
            }

        target_variable_items = list(self.model.target_variables.items())
        csv_row: dict[str, Any] = {"run_id": self.run_id}
        objective_list = []
        for sim, eval, (var_name, var_info) in zip(simulation, evaluation, target_variable_items):
            sim = np.asarray(sim)
            eval = np.asarray(eval)
            if len(sim) != len(eval):
                destroy_all_processes(f"Simulation and observation are not equal length")
                return float("nan")
            if np.sum(eval) == 0:
                # Since the value cannot be negative, this means all values here are 0.
                eval = eval + np.float64(1e-10)
            
            objective_metric_individual = self.obj_func(eval, sim)

            #if the norm is true, we do not have to multiply by weights
            if self.norm:
                objective_list.append(objective_metric_individual)
            else:
                objective_list.append(objective_metric_individual * var_info["weight"])

            csv_row[f"{self.objective_function_name}_{var_name}"] = objective_metric_individual
            if self.writer:
                metrics = calculate_metrics(eval, sim)
                variable_tag = f"TargetVariables/{var_name}"
                self.writer.add_scalar(
                    f"{variable_tag}/Objective_Function",
                    objective_metric_individual,
                    self.run_id,
                )
                self.writer.add_scalar(
                    f"{variable_tag}/Weighted_Objective_Function",
                    objective_metric_individual * var_info["weight"],
                    self.run_id,
                )
                self.writer.add_scalar(f"{variable_tag}/MAE", metrics["MAE"], self.run_id)
                self.writer.add_scalar(f"{variable_tag}/KGE", metrics["KGE"], self.run_id)
                self.writer.add_scalar(f"{variable_tag}/NSE", metrics["NSE"], self.run_id)
                self.writer.add_scalar(f"{variable_tag}/RMSE", metrics["RMSE"], self.run_id)
                self.writer.add_scalar(
                    f"{variable_tag}/Correlation",
                    metrics["Correlation"],
                    self.run_id,
                )

                if self.run_id % 10 == 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(eval, label="Observed", color="black", linewidth=1.5)
                    ax.plot(sim, label="Simulated", linestyle="--", alpha=0.8)
                    ax.legend()
                    ax.set_title(
                        f"{var_name} - Iteration {self.run_id} - Objective: {objective_metric_individual:.3f}"
                    )
                    ax.set_xlabel("Time step")
                    ax.set_ylabel(var_name)
                    ax.grid(True, alpha=0.3)
                    self.writer.add_figure(
                        f"TargetVariables/{var_name}/Comparison",
                        fig,
                        self.run_id,
                    )
                    plt.close(fig)

                    residuals = eval - sim
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.plot(residuals)
                    ax1.set_title(f"{var_name} Residuals Over Time")
                    ax1.set_xlabel("Time step")
                    ax1.set_ylabel("Residual")
                    ax1.grid(True, alpha=0.3)
                    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)

                    ax2.hist(residuals, bins=30, edgecolor="black")
                    ax2.set_title(f"{var_name} Residual Distribution")
                    ax2.set_xlabel("Residual")
                    ax2.set_ylabel("Frequency")
                    ax2.grid(True, alpha=0.3)

                    self.writer.add_figure(
                        f"TargetVariables/{var_name}/Residuals",
                        fig,
                        self.run_id,
                    )
                    plt.close(fig)

        #if norm is false, the objective metric is simply the sum of weighted KGE
        if not self.norm: 
            objective_metric = float(np.sum(objective_list))        

        if self.norm:
            total_sum = 0
            if self.objective_function_name == "KGE":
                for individual_kge in objective_list:
                    total_sum += (1 - individual_kge)**2  

                #it is multiplied with negative because math works out that way. this can also be seen as a way to preserve best_is_higher, as kge
                #increases, setting that to negative will also increase the objective metric
                objective_metric = -(np.sqrt(total_sum))
            else:
                for individual_RMSE in objective_list:
                    total_sum += individual_RMSE**2
                #same explanation as above on why there is no negative sign here
                objective_metric = np.sqrt(total_sum)

            if self.invert_objective:
                objective_metric = (-1) * objective_metric
        else:
            if self.objective_function_name == "KGE":
                objective_metric = objective_metric - 1
            if self.invert_objective:
                objective_metric = (-1) * objective_metric

        if self.norm:
            csv_row["total_weighted_objective (norm)"] = objective_metric
        else:
            csv_row["total_weighted_objective (weighted)"] = objective_metric

        if self.writer:
            self.writer.add_scalar(
                "Metrics/Objective_Function",
                objective_metric,
                self.run_id,
            )
            if self.run_id % 10 == 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(
                    [var_name for var_name, _ in target_variable_items],
                    objective_list,
                )
                ax.set_title(
                    f"Iteration {self.run_id} - Weighted Objective: {objective_metric:.3f}"
                )
                ax.set_xlabel("Target variable")
                ax.set_ylabel("Weighted objective")
                ax.grid(True, alpha=0.3)
                self.writer.add_figure(
                    "Metrics/Weighted_Objective_Contributions",
                    fig,
                    self.run_id,
                )
                plt.close(fig)
            self.writer.flush()

        csv_path = self.calibration_dir / "spotpy" / f"{self.objective_function_name}_history.csv"
        pd.DataFrame([csv_row]).to_csv(
            csv_path,
            mode="a",
            header=not csv_path.exists(),
            index=False,
        )
        self.run_id += 1
        return objective_metric


# === Function to Run SPOTPY Calibration with TensorBoard ===
def run_spotpy(
    gage_id: str,
    start_date: str,
    end_date: str,
    training_start_date: str,
    target_variables: dict,
    troute_output_path: Path,
    data_dir: Path,
    feature_id: int,
    rank: int,
    algorithm: str,
    objective_function: str,
    norm: bool,
    groups: Any,
    merge_catchment: bool,
    calibration_params: dict,
    tensorboard_logdir: Path,
    repetitions: int = 25,
    dds_trials: int = 5,
    n_pop: int = 10,
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
        target_variables,
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


    if algorithm == "SCE" or algorithm == "NSGAII":
        algorithm_maximizes = False

    else:
        algorithm_maximizes = True


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
        norm,
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
            sampler = spotpy.algorithms.dds(optimizer, dbname=db_name, dbformat="csv", random_state = 42)
        else:
            sampler = spotpy.algorithms.dds(
                optimizer, dbname=db_name, dbformat="csv", parallel="mpi", random_state = 42
            )

        if parameters_available:
            parameters = np.array(parameters)
            with spotpy_stdout_control(rank=rank, execution_mode=execution_mode):
                sampler.sample(repetitions, trials=int(dds_trials), x_initial=parameters)
        else:
            with spotpy_stdout_control(rank=rank, execution_mode=execution_mode):
                sampler.sample(repetitions, trials=int(dds_trials))

    if algorithm == "NSGAII":
        nsgaii_population = n_pop
        if execution_mode == "serial":
            sampler = spotpy.algorithms.NSGAII(optimizer, dbname=db_name, dbformat="csv")
            sampler.sample(generations=repetitions, n_obj=1, n_pop=nsgaii_population)
        else:
            sampler = spotpy.algorithms.NSGAII(
                optimizer, dbname=db_name, dbformat="csv", parallel="mpi"
            )
            with spotpy_stdout_control(rank=rank, execution_mode=execution_mode):
                sampler.sample(generations=repetitions, n_obj=1, n_pop=nsgaii_population)
                
    results = sampler.getdata()
    
    # Final results to TensorBoard
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=algorithm_maximizes)
    best_params_value = best_params[0]

    if algorithm_maximizes:
        best_params_index, _ = spotpy.analyser.get_maxlikeindex(results, verbose=False)
        best_params_index = best_params_index[0][0]
    else:
        best_params_index, _ = spotpy.analyser.get_minlikeindex(results, verbose=False)

    #redefine realization path to the main data directory
    realization_path = calibration_dir.parent / "config" / "realization.json"
    write_config(realization_path, best_params_value, param_to_model)

    if writer:
        # Log the parameter traces for all iterations from the SPOTPY CSV database.
        csv_path = Path(f"{db_name}.csv")
        log_parameters_from_spotpy_csv(writer, csv_path, params_names_list)
        writer.close()

    # Generate standard plots
    plot_results(results, optimizer, calibration_dir / "spotpy" / "plots", objective_function, algorithm_maximizes, best_is_higher)

    print(f"\nTensorBoard logs saved to: {run_log_dir}")
    print(f"Run 'tensorboard --logdir={tensorboard_logdir}' to view results\n\n")

    return best_params, best_params_index
