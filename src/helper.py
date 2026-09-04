from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Iterable, Sequence
import pandas as pd
import yaml
from dataretrieval import nwis
from mpi4py import MPI
from merge_catchment.geopackage import GeoPackage
from merge_catchment.interface import *
import time
import shutil
from tensorboardX import SummaryWriter
from collections import defaultdict
from plots import (
    plot_bestmodelrun,
    plot_parameter_correlation,
    plot_parameterInteraction,
    plot_parametertrace,
)
import sys
import subprocess
import numpy as np


def parameters_available_bool(
    realization_path: str | Path,
    params_name_dist_dict: dict[str, Any]
) -> tuple[bool, list[float] | dict[str, Any]]:
    """If parameters already exist in the realization file, use them
    as initial parameters. Only available for DDS algorithm."""
    with open(realization_path, "r") as f:
        config: dict[str, Any] = json.load(f)

    models_list = ["CFE", "NoahOWP"]
    parameters_available = False
    models_config = config["global"]["formulations"][0]["params"]["modules"]
    parameters_realization_file = {}
    parameter_values = []
    for model_type_name in models_list:
        for model in models_config:
            if model["params"]["model_type_name"] == model_type_name:
                if "model_params" in model["params"].keys():
                    parameters_available = True
                    parameters_realization_file.update(model["params"]["model_params"])
                else:
                    parameters_available = False
                    break

    if parameters_available:
        for param_name, param_dist in params_name_dist_dict.items():
            if param_name in parameters_realization_file:
                param_value = np.float64(parameters_realization_file[param_name])
                if param_value < param_dist.minbound or param_value > param_dist.maxbound:
                    print(
                        f"Parameter '{param_name}' value {param_value} is out of bounds ({param_dist.minbound}, {param_dist.maxbound})"
                    )
                    #clips the parameter value to be within the bounds of the distribution
                    param_value = np.clip(param_value, param_dist.minbound, param_dist.maxbound)

                parameter_values.append(param_value)

            else:
                #if it is not in the realization file, but is in calibration_params, then sample it
                #follows the same sampling method as spotpy, which is uniform sampling within the bounds of the distribution
                param_value = param_dist.minbound + np.random.rand() * (param_dist.maxbound - param_dist.minbound)
                parameter_values.append(param_value)

    return parameters_available, parameter_values


def log_parameters_from_spotpy_csv(
    writer: SummaryWriter, csv_path: Path, param_names: Sequence[str], step_offset: int = 0
) -> None:
    """
    Log SPOTPY parameters from the CSV database after the calibration finishes.
    """
    if writer is None:
        return

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[tensorboard] SPOTPY CSV not found: {csv_path}", file=sys.stderr)
        return

    par_cols = [f"par{name}" for name in param_names]
    df = pd.read_csv(csv_path, usecols=lambda c: c in par_cols)
    # Some runs/algorithms may omit columns; log only what exists.
    existing_par_cols = [c for c in par_cols if c in df.columns]
    if not existing_par_cols:
        print(
            f"[tensorboard] No parameter columns found in SPOTPY CSV: {csv_path}",
            file=sys.stderr,
        )
        return

    for i in range(len(df)):
        step = step_offset + i
        for name in param_names:
            col = f"par{name}"
            if col in df.columns:
                writer.add_scalar(f"Parameters/{name}", float(df.at[i, col]), step)

    writer.flush()


def plot_results(
    results: Any,
    optimizer: Any,
    output_dir: str | Path,
    objective_function: str,
    algorithm_maximizes: bool,
    best_is_higher: bool,
) -> None:
    plot_parametertrace(results=results, output_folder=output_dir)
    plot_parameterInteraction(results=results, output_folder=output_dir)
    plot_bestmodelrun(results=results, optimizer=optimizer, objective_function=objective_function, algorithm_maximizes=algorithm_maximizes, best_is_higher=best_is_higher,output_folder=output_dir)
    plot_parameter_correlation(results=results, output_folder=output_dir)

def _update_parameters(file_path: Path, param_updates: dict[str, Any], model_type_name: str) -> None:
    with open(file_path, "r") as f:
        realization: dict[str, Any] = json.load(f)
    models = realization["global"]["formulations"][0]["params"]["modules"]
    for model in models:
        if model["params"]["model_type_name"] == model_type_name:
            ## if model_params key doesn't exist, update model["params"]["model_params"] with param_updates
            if "model_params" not in model["params"]:
                model["params"]["model_params"] = param_updates
            else:
                #this help to only update the parameters that are in param_updates and not touch the other parameters
                for individual_param, new_value in param_updates.items():
                    model["params"]["model_params"][individual_param] = new_value
            break
    with open(file_path, "w") as f:
        json.dump(realization, f, indent=4)


def write_config(
    realization_path_name: str | Path,
    params: Sequence[float],
    param_models: dict[str, str],
) -> None:
    grouped: DefaultDict[str, dict[str, float]] = defaultdict(dict)
    for name, value in zip(param_models.keys(), params, strict=False):
        grouped[param_models[name]][name] = float(value)
    for model_type_name, values in grouped.items():
        _update_parameters(Path(realization_path_name), values, model_type_name)


def get_troute_output_name(path: str | Path) -> str:
    with Path(path).open("r") as file:
        realization: dict[str, Any] = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


def prepare_config(data_dir: Path, start_date: str, end_date: str, execution_mode: str) -> None:
    """This function prepares the realization_file and t-route file
    s.t. ngen and routing is done seperately. And also updates the start and end date in the realization file to match the user input. 
    This is efficient because it avoids the need to run ngen and routing for the entire period, which can be time-consuming."""

    realization_path = data_dir / "config" / "realization.json"
    troute_path = data_dir / "config" / "troute.yaml"
    gpkg_path = data_dir / "config" / f"{data_dir.name}_subset.gpkg"
    print("Preparing configuration for simulation...\n\n")
    # removing routing parameter from the realization file
    with realization_path.open("r") as file:
        realization: dict[str, Any] = json.load(file)
    if "routing" in realization.keys():
        realization.pop("routing", None)

    #also change the start date and end date in the realization file, should be in the format ""2015-06-05 00:00:00""
    realization["time"]["start_time"] = start_date + " 00:00:00"
    realization["time"]["end_time"] = end_date + " 00:00:00"

    with realization_path.open("w") as file:
        json.dump(realization, file, indent=4)

    # catchment routing should be done nexus routing is not an option for the merged geopackage
    # this doesn't preserve identation, but that shouldn't be an issue for the routing file
    with troute_path.open("r") as f:
        data = yaml.safe_load(f)
    data["compute_parameters"]["forcing_parameters"]["qlat_file_pattern_filter"] = "cat-*"
    data["compute_parameters"]["forcing_parameters"]["qlat_file_value_col"] = "Q_OUT"
    data["compute_parameters"]["restart_parameters"]["start_datetime"] = start_date + "_00:00"
    with troute_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=100)

    # remove existing partiton files if any
    partiton_files = list(data_dir.glob("partitions_*.json"))
    if len(partiton_files) > 0:
        os.system(f"rm -rf {data_dir}/partitions_*.json")

    if execution_mode == "serial":
        get_partitions(data_dir, gpkg_path)


def print_calibration_configuration(args, size):
    """Prints calibration configuration before the calibration"""

    print(f"\n{'=' * 60}")
    print("CALIBRATION CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Gage ID: {args.gage_id}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Training Start: {args.training_start_date}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Objective Function: {args.objective_function}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Execution Mode: {args.execution_mode}")
    print(f"MPI Processes: {size} (Master: 1, Workers: {size - 1})")
    print(f"{'=' * 60}\n")


def get_partitions(data_dir: Path, geopackage_path: Path) -> Path | None:
    size = min(30, os.cpu_count()-1)   # reserving one core for system processes
    partition_file = next(data_dir.glob(f"partitions_{size}.json"), None)
    if partition_file == None:
        hpc_command = (
            f"apptainer exec --cleanenv "
            f"--bind {data_dir}:/ngen/ngen/data "
            f"--pwd /ngen/ngen/data "
            f"ngiab_owp_openmpihpc.sif "
            f"python /dmod/utils/partitioning/round_robin.py "
            f"./config/{geopackage_path.name} {size} ."
        )
        subprocess.run(hpc_command, shell=True, capture_output=True, text=True, check=True)
    partition_file = next(data_dir.glob(f"partitions_{size}.json"), None)
    return partition_file  # return last element to get largest partitions


def merge_and_prepare_forcing(
    data_dir: Path, execution_mode: str, merge_area: float
) -> list[list[int]]:
    """Merges the geopackage, prepares forcing data, and creates partitions for the merged geopackage simulation."""

    #a better way to not flood the restore function with argument
    global merge_area_string
    merge_area_string = str(merge_area)

    original_gpkg = data_dir / "config" / f"{data_dir.name}_subset.gpkg"
    forcing_path = data_dir / "forcings" / "forcings.nc"
    merged_geopackage = data_dir / "config" / "merged.gpkg"

    #save the path stored in ~/.ngiab/preprocessor to a variable first
    with open(Path("~/.ngiab/preprocessor").expanduser(), "r") as f:
        preprocessor_path = f.read().strip()
    #change the path stored in ~/.ngiab/preprocessor to the current data directory
    os.system(f"echo {data_dir.parent} > ~/.ngiab/preprocessor")


    realization = data_dir / "config" / "realization.json"
    troute = data_dir / "config" / "troute.yaml"
    start, end = get_dates(realization)

    #back up these files because -r flag in preprocessing will alter the files
    backup(realization)
    backup(troute)

    #delete the forcing file
    forcing_path.unlink()

    #both merged file exists, so just copy from the archive directory to avoid preprocessing
    if (data_dir.parent.parent / "archive" / merge_area_string / "merged.gpkg").exists() and (data_dir.parent.parent / "archive" / merge_area_string / "forcings.nc").exists():
        print(f"Found merged geopackage and forcing for merge_area {merge_area_string};so, using these merged files for calibration\n\n")
        groups = group_catchments(original_gpkg, merge_area)
        shutil.copy2(data_dir.parent.parent / "archive" / merge_area_string / "merged.gpkg", merged_geopackage)
        shutil.copy2(data_dir.parent.parent / "archive" / merge_area_string / "forcings.nc", forcing_path)
        backup(original_gpkg)

        # rename merged geopackage to original in the folder
        os.system(f"mv {merged_geopackage} {original_gpkg}")
        
        cmd = (f"uvx -p 3.10 ngiab-prep -i {data_dir.name} -o {data_dir.name} --start {start} --end {end} -r")
        os.system(cmd)
        os.system(f"mv {original_gpkg} {merged_geopackage}")
        
    else:
        print("Merging geopackage and preparing forcing data...\n\n")
        # merge the geopackage
        try:
            hf = GeoPackage(original_gpkg)
            groups = group_catchments(original_gpkg, merge_area)
            hf.merge(groups)
            hf.save(merged_geopackage)
        except Exception as e:
            print(f"Merging failed with error: {e}\n\n")
            print("The merge_area value might be too small. Bump that value up and try calibrating again.")
            destroy_all_processes(f"Merging failed with error: {e}\n\nThe merge_area value might be too small. Bump that value up and try calibrating again.")
            
        backup(original_gpkg)
        # rename merged geopackage to original in the folder
        os.system(f"mv {merged_geopackage} {original_gpkg}")  
        cmd = (
            f"uvx -p 3.10 ngiab-prep -i {data_dir.name} -o {data_dir.name} --start {start} --end {end} -fr"
        )

        os.system(cmd)

        # rename original geopackage back to merged in the folder
        # with this, we will have both merged geopackage and the original one
        os.system(f"mv {original_gpkg} {merged_geopackage}")
    
    restore(original_gpkg)
    restore(realization)
    restore(troute)

    #renaming the path back tro default
    os.system(f"echo {preprocessor_path} > ~/.ngiab/preprocessor")

    # remove existing partiton files if any
    partiton_files = list(data_dir.glob("partitions_*.json"))
    if len(partiton_files) > 0:
        os.system(f"rm -rf {data_dir}/partitions_*.json")

    # only create partitions if the execution mode is serial as the ngen simulation runs in parallel mode
    if execution_mode == "serial":
        # partitions for merged geopackage
        get_partitions(data_dir, merged_geopackage)

    return groups


def create_directories(data_dir: Path) -> Path:
    """Create necessary directories for Calibration before hand to avoid race conditions when multiple processes are trying to create the same directory at the same time."""
    #just for sanity
    if (data_dir / "calibration" / "temp_runs").exists():
        shutil.rmtree(data_dir / "calibration" / "temp_runs")
    
    #delete this to avoid output clutter
    if (data_dir / "calibration" / "spotpy").exists():
        shutil.rmtree(data_dir / "calibration" / "spotpy")
    
    (data_dir / "calibration" / "spotpy" / "plots").mkdir(parents=True)

    #create clone root diretory inside "Temp_Runs" to keep the main directory clean and untouched 
    clone_root = data_dir / "calibration" / "temp_runs" / f"{data_dir.name}"
    clone_root.mkdir(parents=True, exist_ok=True)

    # --- config: full copy so each process can mutate its own files freely ---
    shutil.copytree(data_dir / "config", clone_root / "config")

    # --- metadata: full copy ---
    metadata_src = data_dir / "metadata"
    shutil.copytree(metadata_src, clone_root / "metadata")

    # --- forcings: hard-link the two large NetCDF files to avoid duplication ---
    forcings_dst = clone_root / "forcings"
    forcings_dst.mkdir(parents=True)
    
    shutil.copy2(data_dir / "forcings" / "forcings.nc", forcings_dst / "forcings.nc")


    # --- outputs: empty dirs ready for ngen / troute ---
    (clone_root / "outputs" / "ngen").mkdir(parents=True)
    (clone_root / "outputs" / "troute").mkdir(parents=True)

    return clone_root


def restore_data_dir(data_dir: Path) -> None:
    """Removes merged geopackage,forcing data prepared for merged geopackage simulation. And removes
    extra tmp yaml and json files created by staggering multiprocessing calibration. Also removes partiton files."""

    #restore .bak files 

    calibration_dir = data_dir.parent.parent
    merged_geopackage = data_dir / "config" / "merged.gpkg"

    if merged_geopackage.exists():
        forcing_path = data_dir / "forcings" / "forcings.nc"
        archive_dir = calibration_dir / "archive" / merge_area_string
        archive_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"mv {merged_geopackage} {archive_dir}")
        if forcing_path.exists():
            os.system(f"mv {forcing_path} {archive_dir}")

    # remove temporary cloned run directory (created under calibration/temp_runs)
    temp_runs_dir = data_dir.parent
    if temp_runs_dir.exists():
        shutil.rmtree(temp_runs_dir, ignore_errors=True)


def get_feature_id(data_dir: Path) -> str:
    gpkg = data_dir / "config" / f"{data_dir.name}_subset.gpkg"
    with sqlite3.connect(gpkg) as conn:
        cmd = (
            f"SELECT id FROM 'flowpath-attributes' WHERE gage='{data_dir.name.replace('gage-', '')}'"
        )
        results = conn.execute(cmd).fetchall()
        return results[0][0].split("-")[1]


# === Utility Function to Retrieve and Preprocess USGS Streamflow ===
def process_usgs_streamflow(
    site: str, start: str, end: str, output_path: str | Path | None = None
) -> pd.DataFrame:
    adjusted_start = pd.to_datetime(start) - pd.Timedelta(days=1)
    adjusted_end = pd.to_datetime(end) + pd.Timedelta(days=1)
    adjusted_start = adjusted_start.strftime("%Y-%m-%d")
    adjusted_end = adjusted_end.strftime("%Y-%m-%d")

    for attempt in range(1, 6):
        try:
            dfo_usgs = nwis.get_record(sites=site, service="iv", start=adjusted_start, end=adjusted_end)
            dfo_usgs.index = pd.to_datetime(dfo_usgs.index)
            dfo_usgs["Time"] = dfo_usgs.index.floor("h")
            dfo_usgs["00060"] = pd.to_numeric(dfo_usgs["00060"], errors="coerce")
            dfo_usgs_hr = dfo_usgs.groupby("Time")["00060"].mean().reset_index()
            dfo_usgs_hr["values"] = dfo_usgs_hr["00060"] / 35.3147
            dfo_usgs_hr = dfo_usgs_hr[["Time", "values"]]
            dfo_usgs_hr["values"] = dfo_usgs_hr["values"].interpolate(method="linear")
            break 
        except Exception as e:
            print(f"Attempt {attempt}/10: Failed to retrieve data — {e}. Retrying in 2 seconds...")
            time.sleep(2)
    else:
        destroy_all_processes("Failed to retrieve data after 10 attempts. No data may be available for this gage/period.")

    if output_path:
        dfo_usgs_hr.to_csv(Path(output_path), index=False)

    return dfo_usgs_hr


def destroy_all_processes(message: str) -> None:
    rank = MPI.COMM_WORLD.rank
    for i in range(3):
        print("\n\n***ERROR ERROR ERROR***\n")
        print(f"Reported by process number: {rank}")
        print(f"Message: {message} \n\n\n")
    MPI.COMM_WORLD.Abort(rank)


def str_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value)
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    destroy_all_processes("Boolean value expected.")
    return False


def validate_config_choice(
    config: dict[str, Any],
    field: str,
    uppercase: bool = True,
) -> None:
    
    if field == "algorithm":
        supported_values = ("SCE", "DDS")
    elif field == "objective_function":
        supported_values = ("KGE", "RMSE")
    else:
        supported_values = ("serial", "parallel")

    raw_value = config.get(field)
    value = str(raw_value).strip()
    value = value.upper() if uppercase else value.lower()

    if value not in supported_values:
        allowed = ", ".join(supported_values)
        destroy_all_processes(
            f"Invalid config value for '{field}': {raw_value!r}. "
            f"Supported values are: {allowed}."
        )
        return

    config[field] = value
    

def load_calibration_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser()
    if not config_path.exists():
        destroy_all_processes(f"Config file does not exist: {config_path}")

    with config_path.open("r") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        destroy_all_processes("Config file must contain a YAML mapping.")

    calibration_config = config.get("calibration", config)
    if not isinstance(calibration_config, dict):
        destroy_all_processes("The 'calibration' section must be a YAML mapping.")

    required_fields = [
        "gage_id",
        "start_date",
        "end_date",
        "training_start_date",
        "data_root",
    ]
    missing_fields = [
        field for field in required_fields if calibration_config.get(field) is None
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        destroy_all_processes(f"Missing required config field(s): {missing}")

    defaults = {
        "algorithm": "DDS",
        "objective_function": "KGE",
        "repetitions": 100,
        "dds_trials": 1,
        "execution_mode": "parallel",
        "merge_catchment": True,
        "merge_area": 200,
    }
    config_values = {**defaults, **calibration_config}

    validate_config_choice(config_values, "algorithm")
    validate_config_choice(config_values, "objective_function")
    validate_config_choice(config_values, "execution_mode", uppercase=False)
    config_values["merge_catchment"] = str_to_bool(config_values["merge_catchment"])
    return config_values
