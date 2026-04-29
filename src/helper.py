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


def parameters_available_bool(
    realization_path: str | Path,
) -> tuple[bool, list[float] | dict[str, Any]]:
    """If parameters already exist in the realization file, use them
    as initial parameters. Only available for DDS algorithm."""
    with open(realization_path, "r") as f:
        config: dict[str, Any] = json.load(f)

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
    observation_data: Any,
    output_dir: str | Path,
    objective_function: str,
    invert_objective: bool,
) -> None:
    plot_parametertrace(results=results, output_folder=output_dir)
    plot_parameterInteraction(results=results, output_folder=output_dir)
    plot_bestmodelrun(results=results, evaluation=observation_data, objective_function=objective_function, invert_objective=invert_objective, output_folder=output_dir)
    plot_parameter_correlation(results=results, output_folder=output_dir)


def _update_parameters(file_path: Path, param_updates: dict[str, Any], model_type_name: str) -> None:
    with open(file_path, "r") as f:
        realization: dict[str, Any] = json.load(f)
    models = realization["global"]["formulations"][0]["params"]["modules"]
    for model in models:
        if model["params"]["model_type_name"] == model_type_name:
            model["params"]["model_params"] = param_updates
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


def prepare_config(data_dir: Path, execution_mode: str) -> None:
    """This function prepares the realization_file and t-route file
    s.t. ngen and routing is done seperately"""

    realization_path = data_dir / "config" / "realization.json"
    troute_path = data_dir / "config" / "troute.yaml"
    gpkg_path = data_dir / "config" / f"{data_dir.name}_subset.gpkg"
    print("Preparing configuration files for merged geopackage simulation...\n\n")
    # removing routing parameter from the realization file
    with realization_path.open("r") as file:
        realization: dict[str, Any] = json.load(file)
    if "routing" in realization.keys():
        realization.pop("routing", None)
    with realization_path.open("w") as file:
        json.dump(realization, file, indent=4)

    # catchment routing should be done nexus routing is not an option for the merged geopackage
    # this doesn't preserve identation, but that shouldn't be an issue for the routing file
    with troute_path.open("r") as f:
        data = yaml.safe_load(f)
    data["compute_parameters"]["forcing_parameters"]["qlat_file_pattern_filter"] = "cat-*"
    data["compute_parameters"]["forcing_parameters"]["qlat_file_value_col"] = "Q_OUT"
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
    size = os.cpu_count() - 1  # reserving one core for system processes
    partition_file = next(data_dir.glob(f"partitions_{size}.json"), None)
    if partition_file == None:
        cmd_base = f"docker run --entrypoint python -w /ngen/ngen/data -v {data_dir}:/ngen/ngen/data awiciroh/ciroh-ngen-image /dmod/utils/partitioning/round_robin.py "
        cmd_opts = f"./config/{geopackage_path.name} {size} ."
        os.system(cmd_base + cmd_opts)
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
            MPI.COMM_WORLD.Abort(0)
            
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
    (data_dir / "calibration" / "spotpy" / "plots").mkdir(parents=True, exist_ok=True)

    #just for sanity
    if (data_dir / "calibration" / "temp_runs").exists():
        shutil.rmtree(data_dir / "calibration" / "temp_runs")

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
        print("Failed to retrieve data after 10 attempts. No data may be available for this gage/period.")
        MPI.COMM_WORLD.Abort(0)

    if output_path:
        dfo_usgs_hr.to_pickle(Path(output_path))

    return dfo_usgs_hr
