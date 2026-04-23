import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
from numpy import partition
import pandas as pd
import yaml
from dataretrieval import nwis
from mpi4py import MPI

from merge_catchment.geopackage import GeoPackage
from merge_catchment.interface import *
import time
import shutil
from contextlib import contextmanager

def get_troute_output_name(path):
    with Path(path).open("r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


def prepare_config_merged_simulation(data_dir, execution_mode):
    """This function prepares the realization_file and t-route file
    s.t. ngen and routing is done seperately"""

    realization_path = data_dir / "config" / "realization.json"
    troute_path = data_dir / "config" / "troute.yaml"
    gpkg_path = data_dir / "config" / f"{data_dir.name}_subset.gpkg"
    print("Preparing configuration files for merged geopackage simulation...\n\n")
    # removing routing parameter from the realization file
    with realization_path.open("r") as file:
        realization = json.load(file)
    if "routing" in realization.keys():
        realization.pop("routing", None)
    with realization_path.open("w") as file:
        json.dump(realization, file, indent=4)

    # catchment routing should be done nexus routing is not an option for the merged geopackage
    # this doesn't preserve identation, but that shouldn't be an issue for routing
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


def get_partitions(data_dir, geopackage_path):
    size = os.cpu_count() - 1  # reserving one core for system processes
    partition_file = next(data_dir.glob(f"partitions_{size}.json"), None)
    if partition_file == None:
        cmd_base = f"docker run --entrypoint python -w /ngen/ngen/data -v {data_dir}:/ngen/ngen/data awiciroh/ciroh-ngen-image /dmod/utils/partitioning/round_robin.py "
        cmd_opts = f"./config/{geopackage_path.name} {size} ."
        os.system(cmd_base + cmd_opts)
    partition_file = next(data_dir.glob(f"partitions_{size}.json"), None)
    return partition_file  # return last element to get largest partitions


def merge_and_prepare_forcing(data_dir, execution_mode, merge_area):
    """Merges the geopackage, prepares forcing data, and creates partitions for the merged geopackage simulation."""

    # prepare partitions before merging
    original_gpkg = data_dir / "config" / f"{data_dir.name}_subset.gpkg"
    forcing_path = data_dir / "forcings" / "forcings.nc"
    merged_geopackage = data_dir / "config" / "merged.gpkg"

    # remove existing partiton files if any
    partiton_files = list(data_dir.glob("partitions_*.json"))
    if len(partiton_files) > 0:
        os.system(f"rm -rf {data_dir}/partitions_*.json")

    print("Merging geopackage and preparing forcing data...\n\n")
    # merge the geopackage
    hf = GeoPackage(original_gpkg)
    groups = group_catchments(original_gpkg, merge_area)
    hf.merge(groups)
    hf.save(merged_geopackage)

    realization = data_dir / "config" / "realization.json"
    troute = data_dir / "config" / "troute.yaml"
    start, end = get_dates(realization)
    backup(original_gpkg)

    # move forcing file of the forcing directory just outside of the forcings folder so that new data prepared will be for the merged geopackage
    os.system(f"mv {forcing_path} {data_dir}")

    # rename merged geopackage to original in the folder
    os.system(f"mv {merged_geopackage} {original_gpkg}")

    backup(realization)
    backup(troute)

    #cat ~/.ngiab/preprocessor gives the path to the folder where preprocesor downloads the data
    #so that path should be changed to the current data directory to prepare forcing data for the merged geopackage simulation
    #but after the merged data is downlaoded, should be changed back to original path
    
    #save the path stored in ~/.ngiab/preprocessor to a variable first
    with open(Path("~/.ngiab/preprocessor").expanduser(), "r") as f:
        preprocessor_path = f.read().strip()
    #change the path stored in ~/.ngiab/preprocessor to the current data directory
    os.system(f"echo {data_dir.parent} > ~/.ngiab/preprocessor")
    
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

    os.system(f"echo {preprocessor_path} > ~/.ngiab/preprocessor")

    # only create partitions if the execution mode is serial as the ngen simulation runs in parallel mode
    if execution_mode == "serial":
        # partitions for merged geopackage
        get_partitions(data_dir, merged_geopackage)

    return groups


def create_directories(data_dir):
    """Create necessary directories for Calibration before hand to avoid race conditions when multiple processes are trying to create the same directory at the same time."""
    (data_dir / "calibration" / "spotpy" / "plots").mkdir(parents=True, exist_ok=True)
    # (data_dir / "calibration" / "Temp_Runs").mkdir(parents=True, exist_ok=True)

    #just for sanity
    if (data_dir / "calibration" / "temp_Runs").exists():
        shutil.rmtree(data_dir / "calibration" / "temp_Runs")

    #create clone root diretory inside "Temp_Runs" to keep the main directory clean and untouched 
    clone_root = data_dir / "calibration" / "temp_Runs" / f"{data_dir.name}"
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

    #hardlink raw gridded forcing data as well to avoid duplication
    os.link(data_dir / "forcings" / "raw_gridded_data.nc", forcings_dst / "raw_gridded_data.nc")

    # --- outputs: empty dirs ready for ngen / troute ---
    (clone_root / "outputs" / "ngen").mkdir(parents=True)
    (clone_root / "outputs" / "troute").mkdir(parents=True)

    return clone_root

def restore_data_dir(data_dir):
    """Removes merged geopackage,forcing data prepared for merged geopackage simulation. And removes
    extra tmp yaml and json files created by staggering multiprocessing calibration. Also removes partiton files."""

    #restore .bak files 

    # bak_files = list((data_dir / "config").glob("*.bak"))
    # for bak_file in bak_files:
    #     restore(bak_file)
    calibration_dir = data_dir.parent.parent
    merged_geopackage = data_dir / "config" / "merged.gpkg"

    if merged_geopackage.exists():
        forcing_path = data_dir / "forcings" / "forcings.nc"
        archive_dir = calibration_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        os.system(f"mv {merged_geopackage} {archive_dir}")

        if forcing_path.exists():
            os.system(f"mv {forcing_path} {archive_dir}")

        # # move original forcing file back to forcings directory
        # os.system(f"mv {data_dir}/forcings.nc {forcing_path}")
        # print("Moved merged geopackage and forcing data used to archive\n\n")

    # # remove partiton files
    # os.system(f"rm -rf {data_dir}/partitions_*.json")

    # remove temporary cloned run directory (created under calibration/Temp_Runs)
    temp_runs_dir = data_dir.parent
    if temp_runs_dir.exists():
        shutil.rmtree(temp_runs_dir, ignore_errors=True)


def get_feature_id(data_dir):
    folder = Path(data_dir)
    gpkg = folder / "config" / f"{folder.name}_subset.gpkg"
    with sqlite3.connect(gpkg) as conn:
        cmd = (
            f"SELECT id FROM 'flowpath-attributes' WHERE gage='{folder.name.replace('gage-', '')}'"
        )
        results = conn.execute(cmd).fetchall()
        return results[0][0].split("-")[1]


# === Utility Function to Retrieve and Preprocess USGS Streamflow ===
def process_usgs_streamflow(site, start, end, output_path=None):
    start = pd.to_datetime(start) - pd.Timedelta(days=1)
    end = pd.to_datetime(end) + pd.Timedelta(days=1)
    adjusted_start = start.strftime("%Y-%m-%d")
    adjusted_end = end.strftime("%Y-%m-%d")

    for attempt in range(1, 11):
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

    # Check that returned data covers the full requested time period
    # breakpoint()
    # if dfo_usgs_hr["Time"].min().tz_localize(None) > start or dfo_usgs_hr["Time"].max().tz_localize(None) < end:
    #     print("Data from NWIS does not cover the full time period. Check gage data availability.")
    #     MPI.COMM_WORLD.Abort(0)

    if output_path:
        dfo_usgs_hr.to_pickle(Path(output_path))

    return dfo_usgs_hr
