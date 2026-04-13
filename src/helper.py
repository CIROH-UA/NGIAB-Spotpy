import json
import os
from datetime import datetime
from pathlib import Path

from numpy import partition
import pandas as pd
import yaml
from dataretrieval import nwis

from merge_catchment.geopackage import GeoPackage
from merge_catchment.interface import *


def get_troute_output_name(path):
    with Path(path).open("r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


def prepare_config_merged_simulation(data_dir, realization_path, troute_path, execution_mode):
    """This function prepares the realization_file and t-route file
    s.t. ngen and routing is done seperately"""

    gpkg_path = data_dir / "config" / f"{data_dir.name}_subset.gpkg"
    print("Preparing configuration files for merged geopackage simulation...")
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

    print("Merging geopackage and preparing forcing data...")
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

    # only create partitions if the execution mode is serial as the ngen simulation runs in parallel mode
    if execution_mode == "serial":
        # partitions for merged geopackage
        get_partitions(data_dir, merged_geopackage)

    return groups


def restore_data_dir(data_dir, merge_catchment):
    """Removes merged geopackage,forcing data prepared for merged geopackage simulation. And removes
    extra tmp yaml and json files created by staggering multiprocessing calibration. Also removes partiton files."""

    # instead of removing merged geopackage and forcing, create an archive directory and move those files there.
    if merge_catchment:
        merged_geopackage = data_dir / "config" / "merged.gpkg"
        forcing_path = data_dir / "forcings" / "forcings.nc"

        archive_dir = data_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        if merged_geopackage.exists():
            os.system(f"mv {merged_geopackage} {archive_dir}")
        if forcing_path.exists():
            os.system(f"mv {forcing_path} {archive_dir}")

        # move original forcing file back to forcings directory
        os.system(f"mv {data_dir}/forcings.nc {forcing_path}")
        print("Moved merged geopackage and forcing data used to archive...")

    # remove extra tmp yaml and json files created by staggering multiprocessing calibration
    tmp_files = list(data_dir.glob("config/tmp*"))
    if len(tmp_files) > 0:
        os.system(f"rm -rf {data_dir}/config/tmp*")

    # remove partiton files
    os.system(f"rm -rf {data_dir}/partitions_*.json")

    #remove calibration directory
    calibration_dir = data_dir / "Calibration"
    if calibration_dir.exists():
        os.system(f"rm -rf {calibration_dir}")


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

    try:
        dfo_usgs = nwis.get_record(sites=site, service="iv", start=adjusted_start, end=adjusted_end)
        dfo_usgs.index = pd.to_datetime(dfo_usgs.index)
        dfo_usgs["Time"] = dfo_usgs.index.floor("h")
        dfo_usgs["00060"] = pd.to_numeric(dfo_usgs["00060"], errors="coerce")
        dfo_usgs_hr = dfo_usgs.groupby("Time")["00060"].mean().reset_index()
        dfo_usgs_hr["values"] = dfo_usgs_hr["00060"] / 35.3147
        dfo_usgs_hr = dfo_usgs_hr[["Time", "values"]]
        # interpolate missing values
        dfo_usgs_hr["values"] = dfo_usgs_hr["values"].interpolate(method="linear")
    except:
        raise RuntimeError("There is no streamflow data for the provided gage!")
    if output_path:
        dfo_usgs_hr.to_pickle(Path(output_path))
    return dfo_usgs_hr
