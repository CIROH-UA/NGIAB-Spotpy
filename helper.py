import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from merge_catchment.geopackage import GeoPackage
from merge_catchment.interface import *
from dataretrieval import nwis
import pandas as pd

def get_troute_output_name(path):
    with open(path, "r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


def prepare_config_merged_simulation(realization_path, troute_path):
    '''This function prepares the realization_file and t-route file
    s.t. ngen and routing is done seperately'''

    print("Preparing configuration files for merged geopackage simulation...")
    #removing routing parameter from the realization file
    with open(realization_path, "r") as file:
        realization = json.load(file)
    if "routing" in realization.keys():
        realization.pop("routing", None)
    with open(realization_path, "w") as file:
        json.dump(realization, file, indent=4)
    

    #catchment routing should be done nexus routing is not an option for the merged geopackage
    #this doesn't preserve identation, but that shouldn't be an issue for routing
    with open(troute_path, 'r') as f:
        data = yaml.safe_load(f)
    data['compute_parameters']['forcing_parameters']['qlat_file_pattern_filter'] = "cat-*"
    data['compute_parameters']['forcing_parameters']['qlat_file_value_col'] = "Q_OUT"
    with open(troute_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=100)

def print_calibration_configuration(args, size):
    '''Prints calibration configuration before the calibration'''

    print(f"\n{'='*60}")
    print(f"CALIBRATION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Gage ID: {args.gage_id}")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Training Start: {args.training_start_date}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Objective Function: {args.objective_function}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Execution Mode: {args.execution_mode}")
    print(f"MPI Processes: {size} (Master: 1, Workers: {size-1})")
    print(f"{'='*60}\n")

def get_partitions(data_dir, geopackage_path) -> Path:
    data_dir = Path(data_dir)
    size = os.cpu_count() - 1  # reserving one core for system processes
    partition_file = next(data_dir.glob(f"partitions_{size}.json"), None)
    if partition_file == None:
        cmd_base = f"docker run --entrypoint python -w /ngen/ngen/data -v {data_dir}:/ngen/ngen/data awiciroh/ciroh-ngen-image /dmod/utils/partitioning/round_robin.py "
        cmd_opts = f"./config/{geopackage_path.name} {size} ."
        os.system(cmd_base + cmd_opts)
    partition_file = next(data_dir.glob(f"partitions_{size}.json"), None)
    return partition_file  # return last element to get largest partitions


def merge_and_prepare_forcing(data_dir, execution_mode):
    '''Merges the geopackage, prepares forcing data, and creates partitions for the merged geopackage simulation.'''

    #prepare partitions before merging
    folder = Path(data_dir)
    original_gpkg = folder / "config" / f"{folder.name}_subset.gpkg"
    forcing_path = folder / "forcings"/ "forcings.nc"
    merged_geopackage = folder / "config" / "merged.gpkg"

    
    #remove existing partiton files if any
    partiton_files = list(folder.glob("partitions_*.json"))
    if len(partiton_files) > 0:
        os.system(f"rm -rf {folder}/partitions_*.json")

    print("Merging geopackage and preparing forcing data...")
    #merge the geopackage
    hf = GeoPackage(original_gpkg)
    groups = group_catchments(original_gpkg)
    hf.merge(groups)
    hf.save(merged_geopackage)

    realization = folder / "config" / "realization.json"
    troute = folder / "config" / "troute.yaml"
    start, end = get_dates(realization)
    backup(original_gpkg)

    #move forcing file of the forcing directory just outside of the forcings folder so that new data prepared will be for the merged geopackage
    os.system(f"mv {forcing_path} {folder}")

    #rename merged geopackage to original in the folder
    os.system(f"mv {merged_geopackage} {original_gpkg}")

    backup(realization)
    backup(troute)
    cmd = f"uvx -p 3.10 ngiab-prep -i {folder.name} -o {folder.name} --start {start} --end {end} -fr --source aorc"

    os.system(cmd)

    #rename original geopackage back to merged in the folder
    #with this, we will have both merged geopackage and the original one
    os.system(f"mv {original_gpkg} {merged_geopackage}")
    restore(original_gpkg)
    restore(realization)
    restore(troute)
    
    #only create partitions if the execution mode is serial as the ngen simulation runs in parallel mode
    if execution_mode == "serial":
        #partitions for merged geopackage
        get_partitions(data_dir, merged_geopackage)

    return groups

def restore_data_dir(data_dir, merge_catchment):
    '''Removes merged geopackage,forcing data prepared for merged geopackage simulation. And removes 
    extra tmp yaml and json files created by staggering multiprocessing calibration. Also removes partiton files.'''

    folder = Path(data_dir)
    #instead of removing merged geopackage and forcing, create an archive directory and move those files there. 
    if merge_catchment:
        print("Moving merged geopackage and forcing data used to archive...")
        merged_geopackage = folder / "config" / "merged.gpkg"
        forcing_path = folder / "forcings"/ "forcings.nc"

        archive_dir = folder / "archive"
        archive_dir.mkdir(exist_ok=True)

        if merged_geopackage.exists():
            os.system(f"mv {merged_geopackage} {archive_dir}")
        if forcing_path.exists():
            os.system(f"mv {forcing_path} {archive_dir}")
        
        #move original forcing file back to forcings directory
        os.system(f"mv {folder}/forcings.nc {forcing_path}")

    #remove extra tmp yaml and json files created by staggering multiprocessing calibration
    tmp_files = list(folder.glob("config/tmp*"))
    if len(tmp_files) > 0:
        os.system(f"rm -rf {folder}/config/tmp*")

    #remove partiton files
    os.system(f"rm -rf {folder}/partitions_*.json")

def get_feature_id(data_dir):
    folder = Path(data_dir)
    gpkg = folder / "config" / f"{folder.name}_subset.gpkg"
    with sqlite3.connect(gpkg) as conn:
        cmd = f"SELECT id FROM 'flowpath-attributes' WHERE gage='{folder.name.replace('gage-', '')}'"
        results = conn.execute(cmd).fetchall()
        return results[0][0].split("-")[1]
    
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
    #interpolate missing values
    dfo_usgs_hr["values"] = dfo_usgs_hr["values"].interpolate(method='linear')
    if output_path:
        dfo_usgs_hr.to_pickle(output_path)
    return dfo_usgs_hr