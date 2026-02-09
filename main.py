import sys
import os
from mpi4py import MPI
import json
import argparse
import yaml
from datetime import datetime
from pathlib import Path
from merge_catchment.geopackage import GeoPackage
from merge_catchment.interface import *
from cal_utils import process_usgs_streamflow, run_spotpy


def get_troute_output_name(path):
    with open(path, "r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


def prepare_config_merged_simulation(realization_path, troute_path):
    '''This function prepares the realization_file and t-route file
    for merged catchment simulation.'''

    print("Preparing configuration files for merged geopackage simulation...")
    #removing routing parameter from the realization file
    with open(realization_path, "r") as file:
        realization = json.load(file)
    if "routing" in realization.keys():
        realization.pop("routing", None)
    with open(realization_path, "w") as file:
        json.dump(realization, file, indent=4)
    

    #catchment routing should be done nexus routing is not an option for the merged geopackage
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
    print(f"Feature ID: {args.feature_id}")
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

def restore_data_dir(data_dir):
    '''Removes merged geopackage,forcing data prepared for merged geopackage simulation. And removes 
    extra tmp yaml and json files created by staggering multiprocessing calibration. Also removes partiton files.'''

    print("Removing merged geopackage and forcing data used for merged geopackage simulation...")
    folder = Path(data_dir)
    merged_geopackage = folder / "config" / "merged.gpkg"
    forcing_path = folder / "forcings"/ "forcings.nc"

    if merged_geopackage.exists():
        merged_geopackage.unlink()
    if forcing_path.exists():
        forcing_path.unlink()
    
    #move original forcing file back to forcings directory
    os.system(f"mv {folder}/forcings.nc {forcing_path}")

    #remove extra tmp yaml and json files created by staggering multiprocessing calibration
    tmp_files = list(folder.glob("config/tmp*"))
    if len(tmp_files) > 0:
        os.system(f"rm -rf {folder}/config/tmp*")

    #remove partiton files
    os.system(f"rm -rf {folder}/partitions_*.json")

def main():
    parser = argparse.ArgumentParser(description="Run SPOTPY calibration for NextGen hydrologic model")
    
    # Required arguments
    parser.add_argument("--gage_id", type=str, required=True, help="USGS gage ID")
    parser.add_argument("--feature_id", type=int, required=True, help="Feature ID for routing")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--training_start_date", type=str, required=True, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    
    # Optional arguments
    parser.add_argument("--algorithm", type=str, default="SCE", choices=["SCE", "DDS"], help="Optimization algorithm")
    parser.add_argument("--objective_function", type=str, default="KGE", choices=["KGE", "RMSE"], help="Objective function")
    parser.add_argument("--repetitions", type=int, default=10, help="Number of repetitions/iterations")
    parser.add_argument("--dds_trials", type=int, default=5, help="DDS trials (only used if algorithm=DDS)")
    parser.add_argument("--execution_mode", type=str, default="parallel", choices=["serial", "parallel"], help="Serial or parallel execution")
    
    args = parser.parse_args()
 
    # Setup paths
    realization_path = f"{args.data_root}/gage-{args.gage_id}/config/realization.json"
    troute_path = f"{args.data_root}/gage-{args.gage_id}/config/troute.yaml"
    observed_flow_path = f"{args.data_root}/{args.gage_id}_observed_flow_{args.start_date}_{args.end_date}.pkl"
    troute_output_path = (
        f"{args.data_root}/gage-{args.gage_id}/outputs/troute/{get_troute_output_name(realization_path)}"
    )
    data_dir = f"{args.data_root}/gage-{args.gage_id}"
    tensorboard_logdir = f"{data_dir}/tensorboard_logs"

    
    # Check execution mode
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    groups = None
    
    if args.execution_mode == "serial" and size > 1:
        if rank == 0:
            raise ValueError("Warning: Running in serial mode but MPI detected multiple processes. For serial execution, run without mpirun.")
    
    if args.execution_mode == "parallel" and size == 1:
        if rank == 0:
            print(f"Warning: Parallel mode requested, but only 1 MPI process detected.")
    
    # Optional: Retrieve and save observed flow
    if rank == 0:
        if not Path(observed_flow_path).exists():
            print(f"Retrieving observed streamflow for gage {args.gage_id}...")
            process_usgs_streamflow(args.gage_id, args.start_date, args.end_date, output_path=observed_flow_path)
        else:
            print(f"Using existing observed flow data: {observed_flow_path}")
    
    comm.Barrier()  # Ensure all processes wait until here, because realization file gets changed here and there might be some conflicts
    try:
        if rank == 0:
            print_calibration_configuration(args=args, size=size)
            prepare_config_merged_simulation(realization_path=realization_path, troute_path=troute_path)
            groups = merge_and_prepare_forcing(data_dir=data_dir, execution_mode=args.execution_mode)

        # Synchronize all processes
        comm.Barrier()
        groups = comm.bcast(groups, root=0)
        comm.Barrier()

        best_params = run_spotpy(
            args.gage_id,
            args.start_date,
            args.end_date,
            args.training_start_date,
            observed_flow_path,
            troute_output_path,
            data_dir,
            args.feature_id,
            algorithm=args.algorithm,
            objective_function=args.objective_function,
            groups=groups,
            repetitions=args.repetitions,
            dds_trials=args.dds_trials,
            execution_mode=args.execution_mode,
            number_of_cores = size if args.execution_mode == "parallel" else 1,
            tensorboard_logdir=tensorboard_logdir,
        )       
        # Only rank 0 saves results
        if rank == 0:
        # Save the best parameters to a file
            output_file = f"{data_dir}/spotpy/best_params.csv"
            with open(output_file, "w") as file:
                header = ",".join([name[3:] for name in best_params[0].dtype.names])
                file.write(header + "\n")
                values = ",".join([str(value) for value in best_params[0]])
                file.write(values + "\n")
            
            print(f"\n{'='*60}")
            print(f"CALIBRATION COMPLETE")
            print(f"{'='*60}") 
            print(f"Best parameters saved to: {output_file}")
            print(f"\nTo view TensorBoard results, run:")
            print(f"tensorboard --logdir={tensorboard_logdir}")
            print(f"{'='*60}\n")
            restore_data_dir(data_dir=data_dir)
            #stops all other ongoing processes
            MPI.COMM_WORLD.Abort(0)
            
    except Exception as e:
        print(f"run_spotpy failed with error: {e} (Process rank {rank})")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()