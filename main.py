import sys
import os
from mpi4py import MPI
import json
import argparse
from datetime import datetime
from pathlib import Path

from cal_utils import process_usgs_streamflow, run_spotpy


def get_troute_output_name(path):
    with open(path, "r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


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
    
    # Synchronize all processes
    comm.Barrier()
    try:
        if rank == 0:
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
            repetitions=args.repetitions,
            dds_trials=args.dds_trials,
            execution_mode=args.execution_mode,
            number_of_cores = size if args.execution_mode == "parallel" else 1,
            tensorboard_logdir=tensorboard_logdir,
        )

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Only rank 0 saves results
        # if rank == 0:
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
        MPI.COMM_WORLD.Abort(0)
            
    except Exception as e:
        print(f"run_spotpy failed with error: {e} (Process rank {rank})")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()