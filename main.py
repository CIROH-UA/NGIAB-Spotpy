from mpi4py import MPI
import argparse
from pathlib import Path
from cal_utils import process_usgs_streamflow, run_spotpy
from helper import *

def main():
    parser = argparse.ArgumentParser(description="Run SPOTPY calibration for NextGen hydrologic model")
    
    # Required arguments
    parser.add_argument("--gage_id", type=str, required=True, help="USGS gage ID")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--training_start_date", type=str, required=True, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    
    # Optional arguments
    parser.add_argument("--algorithm", type=str, default="DDS", choices=["SCE", "DDS"], help="Optimization algorithm")
    parser.add_argument("--objective_function", type=str, default="KGE", choices=["KGE", "RMSE"], help="Objective function")
    parser.add_argument("--repetitions", type=int, default=100, help="Number of repetitions/iterations")
    parser.add_argument("--dds_trials", type=int, default=2, help="DDS trials (only used if algorithm=DDS)")
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
        feature_id = int(get_feature_id(data_dir))
        comm.Barrier()

        best_params = run_spotpy(
            args.gage_id,
            args.start_date,
            args.end_date,
            args.training_start_date,
            observed_flow_path,
            troute_output_path,
            data_dir,
            feature_id,
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