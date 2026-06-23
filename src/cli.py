from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from mpi4py import MPI
import typer
import traceback
from cal_utils import run_spotpy
from helper import *


def set_calibration_params(params: dict) -> None:
    global CALIBRATION_PARAMS
    CALIBRATION_PARAMS = params

app = typer.Typer(help="Run SPOTPY calibration for NextGen hydrologic model")

@app.command()
def calibration(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="YAML configuration file for streamflow calibration",
        ),
    ],
) -> int:
    #comm, rank, and size to handle multiprocess and race_condition
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    config_values = load_calibration_config(config)
    target_variables = None
    comm.Barrier()
    if rank == 0:
        #parsing target variables support automatic downloading of streamflow data if it doesn't
        #exist. So, only letting rank 0 download it
        target_variables = parse_target_variables(config_values["target_variables"], config_values)
    comm.barrier()
    target_variables = comm.bcast(target_variables, root=0)

    gage_id = str(config_values["gage_id"])
    start_date = str(config_values["start_date"])
    end_date = str(config_values["end_date"])
    training_start_date = str(config_values["training_start_date"])
    data_root = Path(config_values["data_root"])
    algorithm = str(config_values["algorithm"])
    objective_function = str(config_values["objective_function"])
    repetitions = int(config_values["repetitions"])
    dds_trials = int(config_values["dds_trials"])
    execution_mode = str(config_values["execution_mode"])
    merge_catchment = config_values["merge_catchment"]
    merge_area = float(config_values["merge_area"])
    n_pop = int(config_values["n_pop"])
    norm = config_values["norm"]

    data_root = data_root.expanduser()
    merge_catchment_bool = merge_catchment

    args = SimpleNamespace(
        gage_id=gage_id,
        start_date=start_date,
        end_date=end_date,
        training_start_date=training_start_date,
        data_root=data_root,
        algorithm=algorithm,
        objective_function=objective_function,
        repetitions=repetitions,
        dds_trials=dds_trials,
        execution_mode=execution_mode,
        merge_catchment_bool=merge_catchment_bool,
        merge_area=merge_area,
        target_variables=target_variables,
        norm = norm,
    )

    data_dir = data_root / f"gage-{gage_id}"
    troute_output_path = data_dir / "outputs" / "troute" / get_troute_output_name(data_dir / "config" / "realization.json") 
    tensorboard_logdir = data_dir / "calibration" / "tensorboard_logs"

    #need to define this to broadcast to other ranks
    groups = None
    clone_root = None

    if execution_mode == "serial" and size > 1:
        if rank == 0:
            destroy_all_processes("Running in serial mode but MPI detected multiple processes. For serial execution, run without mpirun.\n\n")

    if execution_mode == "parallel" and size == 1:
        if rank == 0:
            destroy_all_processes("Parallel mode requested, but only 1 MPI process detected.\n\n")
            
    comm.Barrier()
    try:
        if rank == 0:
            clone_root = create_directories(data_dir, objective_function)
            prepare_config(
                clone_root,
                execution_mode=execution_mode,
                target_variables=target_variables
            )
            if merge_catchment_bool:
                groups = merge_and_prepare_forcing(
                    data_dir=clone_root,
                    execution_mode=execution_mode,
                    merge_area=float(merge_area),
                )
            print_calibration_configuration(args=args, size=size)  

        comm.Barrier()
        clone_root = comm.bcast(clone_root, root=0)
        groups = comm.bcast(groups, root=0)
        comm.Barrier()
        feature_id = int(get_feature_id(clone_root))
        comm.Barrier()

        best_params, best_params_index = run_spotpy(
            gage_id,
            start_date,
            end_date,
            training_start_date,
            target_variables,
            troute_output_path,
            clone_root ,
            feature_id,
            rank,
            algorithm=algorithm,
            objective_function=objective_function,
            norm = norm,
            groups=groups,
            merge_catchment=merge_catchment_bool,
            calibration_params=CALIBRATION_PARAMS,
            tensorboard_logdir=tensorboard_logdir,
            repetitions=repetitions,
            dds_trials=dds_trials,
            n_pop=n_pop,
            execution_mode=execution_mode,
            number_of_cores=size if execution_mode == "parallel" else 1,
        )

        if rank == 0:
            output_file = data_dir / "calibration" / "spotpy" / "best_params.csv"

            with open(output_file, "w") as file:
                header = ",".join(
                    ["best_index"] + [name[3:] for name in best_params[0].dtype.names]
                )
                file.write(header + "\n")

                values = ",".join(
                    [str(best_params_index)] + [str(value) for value in best_params[0]]
                )
                file.write(values + "\n")

            print(f"\n{'=' * 60}")
            print("CALIBRATION COMPLETE")
            print(f"{'=' * 60}")
            print(f"Best parameters saved to: {output_file}")
            print("\nTo view TensorBoard results, run:")
            print(f"tensorboard --logdir={tensorboard_logdir}")
            print(f"{'=' * 60}\n")
            restore_data_dir(clone_root)

    except Exception as e:
        # restore_data_dir(clone_root)
        traceback.print_exc()
        destroy_all_processes(f"run_spotpy failed with error: {e} (Process rank {rank})\n\n")
    return 0


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
