from enum import Enum
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

class Algorithm(str, Enum):
    SCE = "SCE"
    DDS = "DDS"


class ObjectiveFunction(str, Enum):
    KGE = "KGE"
    RMSE = "RMSE"


class ExecutionMode(str, Enum):
    SERIAL = "serial"
    PARALLEL = "parallel"


app = typer.Typer(help="Run SPOTPY calibration for NextGen hydrologic model")


def str_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value)
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise typer.BadParameter("Boolean value expected.")


@app.command()
def calibration(
    gage_id: Annotated[
        str, typer.Option("--gage_id", help="USGS gage ID")
    ],
    start_date: Annotated[
        str, typer.Option("--start_date", help="Start date (YYYY-MM-DD)")
    ],
    end_date: Annotated[
        str, typer.Option("--end_date", help="End date (YYYY-MM-DD)")
    ],
    training_start_date: Annotated[
        str,
        typer.Option("--training_start_date", help="Training start date (YYYY-MM-DD)"),
    ],
    data_root: Annotated[
        Path, typer.Option("--data_root", help="Root directory for data")
    ],
    algorithm: Annotated[
        Algorithm, typer.Option("--algorithm", help="Optimization algorithm")
    ] = Algorithm.DDS,
    objective_function: Annotated[
        ObjectiveFunction, typer.Option("--objective_function", help="Objective function")
    ] = ObjectiveFunction.KGE,
    repetitions: Annotated[
        int, typer.Option("--repetitions", help="Number of repetitions/iterations")
    ] = 100,
    dds_trials: Annotated[
        int, typer.Option("--dds_trials", help="DDS trials (only used if algorithm=DDS)")
    ] = 1,
    execution_mode: Annotated[
        ExecutionMode, typer.Option("--execution_mode", help="Serial or parallel execution")
    ] = ExecutionMode.PARALLEL,
    merge_catchment: Annotated[
        str,
        typer.Option("--merge_catchment", help="Whether to merge catchments for calibration"),
    ] = "True",
    merge_area: Annotated[
        float,
        typer.Option(
            "--merge_area",
            help="The catchment area to merge the divides in square miles",
        ),
    ] = 330,
) -> int:
    data_root = data_root.expanduser()
    merge_catchment_bool = str_to_bool(merge_catchment) 

    args = SimpleNamespace(
        gage_id=gage_id,
        start_date=start_date,
        end_date=end_date,
        training_start_date=training_start_date,
        data_root=data_root,
        algorithm=algorithm.value,
        objective_function=objective_function.value,
        repetitions=repetitions,
        dds_trials=dds_trials,
        execution_mode=execution_mode.value,
        merge_catchment_bool=merge_catchment_bool,
        merge_area=merge_area,
    )

    data_dir = data_root / f"gage-{gage_id}"
    observed_flow_path = data_root / f"{gage_id}_observed_flow_{start_date}_{end_date}.pkl"
    troute_output_path = data_dir / "outputs" / "troute" / get_troute_output_name(data_dir / "config" / "realization.json") 
    tensorboard_logdir = data_dir / "calibration" / "tensorboard_logs"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #need to define this to broadcast to other ranks
    groups = None
    clone_root = None

    if execution_mode.value == "serial" and size > 1:
        if rank == 0:
            raise ValueError(
                "Warning: Running in serial mode but MPI detected multiple processes. For serial execution, run without mpirun.\n\n"
            )

    if execution_mode.value == "parallel" and size == 1:
        if rank == 0:
            raise ValueError("Parallel mode requested, but only 1 MPI process detected.\n\n")

    if rank == 0:
        if not observed_flow_path.exists():
            print(f"\n\nRetrieving observed streamflow for gage {gage_id}...\n\n")
            process_usgs_streamflow(
                gage_id,
                start_date,
                end_date,
                output_path=observed_flow_path,
            )
        else:
            print(f"\n\nUsing existing observed flow data: {observed_flow_path}\n\n")

    comm.Barrier()
    try:
        if rank == 0:
            clone_root = create_directories(data_dir)
            prepare_config(
                clone_root,
                execution_mode=execution_mode.value
            )
            if merge_catchment_bool:
                groups = merge_and_prepare_forcing(
                    data_dir=clone_root,
                    execution_mode=execution_mode.value,
                    merge_area=float(merge_area),
                )
            print_calibration_configuration(args=args, size=size)  

        comm.Barrier()
        clone_root = comm.bcast(clone_root, root=0)
        groups = comm.bcast(groups, root=0)
        comm.Barrier()
        feature_id = int(get_feature_id(clone_root))
        comm.Barrier()

        best_params = run_spotpy(
            gage_id,
            start_date,
            end_date,
            training_start_date,
            observed_flow_path,
            troute_output_path,
            clone_root ,
            feature_id,
            rank,
            algorithm=algorithm.value,
            objective_function=objective_function.value,
            groups=groups,
            merge_catchment=merge_catchment_bool,
            calibration_params=CALIBRATION_PARAMS,
            tensorboard_logdir=tensorboard_logdir,
            repetitions=repetitions,
            dds_trials=dds_trials,
            execution_mode=execution_mode.value,
            number_of_cores=size if execution_mode.value == "parallel" else 1,
        )

        if rank == 0:
            output_file = data_dir / "calibration" / "spotpy" / "best_params.csv"
            with open(output_file, "w") as file:
                header = ",".join([name[3:] for name in best_params[0].dtype.names])
                file.write(header + "\n")
                values = ",".join([str(value) for value in best_params[0]])
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
        print(f"run_spotpy failed with error: {e} (Process rank {rank})\n\n")
        # restore_data_dir(clone_root)
        traceback.print_exc()

    return 0


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
