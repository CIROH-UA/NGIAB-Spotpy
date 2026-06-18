from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any
from mpi4py import MPI
import typer
import traceback
import yaml
from cal_utils import run_spotpy
from helper import *


def set_calibration_params(params: dict) -> None:
    global CALIBRATION_PARAMS
    CALIBRATION_PARAMS = params

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


def load_calibration_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser()
    if not config_path.exists():
        raise typer.BadParameter(f"Config file does not exist: {config_path}")

    with config_path.open("r") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise typer.BadParameter("Config file must contain a YAML mapping.")

    calibration_config = config.get("calibration", config)
    if not isinstance(calibration_config, dict):
        raise typer.BadParameter("The 'calibration' section must be a YAML mapping.")

    required_fields = [
        "gage_id",
        "start_date",
        "end_date",
        "training_start_date",
        "data_root",
        "target_variables",
    ]
    missing_fields = [
        field for field in required_fields if calibration_config.get(field) is None
    ]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise typer.BadParameter(f"Missing required config field(s): {missing}")

    defaults = {
        "algorithm": "DDS",
        "objective_function": "KGE",
        "repetitions": 100,
        "dds_trials": 1,
        "execution_mode": "parallel",
        "merge_catchment": True,
        "merge_area": 200,
        "n_pop" : 10,
    }
    return {**defaults, **calibration_config}


def parse_target_variables(target_variables: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(target_variables, dict) or not target_variables:
        raise typer.BadParameter(
            "'target_variables' must be a non-empty mapping of variable names to target settings."
        )

    parsed_target_variables = {}
    total_weight = 0.0
    for variable, target_config in target_variables.items():
        variable_name = str(variable).strip()
        if not variable_name:
            raise typer.BadParameter("'target_variables' contains an empty variable name.")

        if not isinstance(target_config, dict):
            raise typer.BadParameter(
                f"'target_variables.{variable_name}' must be a mapping with output_path and weight."
            )

        output_path = target_config.get("observed_data_path")
        if output_path is None:
            raise typer.BadParameter(
                f"'target_variables.{variable_name}.output_path' must define an observed data path."
            )

        weight = target_config.get("weight", target_config.get("weights"))
        if weight is None:
            raise typer.BadParameter(
                f"'target_variables.{variable_name}' must define a weight."
            )

        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise typer.BadParameter(
                f"'target_variables.{variable_name}.weight' must be numeric."
            ) from exc

        if weight < 0:
            raise typer.BadParameter(
                f"'target_variables.{variable_name}.weight' cannot be negative."
            )

        total_weight += weight
        parsed_target_variables[variable_name] = {
            "observed_data_path": Path(str(output_path)).expanduser(),
            "weight": weight,
        }

    if not abs(total_weight - 1.0) <= 1e-9:
        raise typer.BadParameter(
            f"The sum of target variable weights must equal 1.0; got {total_weight}."
        )

    return parsed_target_variables


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
    config_values = load_calibration_config(config)

    target_variables = parse_target_variables(config_values["target_variables"])
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

    data_root = data_root.expanduser()
    merge_catchment_bool = str_to_bool(merge_catchment) 

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
    )

    data_dir = data_root / f"gage-{gage_id}"
    troute_output_path = data_dir / "outputs" / "troute" / get_troute_output_name(data_dir / "config" / "realization.json") 
    tensorboard_logdir = data_dir / "calibration" / "tensorboard_logs"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #need to define this to broadcast to other ranks
    groups = None
    clone_root = None

    if execution_mode == "serial" and size > 1:
        if rank == 0:
            raise ValueError(
                "Warning: Running in serial mode but MPI detected multiple processes. For serial execution, run without mpirun.\n\n"
            )

    if execution_mode == "parallel" and size == 1:
        if rank == 0:
            raise ValueError("Parallel mode requested, but only 1 MPI process detected.\n\n")

    comm.Barrier()
    try:
        if rank == 0:
            clone_root = create_directories(data_dir)
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
        print(f"run_spotpy failed with error: {e} (Process rank {rank})\n\n")
        # restore_data_dir(clone_root)
        traceback.print_exc()

    return 0


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
