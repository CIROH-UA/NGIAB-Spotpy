import json
from datetime import datetime
from pathlib import Path

from cal_utils import process_usgs_streamflow, run_spotpy
from mpi4py import MPI

def get_troute_output_name(path):
    with open(path, "r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


gage_id = "10109001"
feature_id = 2861391
start_date = "2015-10-01"
end_date = "2019-12-01"
training_start_date = "2017-10-02"  # obs start 2017-10-01:7am
data_root = "/home/slama/Documents/hf3_remap/hf3_remap/output"

realization_path = f"{data_root}/gage-{gage_id}/config/realization.json"
observed_flow_path = f"{data_root}/{gage_id}_observed_flow.pkl"
troute_output_path = (
    f"{data_root}/gage-{gage_id}/outputs/troute/{get_troute_output_name(realization_path)}"
)
data_dir = f"{data_root}/gage-{gage_id}"
tensorboard_logdir = f"{data_dir}/tensorboard_logs"  # TensorBoard logs location

# Optional: Retrieve and save observed flow
if not Path(observed_flow_path).exists():
    process_usgs_streamflow(gage_id, start_date, end_date, output_path=observed_flow_path)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
try:
    print(f"Process rank {rank} before run_spotpy on host.")
    best_params = run_spotpy(
        gage_id,
        start_date,
        end_date,
        training_start_date,
        observed_flow_path,
        troute_output_path,
        data_dir,
        feature_id,
        algorithm="SCE",
        objective_function="KGE",
        repetitions=10,
        dds_trials=5,
        tensorboard_logdir=tensorboard_logdir,  # Add TensorBoard logging
    )
except Exception as e:
    rank = comm.Get_rank()
    print(f"run_spotpy failed {e} with process rank {rank}")

# save the best parameters to a file
with open(f"{data_dir}/spotpy/best_params.csv", "w") as file:
    header = ",".join([name[3:] for name in best_params[0].dtype.names])
    file.write(header + "\n")
    values = ",".join([str(value) for value in best_params[0]])
    file.write(values + "\n")

print("\nTo view TensorBoard results, run:")
print(f"tensorboard --logdir={tensorboard_logdir}")