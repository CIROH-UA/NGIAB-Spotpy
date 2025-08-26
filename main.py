import json
from datetime import datetime
from pathlib import Path

from cal_utils import process_usgs_streamflow, run_spotpy


def get_troute_output_name(path):
    with open(path, "r") as file:
        realization = json.load(file)
    start_date = datetime.strptime(realization["time"]["start_time"], "%Y-%m-%d %H:%M:%S")
    return f"troute_output_{start_date.strftime('%Y%m%d%H%M')}.nc"


gage_id = "10109001"
feature_id = 2861391
start_date = "2018-09-30"
end_date = "2022-09-30"
training_start_date = "2018-09-30"
# data_root = "/home/jovyan/ngiab_preprocess_output"
data_root = "/home/josh/work/ayman_cal/data/gage-10109001"

realization_path = f"{data_root}/gage-{gage_id}/config/realization.json"
observed_flow_path = f"{data_root}/{gage_id}_observed_flow.pkl"
troute_output_path = (
    f"{data_root}/gage-{gage_id}/outputs/troute/{get_troute_output_name(realization_path)}"
)
cfe_dir = f"{data_root}/gage-{gage_id}/config/cat_config/CFE"
data_dir = f"{data_root}/gage-{gage_id}"
noah_path = f"{data_root}/gage-{gage_id}/config/noah_owp/MPTABLE.TBL"
# print(troute_output_path)
# Optional: Retrieve and save observed flow
if not Path(observed_flow_path).exists():
    process_usgs_streamflow(gage_id, start_date, end_date, output_path=observed_flow_path)
# best_params = run_spotpy(gage_id, start_date, end_date, training_start_date,
#                          observed_flow_path, troute_output_path, cfe_dir, noah_path,
#                          data_dir, feature_id, repetitions=1000)
best_params = run_spotpy(
    gage_id,
    start_date,
    end_date,
    training_start_date,
    observed_flow_path,
    troute_output_path,
    cfe_dir,
    noah_path,
    data_dir,
    feature_id,
    algorithm="DDS",
    repetitions=200,
    dds_trials=1,
)
