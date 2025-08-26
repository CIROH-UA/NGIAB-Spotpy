import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spotpy
import spotpy.objectivefunctions as objf
import xarray as xr
from spotpy.parameter import Uniform

sys.path.append("/ngen/pyngiab")


def update_cfe_parameters(directory, parameters):
    """
    Updates specific parameters in all files within a directory.

    Parameters:
    - directory (str): Path to the directory containing the files to update.
    - parameters (dict): Dictionary of parameters and their new values.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        # Read the file content
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Update the parameters
        updated_lines = []
        for line in lines:
            for key, new_value in parameters.items():
                if line.startswith(key + "="):
                    line = f"{key}={new_value}\n"
            updated_lines.append(line)

        # Write the updated content back to the file
        with open(file_path, "w") as file:
            file.writelines(updated_lines)

    print(f"Parameters updated in all files within '{directory}'.")


def update_noah_parameters(file_path, param_updates):
    """
    Update selected NOAH LSM parameters in the MPTABLE.TBL file.

    Parameters:
        directory_path (str): Path to the 'noah_om/parameters' directory.
        param_updates (dict): Keys are parameter names (e.g., 'MFSNO'), values are strings to insert.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MPTABLE.TBL not found at {file_path}")

    with open(file_path, "r") as file:
        lines = file.readlines()
        print(f"Updating parameters in {file_path}...")

    is27 = True
    for key, value in param_updates.items():
        if key == "MFSNO":
            MFSNO27_str = ", ".join([f"{value}"] * 27)
            MFSNO20_str = ",   ".join([f"{value}"] * 19)
            lines[220] = f" {key} =  {value},   {MFSNO20_str},\n"

        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}"):
                if key == "MFSNO" and is27:
                    lines[i] = f" {key} =  {MFSNO27_str},\n"
                    is27 = False
                else:
                    lines[i] = f"  {key} = {value}\n"
                break

    with open(file_path, "w") as file:
        file.writelines(lines)


# NextGen calibration set-up

# Ensure hydroeval is installed
try:
    import hydroeval
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "hydroeval"])

# Ensure dataretrieval is installed
try:
    from dataretrieval import nwis
except ImportError:
    import subprocess

    subprocess.check_call(["pip", "install", "dataretrieval"])
    from dataretrieval import nwis


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
    if output_path:
        dfo_usgs_hr.to_pickle(output_path)
    return dfo_usgs_hr


# === Wrapper to Set Up NextGen Model Execution ===
class NextGenSetup:
    def __init__(
        self,
        gage_id,
        start_date,
        end_date,
        training_start_date,
        observed_flow_path,
        troute_output_path,
        cfe_dir,
        noah_path,
    ):
        self.gage_id = gage_id
        self.training_start_date = pd.to_datetime(training_start_date)
        self.end_date = pd.to_datetime(end_date)
        self.observed = pd.read_pickle(observed_flow_path)
        self.observed["Time"] = pd.to_datetime(self.observed["Time"]).dt.tz_localize(None)
        self.observed = self.observed[
            (self.observed["Time"] >= self.training_start_date)
            & (self.observed["Time"] <= self.end_date)
        ]
        self.observed = self.observed.set_index("Time")
        self.troute_output_path = troute_output_path
        self.cfe_parameters_directory_path = cfe_dir
        self.noah_params_path = noah_path

    def write_config(self, params):
        param_map = {
            "soil_params.b": params[0],
            "soil_params.satpsi": params[1],
            "soil_params.satdk": params[2],
            "soil_params.smcmax": params[3],
            "expon": params[4],
            "soil_params.slop": params[5],
            "K_nash_subsurface": params[6],
            "K_lf": params[7],
        }

        update_cfe_parameters(self.cfe_parameters_directory_path, param_map)

        # Create updated NOAH parameters dictionary
        noah_param_updates = {
            "MFSNO": params[8],  # Pass float directly
            "MP": params[9],
            "RSURF_EXP": params[10],
            "SNOW_EMIS": params[11],
            "CWP": params[12],
            "VCMX25": params[13],
            "RSURF_SNOW": params[14],
            "SCAMAX": params[15],
        }

        update_noah_parameters(self.noah_params_path, noah_param_updates)
        # Append run info to summary_df

    def run_model(self, data_dir):
        troute_output_path = (
            f"/home/jovyan/ngiab_preprocess_output/gage-{self.gage_id}/outputs/troute/"
        )
        for filename in os.listdir(troute_output_path):
            full_file_path = os.path.join(troute_output_path, filename)
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
                print("T-route has been removed from previous run")
        try:
            # model = PyNGIAB(data_dir, serial_execution_mode=True)
            # model.run()
            command = f'docker run --rm -it -v "{data_dir}:/ngen/ngen/data" joshcu/ngiab:fast /ngen/ngen/data/ auto 100 local'
            subprocess.run(command, shell=True)
            print("Next Gen run complete.")
        except:
            print("Next Gen run failed.")

    def evaluate(self, feature_id):
        ds = xr.open_dataset(self.troute_output_path)
        simulated = ds["flow"].sel(feature_id=feature_id).values
        simulated = simulated[ds["time"] >= self.training_start_date]
        simulated = simulated[: len(self.observed) - 1]
        return simulated


# === SPOTPY Setup Class for Calibration ===
class SpotpySetup:
    # CFE model parmaters
    soil_params_b = Uniform(2.0, 15.0)
    satpsi = Uniform(0.03, 0.955)
    satdk = Uniform(0.0000001, 0.000726)
    maxsmc = Uniform(0.5, 0.8)
    expon = Uniform(1.0, 8.0)
    slope = Uniform(0.0, 1.0)
    K_nash_subsurface = Uniform(0.01, 1.0)
    K_lf = Uniform(0.005, 1.0)

    # Additional NOAH OWP Modular parameters
    MFSNO = Uniform(low=1.0, high=5.0)  # multiplier on snowfall melt factor
    MP = Uniform(3.6, 12.6)
    RSURF_EXP = Uniform(4.5, 8.5)
    SNOW_EMIS = Uniform(low=0.90, high=1.0)  # snow emissivity
    CWP = Uniform(0.09, 0.36)
    VCMX25 = Uniform(24.0, 112.0)
    RSURF_SNOW = Uniform(0.001, 50.0)
    SCAMAX = Uniform(0.7, 1.0)

    def __init__(self, model_setup, data_dir, feature_id, algorithm_minimize):
        self.algorithm_minimize = algorithm_minimize
        self.model = model_setup
        self.data_dir = data_dir
        self.feature_id = feature_id
        self.run_id = 0
        # self.best_like = -np.inf
        self.best_like = np.inf
        self.best_run = -1

        # Ensure spotpy directory exists
        self.output_dir = "/home/jovyan/spotpy"
        os.makedirs(self.output_dir, exist_ok=True)

        self.summary_df = pd.DataFrame(
            columns=[
                "Run_ID",
                "soil_params.b",
                "satpsi",
                "satdk",
                "maxsmc",
                "expon",
                "slope",
                "K_nash_subsurface",
                "K_lf",
                "MFSNO",
                "MP",
                "RSURF_EXP",
                "SNOW_EMIS",
                "CWP",
                "VCMX25",
                "RSURF_SNOW",
                "SCAMAX",
                "KGE",
            ]
        )

    def simulation(self, vector):
        self.current_params = vector
        self.model.write_config(vector)
        self.model.run_model(self.data_dir)
        return self.model.evaluate(self.feature_id)

    def evaluation(self):
        return self.model.observed.values.squeeze()[1:]

    def objectivefunction(self, simulation, evaluation):
        rmse = objf.rmse(simulation, evaluation)
        if not self.algorithm_minimize:
            rmse = -rmse
        return rmse

        # kge = self.kling_gupta_efficiency(simulation, evaluation)

        # Plot each calibration iteration
        plt.figure(figsize=(10, 4))
        plt.plot(evaluation, label="Observed", color="black")
        plt.plot(simulation, label="Simulated", linestyle="--")
        # plt.text(0.5, 0.9, f'KGE: {kge:.2f}', transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.5, 0.9, f"RMSE: {rmse:.2f}", transform=plt.gca().transAxes, fontsize=12)
        plt.legend()
        plt.title(f"CFE-NOM-TR Streamflow for Gage {self.model.gage_id} - Run {self.run_id}")
        plt.xlabel("Time step")
        plt.ylabel("Streamflow [m3/sec]")
        plt.savefig(f"{self.output_dir}/spotpy_run_{self.run_id}.png")
        plt.close()

        # Append run info to summary_df
        param_row = {
            "Run_ID": self.run_id,
            "soil_params.b": self.current_params[0],
            "satpsi": self.current_params[1],
            "satdk": self.current_params[2],
            "maxsmc": self.current_params[3],
            "expon": self.current_params[4],
            "slope": self.current_params[5],
            "K_nash_subsurface": self.current_params[6],
            "K_lf": self.current_params[7],
            "MFSNO": self.current_params[8],
            "MP": self.current_params[9],
            "RSURF_EXP": self.current_params[10],
            "SNOW_EMIS": self.current_params[11],
            "CWP": self.current_params[12],
            "VCMX25": self.current_params[13],
            "RSURF_SNOW": self.current_params[14],
            "SCAMAX": self.current_params[15],
            "RMSE": rmse,
        }
        # 'KGE': kge
        #             }
        self.summary_df.loc[len(self.summary_df)] = param_row
        print(self.summary_df.tail(1).to_string(index=False))

        # if kge > self.best_like:
        #     self.best_like = kge
        #     self.best_run = self.run_id

        # self.run_id += 1
        # return kge

        if rmse < self.best_like:
            self.best_like = rmse
            self.best_run = self.run_id

        self.run_id += 1
        return rmse


# === Function to Run SPOTPY Calibration ===
def run_spotpy(
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
    algorithm,
    repetitions=25,
    dds_trials=5,
):
    # Model setup
    model_setup = NextGenSetup(
        gage_id,
        start_date,
        end_date,
        training_start_date,
        observed_flow_path,
        troute_output_path,
        cfe_dir,
        noah_path,
    )

    # set up optimizer
    if algorithm == "SCE":
        algorithm_minimize = True
    elif algorithm == "DDS":
        algorithm_minimize = False

    optimizer = SpotpySetup(model_setup, data_dir, feature_id, algorithm_minimize)

    # SCE hyperparameters
    if algorithm == "SCE":
        sampler = spotpy.algorithms.sceua(
            optimizer, dbname=f"spotpy_results_{gage_id}", dbformat="csv"
        )
        sampler.sample(repetitions, ngs=20)

    ## ADD hyperparameters
    elif algorithm == "DDS":
        sampler = spotpy.algorithms.dds(
            optimizer, dbname=f"spotpy_results_{gage_id}", dbformat="csv"
        )
        sampler.sample(repetitions, trials=int(dds_trials))

    results = spotpy.analyser.load_csv_results(f"spotpy_results_{gage_id}")
    best_params = spotpy.analyser.get_best_parameterset(results, maximize=False)

    # Plot objective function trace
    fig = plt.figure(1, figsize=(9, 5))
    plt.plot(results["like1"])
    # plt.ylabel('KGE')
    plt.ylabel("RMSE")
    plt.xlabel("Iteration")
    fig.savefig("/home/jovyan/spotpy/SCEUA_objectivefunctiontrace.png", dpi=300)

    # Plot best model run
    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
    best_model_run = results[bestindex]
    fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
    best_simulation = list(best_model_run[fields])
    time_index = pd.date_range(start=training_start_date, periods=len(best_simulation), freq="D")

    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    ax.plot(
        time_index,
        best_simulation,
        color="black",
        linestyle="solid",
        label="Best objf.=" + str(bestobjf),
    )
    ax.plot(time_index, optimizer.evaluation(), "r.", markersize=3, label="Observation data")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (m3/sec)")
    plt.legend(loc="upper right")
    fig.savefig("/home/jovyan/spotpy/SCEUA_best_modelrun.png", dpi=300)

    print(f"Best objective function value: {optimizer.best_like}")
    print(f"Best run ID: {optimizer.best_run}")

    # Save summary of all runs to csv file with highlight for best KGE/RMSE
    summary_csv_path = os.path.join(optimizer.self.output_dir, f"spotpy_summary_{gage_id}.csv")
    # best_kge_index = optimizer.summary_df['KGE'].idxmax()
    best_rmse_index = optimizer.summary_df["RMSE"].idxmin()

    optimizer.summary_df.to_csv(summary_csv_path)

    print(f"\nSaved highlighted summary to: {summary_csv_path}")

    return best_params
