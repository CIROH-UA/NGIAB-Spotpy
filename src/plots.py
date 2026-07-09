from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Any, Sequence
from flush_output import suppress_spotpy_syntax_warnings
from spotpy.objectivefunctions import kge, rmse

suppress_spotpy_syntax_warnings()
from spotpy.analyser import (
    get_maxlikeindex,
    get_parameternames,
    get_parameters,
    get_simulation_fields,
    get_minlikeindex,
)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook")


def plot_parametertrace(
    results: Any,
    parameternames: Sequence[str] | None = None,
    fig_name: str = "Parameter_trace.png",
    output_folder: str | Path | None = None,
) -> None:
    """Plot parameter traces using seaborn styling"""
    if not parameternames:
        parameternames = get_parameternames(results)

    # Create figure with seaborn styling
    fig, axes = plt.subplots(len(parameternames), 1, figsize=(16, len(parameternames) * 3))

    # Handle single parameter case
    if len(parameternames) == 1:
        axes = [axes]

    # Set color palette
    colors = sns.color_palette("husl", len(parameternames))

    for i, name in enumerate(parameternames):
        ax = axes[i]
        # Use seaborn line plot styling
        data = results["par" + name]
        x_range = range(len(data))

        # Plot with seaborn styling
        sns.lineplot(x=x_range, y=data, ax=ax, color=colors[i], linewidth=1.5)

        # Customize axes
        ax.set_ylabel(name, fontsize=11)
        ax.set_xlabel("Repetitions" if i == len(parameternames) - 1 else "")

        # Add title only to first subplot
        if i == 0:
            ax.set_title("Parameter Trace", fontsize=14, fontweight="bold")

        # Add legend with parameter name
        ax.legend([name], loc="upper right", frameon=True, fancybox=True)

        # Add subtle grid
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Handle output folder
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        save_path = output_folder / fig_name
    else:
        save_path = Path(fig_name)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f'The figure has been saved as "{save_path}"')


def plot_parameterInteraction(
    results: Any,
    fig_name: str = "ParameterInteraction.png",
    output_folder: str | Path | None = None,
) -> None:
    """Create parameter interaction matrix using seaborn pairplot"""
    parameterdistribution = get_parameters(results)
    parameternames = get_parameternames(results)

    # Create DataFrame
    df = pd.DataFrame(np.asarray(parameterdistribution).T.tolist(), columns=parameternames)

    # Create pairplot with seaborn
    g = sns.pairplot(
        df,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 10, "edgecolor": None, "linewidth": 0},
        diag_kws={"linewidth": 2, "alpha": 0.7},
        corner=False,
    )

    # Customize the plot
    g.fig.suptitle("Parameter Interactions", y=1.02, fontsize=14, fontweight="bold")

    # Adjust layout and save
    plt.tight_layout()

    # Handle output folder
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        save_path = output_folder / fig_name
    else:
        save_path = Path(fig_name)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f'Parameter interaction plot saved as "{save_path}"')


def plot_bestmodelrun(
    results: Any,
    optimizer: Any,
    objective_function: str,
    algorithm_maximizes: bool,
    best_is_higher:bool,
    fig_name: str = "Best_Model_Run",
    output_folder: str | Path | None = None,
) -> None:
    """Plot best model run with seaborn styling"""

    # Set style for this plot
    sns.set_style("darkgrid")

    evaluation = optimizer.evaluation()

    counter_evaluation = 0

    for i, key in enumerate(optimizer.model.target_variables.keys()):

        # Clean evaluation data
        evaluation_key = np.array(evaluation[i], dtype=float)
        evaluation_key[evaluation_key == -9999] = np.nan

        # Get best simulation indices
        simulation_fields = get_simulation_fields(results)

        if algorithm_maximizes:
            bestindex, bestobjf = get_maxlikeindex(results, verbose=False)
            bestindex = bestindex[0][0]
        else:
            bestindex, bestobjf = get_minlikeindex(results, verbose=False)

        history_df = pd.read_csv(output_folder.parent / f"{objective_function}_history.csv")
        column_name = f"{objective_function}_{key}"
        if best_is_higher:
            bestindexindividual = history_df[column_name].idxmax()
        else:
            bestindexindividual = history_df[column_name].idxmin()

        bestindividualobjf = history_df[column_name].iloc[bestindexindividual]

        # Extract simulations
        best_simulation = list(
            results[simulation_fields][bestindex]
        )[counter_evaluation:counter_evaluation + len(evaluation_key)]

        best_simulation_individual = list(
            results[simulation_fields][bestindexindividual]
        )[counter_evaluation:counter_evaluation + len(evaluation_key)]

        if objective_function == "KGE":
            bestobjf_without_invert = kge(evaluation_key,best_simulation)
        else:
            bestobjf_without_invert = rmse(evaluation_key,best_simulation)

        # ================================
        # Plot best simulation
        # ================================

        fig, ax = plt.subplots(figsize=(16, 9))

        sns.scatterplot(
            x=range(len(evaluation_key)),
            y=evaluation_key,
            color="crimson",
            s=20,
            alpha=0.7,
            label="Observation data",
            ax=ax,
        )

        sns.lineplot(
            x=range(len(best_simulation)),
            y=best_simulation,
            color="royalblue",
            linewidth=2,
            label=f"Best simulation (weighted_objective_func={bestobjf:.2f}) ({objective_function} = {bestobjf_without_invert})",
            ax=ax,
        )

        ax.set_xlabel("Number of Observation Points", fontsize=12)
        ax.set_ylabel(f"Simulated Value_{key}", fontsize=12)
        ax.set_title(f"Best Model Run {key}. Iteration {bestindex}", fontsize=14, fontweight="bold")

        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()


        # Save first plot
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

            save_path = output_folder / f"{fig_name}_{key}.png"
            csv_path = output_folder / f"{fig_name}_{key}.csv"

        else:
            save_path = Path(f"{fig_name}_{key}.png")
            csv_path = Path(f"{fig_name}_{key}.csv")


        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Best simulation plot saved as {save_path}")


        with open(csv_path, "w") as f:
            f.write("index,observed,simulated\n")

            for idx, (obs, sim) in enumerate(zip(evaluation_key, best_simulation)):
                f.write(f"{idx},{obs},{sim}\n")

        print(f"Best simulation CSV saved as {csv_path}")



        # ================================
        # Plot best individual simulation
        # ================================

        fig_name = "Best_Individual_Run"
        fig, ax = plt.subplots(figsize=(16, 9))


        sns.scatterplot(
            x=range(len(evaluation_key)),
            y=evaluation_key,
            color="crimson",
            s=20,
            alpha=0.7,
            label="Observation data",
            ax=ax,
        )


        sns.lineplot(
            x=range(len(best_simulation_individual)),
            y=best_simulation_individual,
            color="orange",
            linewidth=2,
            label=f"Best individual simulation ({objective_function}={bestindividualobjf:.2f})",
            ax=ax,
        )


        ax.set_xlabel("Number of Observation Points", fontsize=12)
        ax.set_ylabel(f"Simulated Value_{key}", fontsize=12)
        ax.set_title(
            f"Best Individual Model Run {key}. Iteration {bestindexindividual}",
            fontsize=14,
            fontweight="bold",
        )


        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()


        # Save second plot
        if output_folder:
            save_path = output_folder / f"{fig_name}_{key}.png"
            csv_path = output_folder / f"Best_model_run_{key}.csv"

        else:
            save_path = Path(f"{fig_name}_{key}.png")
            csv_path = Path(f"{fig_name}_{key}.csv")


        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Best individual plot saved as {save_path}")


        with open(csv_path, "w") as f:
            f.write("index,observed,simulated\n")

            for idx, (obs, sim) in enumerate(zip(evaluation_key, best_simulation_individual)):
                f.write(f"{idx},{obs},{sim}\n")

        print(f"Best individual CSV saved as {csv_path}")
        counter_evaluation += len(evaluation_key)
        fig_name = "Best_Model_Run"  # Reset fig_name for next iteration


# Optional: Add a new function for correlation heatmap
def plot_parameter_correlation(
    results: Any,
    fig_name: str = "ParameterCorrelation.png",
    output_folder: str | Path | None = None,
) -> None:
    """Create a correlation heatmap of parameters using seaborn"""
    parameterdistribution = get_parameters(results)
    parameternames = get_parameternames(results)

    # Create DataFrame
    df = pd.DataFrame(np.asarray(parameterdistribution).T.tolist(), columns=parameternames)

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Parameter Correlation Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Handle output folder
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        save_path = output_folder / fig_name
    else:
        save_path = Path(fig_name)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f'Correlation heatmap saved as "{save_path}"')
