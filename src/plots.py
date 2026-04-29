from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Any, Sequence
from flush_output import suppress_spotpy_syntax_warnings

suppress_spotpy_syntax_warnings()
from spotpy.analyser import (
    get_maxlikeindex,
    get_parameternames,
    get_parameters,
    get_simulation_fields,
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
    evaluation: np.ndarray,
    objective_function: str,
    invert_objective: bool,
    fig_name: str = "Best_model_run.png",
    output_folder: str | Path | None = None,
) -> None:
    """Plot best model run with seaborn styling"""
    # Set style for this plot
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(16, 9))

    # Clean evaluation data
    evaluation = np.array(evaluation, dtype=float)
    evaluation[evaluation == -9999] = np.nan

    # Plot observation data with seaborn styling
    x_obs = range(len(evaluation))
    sns.scatterplot(
        x=x_obs, y=evaluation, color="crimson", s=20, alpha=0.7, label="Observation data", ax=ax
    )

    # Get best simulation
    simulation_fields = get_simulation_fields(results)
    bestindex, bestobjf = get_maxlikeindex(results, verbose=False)
    best_simulation = list(results[simulation_fields][bestindex][0])

    #reversing what is done in the objective_function of spotpy
    if invert_objective:
        if objective_function == "KGE":
            bestobjf = 1 - bestobjf
        else:
            bestobjf = -bestobjf
    else:
        if objective_function == "KGE":
            bestobjf = bestobjf + 1

    # Plot best simulation with seaborn
    x_sim = range(len(best_simulation))
    sns.lineplot(
        x=x_sim,
        y=best_simulation,
        color="royalblue",
        linewidth=2,
        label=f"Best simulation (Obj={bestobjf:.2f})",
        ax=ax,
    )

    # Customize plot
    ax.set_xlabel("Number of Observation Points", fontsize=12)
    ax.set_ylabel("Simulated Value", fontsize=12)
    ax.set_title("Best Model Run", fontsize=14, fontweight="bold")

    # Improve legend
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=11)

    # Add subtle styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Handle output folder
    if output_folder:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        save_path = output_folder / fig_name
        csv_path = output_folder / "Best_model_run.csv"
    else:
        save_path = Path(fig_name)
        csv_path = Path("Best_model_run.csv")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"A plot of the best model run has been saved as {save_path}")

    with open(csv_path, "w") as f:
        f.write("index,observed,simulated\n")
        for i, (obs, sim) in enumerate(zip(evaluation, best_simulation)):
            f.write(f"{i},{obs},{sim}\n")
    print(f"Observed vs best simulated saved to {csv_path}")


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
