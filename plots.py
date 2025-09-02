import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    results, parameternames=None, fig_name="Parameter_trace.png", output_folder=None
):
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
        import os

        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, fig_name)
    else:
        save_path = fig_name

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f'The figure has been saved as "{save_path}"')


def plot_parameterInteraction(results, fig_name="ParameterInteraction.png", output_folder=None):
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
        import os

        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, fig_name)
    else:
        save_path = fig_name

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f'Parameter interaction plot saved as "{save_path}"')


def plot_bestmodelrun(results, evaluation, fig_name="Best_model_run.png", output_folder=None):
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
        import os

        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, fig_name)
    else:
        save_path = fig_name

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"A plot of the best model run has been saved as {save_path}")


# Optional: Add a new function for correlation heatmap
def plot_parameter_correlation(results, fig_name="ParameterCorrelation.png", output_folder=None):
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
        import os

        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, fig_name)
    else:
        save_path = fig_name

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f'Correlation heatmap saved as "{save_path}"')


def create_interactive_plots(
    results, evaluation=None, output_folder=None, fig_name="spotpy_interactive.html"
):
    """Create all plots as interactive Bokeh visualizations in a single HTML file"""
    import os

    from bokeh.layouts import column, gridplot
    from bokeh.models import ColumnDataSource, HoverTool, Panel, Tabs
    from bokeh.palettes import Category10, RdYlBu11
    from bokeh.plotting import figure, output_file, save
    from bokeh.transform import linear_cmap

    # Handle output folder
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, fig_name)
    else:
        save_path = fig_name

    output_file(save_path)

    # Get data
    parameternames = get_parameternames(results)
    parameterdistribution = get_parameters(results)

    # Create tabs for different plot types
    tabs = []

    # 1. Parameter Traces Tab
    trace_plots = []
    colors = Category10[10] if len(parameternames) <= 10 else Category10[20]

    for i, name in enumerate(parameternames):
        data = results["par" + name]
        x_range = list(range(len(data)))

        p = figure(
            width=900,
            height=250,
            title=f"Parameter: {name}",
            x_axis_label="Repetitions",
            y_axis_label=name,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        )

        source = ColumnDataSource(data=dict(x=x_range, y=data))
        p.line("x", "y", source=source, line_width=2, color=colors[i % len(colors)])

        # Add hover tool
        hover = p.select_one(HoverTool)
        hover.tooltips = [("Iteration", "@x"), (name, "@y{0.0000}")]

        trace_plots.append(p)

    trace_tab = Panel(child=column(*trace_plots), title="Parameter Traces")
    tabs.append(trace_tab)

    # 2. Parameter Interactions Tab
    df = pd.DataFrame(np.asarray(parameterdistribution).T.tolist(), columns=parameternames)

    # Create scatter matrix
    scatter_plots = []
    n_params = len(parameternames)

    for i in range(n_params):
        row_plots = []
        for j in range(n_params):
            if i == j:
                # Diagonal - histogram
                hist, edges = np.histogram(df[parameternames[i]], bins=30)
                p = figure(width=200, height=200, tools="")
                p.quad(
                    top=hist,
                    bottom=0,
                    left=edges[:-1],
                    right=edges[1:],
                    fill_color="navy",
                    line_color="white",
                    alpha=0.5,
                )
                p.xaxis.axis_label = parameternames[i] if i == n_params - 1 else ""
                p.yaxis.axis_label = "Frequency" if j == 0 else ""
            else:
                # Off-diagonal - scatter
                p = figure(width=200, height=200, tools="pan,wheel_zoom,box_zoom,reset")
                source = ColumnDataSource(
                    data=dict(x=df[parameternames[j]], y=df[parameternames[i]])
                )
                p.circle("x", "y", size=3, color="navy", alpha=0.5, source=source)
                p.xaxis.axis_label = parameternames[j] if i == n_params - 1 else ""
                p.yaxis.axis_label = parameternames[i] if j == 0 else ""

            row_plots.append(p)
        scatter_plots.append(row_plots)

    scatter_grid = gridplot(scatter_plots, toolbar_location="right")
    interaction_tab = Panel(child=scatter_grid, title="Parameter Interactions")
    tabs.append(interaction_tab)

    # 3. Correlation Heatmap Tab
    corr_matrix = df.corr()

    # Prepare data for heatmap
    x_names = []
    y_names = []
    colors_heat = []
    alphas = []
    corr_values = []

    for i, xi in enumerate(parameternames):
        for j, yj in enumerate(parameternames):
            x_names.append(xi)
            y_names.append(yj)
            corr_val = corr_matrix.iloc[i, j]
            corr_values.append(corr_val)
            colors_heat.append(corr_val)
            alphas.append(abs(corr_val))

    source = ColumnDataSource(
        data=dict(
            x_names=x_names,
            y_names=y_names,
            colors=colors_heat,
            alphas=alphas,
            corr_values=corr_values,
        )
    )

    p_heat = figure(
        width=600,
        height=600,
        title="Parameter Correlation Matrix",
        x_range=parameternames,
        y_range=list(reversed(parameternames)),
        toolbar_location="right",
        tools="hover,save",
    )

    mapper = linear_cmap(field_name="colors", palette=RdYlBu11[::-1], low=-1, high=1)

    p_heat.rect(
        x="x_names",
        y="y_names",
        width=1,
        height=1,
        source=source,
        line_color=None,
        fill_color=mapper,
    )

    p_heat.xaxis.major_label_orientation = 45

    hover = p_heat.select_one(HoverTool)
    hover.tooltips = [("Parameters", "@x_names - @y_names"), ("Correlation", "@corr_values{0.00}")]

    corr_tab = Panel(child=p_heat, title="Correlation Heatmap")
    tabs.append(corr_tab)

    # 4. Best Model Run Tab (if evaluation data provided)
    if evaluation is not None:
        # Clean evaluation data
        evaluation = np.array(evaluation, dtype=float)
        evaluation[evaluation == -9999] = np.nan

        # Get best simulation
        simulation_fields = get_simulation_fields(results)
        bestindex, bestobjf = get_maxlikeindex(results, verbose=False)
        best_simulation = list(results[simulation_fields][bestindex][0])

        p_best = figure(
            width=900,
            height=500,
            title=f"Best Model Run (Objective = {bestobjf:.2f})",
            x_axis_label="Number of Observation Points",
            y_axis_label="Value",
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        )

        # Plot observations
        x_obs = list(range(len(evaluation)))
        obs_source = ColumnDataSource(data=dict(x=x_obs, y=evaluation))
        p_best.circle(
            "x", "y", size=5, color="red", alpha=0.7, legend_label="Observations", source=obs_source
        )

        # Plot best simulation
        x_sim = list(range(len(best_simulation)))
        sim_source = ColumnDataSource(data=dict(x=x_sim, y=best_simulation))
        p_best.line(
            "x", "y", line_width=2, color="blue", legend_label="Best Simulation", source=sim_source
        )

        p_best.legend.location = "top_right"
        p_best.legend.click_policy = "hide"

        best_tab = Panel(child=p_best, title="Best Model Run")
        tabs.append(best_tab)

    # Create final layout with tabs
    final_layout = Tabs(tabs=tabs)

    # Save
    save(final_layout)
    print(f'Interactive plots saved as "{save_path}"')
