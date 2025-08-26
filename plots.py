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
    if not output_folder:
        output_folder = "./"
    if not parameternames:
        parameternames = get_parameternames(results)
    fig = plt.figure(figsize=(16, len(parameternames) * 3))
    names = ""
    i = 1
    for name in parameternames:
        ax = plt.subplot(len(parameternames), 1, i)
        ax.plot(results["par" + name], label=name)
        names += name + "_"
        ax.set_ylabel(name)
        if i == len(parameternames):
            ax.set_xlabel("Repetitions")
        if i == 1:
            ax.set_title("Parametertrace")
        ax.legend()
        i += 1
    fig.savefig(output_folder + fig_name)
    text = 'The figure as been saved as "' + output_folder + fig_name
    print(text)


def plot_parameterInteraction(results, fig_name="ParameterInteraction.png", output_folder=None):
    if not output_folder:
        output_folder = "./"
    parameterdistribtion = get_parameters(results)
    parameternames = get_parameternames(results)
    df = pd.DataFrame(np.asarray(parameterdistribtion).T.tolist(), columns=parameternames)

    pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal="kde")
    plt.savefig(output_folder + fig_name, dpi=300)


def plot_bestmodelrun(results, evaluation, fig_name="Best_model_run.png", output_folder=None):
    if not output_folder:
        output_folder = "./"

    fig = plt.figure(figsize=(16, 9))
    for i in range(len(evaluation)):
        if evaluation[i] == -9999:
            evaluation[i] = np.nan
    plt.plot(evaluation, "ro", markersize=1, label="Observation data")
    simulation_fields = get_simulation_fields(results)
    bestindex, bestobjf = get_maxlikeindex(results, verbose=False)
    plt.plot(
        list(results[simulation_fields][bestindex][0]),
        "b-",
        label="Obj=" + str(round(bestobjf, 2)),
    )
    plt.xlabel("Number of Observation Points")
    plt.ylabel("Simulated value")
    plt.legend(loc="upper right")
    fig.savefig(output_folder + fig_name, dpi=300)
    text = "A plot of the best model run has been saved as " + output_folder + fig_name
    print(text)


def plot_parameter_correlation(results, fig_name="ParameterCorrelation.png", output_folder=None):
    if not output_folder:
        output_folder = "./"
    """Create a correlation heatmap of parameters using seaborn"""
    parameterdistribution = get_parameters(results)
    parameternames = get_parameternames(results)

    df = pd.DataFrame(np.asarray(parameterdistribution).T.tolist(), columns=parameternames)

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
    plt.savefig(output_folder + fig_name, dpi=300, bbox_inches="tight")
    print(f'Correlation heatmap saved as "{output_folder + fig_name}"')


def create_interactive_plots(
    results, evaluation=None, output_folder=None, fig_name="spotpy_interactive.html"
):
    """Create all plots as interactive Bokeh visualizations in a single HTML file"""
    import os

    from bokeh.layouts import column, gridplot
    from bokeh.models import ColumnDataSource, HoverTool, TabPanel, Tabs
    from bokeh.palettes import Category10, Category20, RdYlBu11
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
    colors = Category10[10] if len(parameternames) <= 10 else Category20[20]

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

    trace_tab = TabPanel(child=column(*trace_plots), title="Parameter Traces")
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
                p.circle("x", "y", radius=3, color="navy", alpha=0.5, source=source)
                p.xaxis.axis_label = parameternames[j] if i == n_params - 1 else ""
                p.yaxis.axis_label = parameternames[i] if j == 0 else ""

            row_plots.append(p)
        scatter_plots.append(row_plots)

    scatter_grid = gridplot(scatter_plots, toolbar_location="right")
    interaction_tab = TabPanel(child=scatter_grid, title="Parameter Interactions")
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

    corr_tab = TabPanel(child=p_heat, title="Correlation Heatmap")
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
            "x",
            "y",
            radius=5,
            color="red",
            alpha=0.7,
            legend_label="Observations",
            source=obs_source,
        )

        # Plot best simulation
        x_sim = list(range(len(best_simulation)))
        sim_source = ColumnDataSource(data=dict(x=x_sim, y=best_simulation))
        p_best.line(
            "x", "y", line_width=2, color="blue", legend_label="Best Simulation", source=sim_source
        )

        p_best.legend.location = "top_right"
        p_best.legend.click_policy = "hide"

        best_tab = TabPanel(child=p_best, title="Best Model Run")
        tabs.append(best_tab)

    # Create final layout with tabs
    final_layout = Tabs(tabs=tabs)

    # Save
    save(final_layout)
    print(f'Interactive plots saved as "{save_path}"')
