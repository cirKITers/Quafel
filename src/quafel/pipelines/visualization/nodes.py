"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from typing import Dict, List
import pandas as pd

from bisect import bisect_left
from math import log10, floor
import logging


import numpy as np

import os

duration_perf_regex = r"duration_perf_\d*"
duration_proc_regex = r"duration_proc_\d*"
expressibility_regex = r"expressibility"
entangling_capability = r"entangling_capability"


class design:
    # see https://plotly.com/python/discrete-color/
    qual_main = px.colors.qualitative.Dark2  # set1
    qual_second = px.colors.qualitative.Pastel2  # pastel1

    seq_main = px.colors.sequential.thermal  # alternative: pastel1

    print_figure_title = False

    title_font_size = 18
    legend_font_size = 16

    scatter_legend = dict(
        orientation="v",
        traceorder="normal",
        font=dict(
            size=legend_font_size,
        ),
    )

    base_theme = "simple_white"

    include_framework_term = (
        False  # Set to true to get e.g. 'Framework Pennylane' instead of 'Pennylane'
    )

    standard_ticks_angle = 0
    long_ticks_angle = -40
    long_ticks = 3

    showgrid = False

    time_tick_type = "log"
    time_dtick = 1

    log_tick_type = "log"
    standard_tick_type = "linear"

    heatmap_axis_mode = "array"
    scatter_axis_mode = "array"

    scatter_mode_c = "lines+markers"
    scatter_mode_hl = "lines"
    marker_color = dict(color="#444")


def rgb_to_rgba(rgb_value: str, alpha: float):
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def get_time_scale(pd_time: pd.array):
    """This method takes an array of timestamps and finds the
    approriate time scale as well as the corresponding (exponential) factor.

    Args:
        pd_time (pd.array): pandas array of timestamps
    """

    def find_exp(number) -> int:
        """
        From https://stackoverflow.com/questions/64183806/extracting-the-exponent-from-scientific-notation # noqa
        """
        base10 = log10(abs(number))
        return floor(base10)

    mid = np.mean(
        [
            find_exp(pd_time.min().min()),
            find_exp(pd_time.max().max()),
        ]
    ).astype(int)

    si_time = [["ns", "us", "ms", "s"], [-9, -6, -3, 0], [1e9, 1e6, 1e3, 1]]

    idx = bisect_left(si_time[1], mid)

    return (si_time[0][idx], si_time[2][idx])


def extract_framework_name_from_id(identifier):
    """
    Global method to generate display name from class name.
    E.g.: qiskit_fw -> Qiskit Framework
    """
    if design.include_framework_term:
        return identifier.replace("fw", "framework").capitalize().replace("_", " ")
    else:
        return identifier.replace("fw", "").capitalize().replace("_", " ")


def heatmap_viz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    z_title: str,
    x_title: str,
    log_x: bool,
    y_title: str,
    log_y: bool,
    plot_title: str,
):
    fig = go.Figure(
        [
            go.Heatmap(
                x=np.log2(x) if log_x else x,
                y=np.log2(y) if log_y else y,
                z=np.log10(z),
                colorscale=design.seq_main,
                colorbar=dict(
                    title=z_title,
                    tickvals=np.log10([z.min(), z.mean(), z.max()]),
                    ticktext=[f"{z.min():.2}", f"{z.mean():.2}", f"{z.max():.2}"],
                ),
            )
        ]
    )
    fig.update_layout(
        yaxis=dict(
            type="linear",
            tickmode=design.heatmap_axis_mode,
            tickvals=np.log2(y) if log_y else y,
            ticktext=y if log_y else None,
            tickangle=(
                design.long_ticks_angle
                if len(str(max(y))) >= design.long_ticks
                else design.standard_ticks_angle
            ),
            title=y_title,
            showgrid=design.showgrid,
        ),
        xaxis=dict(
            type="linear",
            tickmode=design.heatmap_axis_mode,
            tickvals=np.log2(x) if log_x else x,
            ticktext=x if log_x else None,
            tickangle=(
                design.long_ticks_angle
                if len(str(max(x))) >= design.long_ticks
                else design.standard_ticks_angle
            ),
            title=x_title,
            showgrid=design.showgrid,
        ),
        title=dict(
            text=plot_title if design.print_figure_title else "",
            font=dict(
                size=design.title_font_size,
            ),
        ),
        hovermode="x",
        font=dict(
            size=design.legend_font_size,
        ),
    )
    return fig


def scatter_viz(
    fig: go.Figure,
    name: str,
    main_color_sel: str,
    sec_color_sel: str,
    x: np.ndarray,
    y: np.ndarray,
    x_title: str,
    log_x: bool,
    y_title: str,
    log_y: bool,
    plot_title: str,
    y_max: np.ndarray = None,
    y_min: np.ndarray = None,
):
    fig.add_trace(
        go.Scatter(
            name=f"{name}",
            x=x,
            y=y,
            mode=design.scatter_mode_c,
            line=dict(color=main_color_sel),
        ),
    )
    if y_max is not None and y_min is not None:
        fig.add_trace(
            go.Scatter(
                name=f"{name} - High",
                x=x,
                y=y_max,
                mode=design.scatter_mode_hl,
                marker=design.marker_color,
                line=dict(width=0),
                showlegend=False,
            ),
        )
        fig.add_trace(
            go.Scatter(
                name=f"{name} - Low",
                x=x,
                y=y_min,
                marker=design.marker_color,
                line=dict(width=0),
                mode=design.scatter_mode_hl,
                fillcolor=sec_color_sel,
                fill="tonexty",
                showlegend=False,
            ),
        )
    fig.update_layout(
        xaxis=dict(
            type=design.log_tick_type if log_x else design.standard_tick_type,
            tickmode=design.scatter_axis_mode,
            tickvals=x,
            tickangle=(
                design.long_ticks_angle
                if len(str(max(x))) >= design.long_ticks
                else design.standard_ticks_angle
            ),
            title=x_title,
            showgrid=design.showgrid,
        ),
        yaxis=(
            dict(
                title=y_title,
                type=design.log_tick_type if log_y else None,
                dtick=design.time_dtick if log_y else None,
                showgrid=design.showgrid,
            )
        ),
        title=dict(
            text=plot_title if design.print_figure_title else "",
            font=dict(
                size=design.title_font_size,
            ),
        ),
        hovermode="x",
        font=dict(
            size=design.legend_font_size,
        ),
        legend=(design.scatter_legend),
        template=design.base_theme,
    )


def shots_qubits_viz(evaluations_combined: Dict):
    figures = {}

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_qubit = qubit_depth_duration.groupby("qubits")

        for q, depth_duration in grouped_by_qubit:
            duration_sorted_by_depth = depth_duration.sort_values("depth")
            durations = duration_sorted_by_depth.filter(regex=duration_perf_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)

            q = int(q)

            figures[f"{fw}_qubits_{q}_time"] = heatmap_viz(
                x=duration_sorted_by_depth["shots"].astype(int),
                y=duration_sorted_by_depth["depth"].astype(int),
                z=duration_mean,
                z_title=f"Time ({si_time})",
                x_title="# of Shots",
                log_x=True,
                y_title="Circuit Depth",
                log_y=True,
                plot_title=f"{framework_name} @ {q} Qubits: "
                f"Circuit Depth and # of Shots",
            )

    return figures


def shots_depths_viz(evaluations_combined: Dict):
    figures = {}

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_duration in grouped_by_depth:
            duration_sorted_by_qubit = qubit_duration.sort_values("qubits")
            durations = duration_sorted_by_qubit.filter(regex=duration_perf_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)

            d = int(d)
            # image = []
            # for s, duration in grouped_by_shots_sorted_by_depth:
            #     image.append(duration['4'].to_numpy())

            figures[f"{fw}_depth_{d}_time"] = heatmap_viz(
                x=duration_sorted_by_qubit["shots"].astype(int),
                y=duration_sorted_by_qubit["qubits"].astype(int),
                z=duration_mean,
                z_title=f"Time ({si_time})",
                x_title="# of Shots",
                log_x=True,
                y_title="# of Qubits",
                log_y=False,
                plot_title=f"{framework_name} @ Circuit Depth {d}: "
                f"# of qubits and # of Shots",
            )

    return figures


def depth_qubits_viz(evaluations_combined: Dict):
    figures = {}

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")

    for fw, qubit_depth_duration in grouped_by_fw:
        framework_name = extract_framework_name_from_id(fw)
        grouped_by_shots = qubit_depth_duration.groupby("shots")

        for s, shots_duration in grouped_by_shots:
            duration_sorted_by_depth = shots_duration.sort_values("depth")
            durations = duration_sorted_by_depth.filter(regex=duration_perf_regex)

            durations *= factor_time

            duration_mean = durations.mean(axis=1)

            s = int(s)

            figures[f"{fw}_shots_{s}_time"] = heatmap_viz(
                x=duration_sorted_by_depth["qubits"].astype(int),
                y=duration_sorted_by_depth["depth"].astype(int),
                z=duration_mean,
                z_title=f"Time ({si_time})",
                x_title="# of Qubits",
                log_x=False,
                y_title="Circuit Depth",
                log_y=True,
                plot_title=f"{framework_name} @ {s} Shots: "
                f"Circuit Depth and # of Qubits",
            )

    return figures


def qubits_time_viz(evaluations_combined: Dict, skip_frameworks: List):
    figures = {}

    # those two color sets are well suited as they correspond regarding
    # their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        if fw in skip_frameworks:
            continue

        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)
        framework_name = extract_framework_name_from_id(fw)

        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_shots_duration in grouped_by_depth:
            grouped_by_shots = qubit_shots_duration.groupby("shots")

            for s, fw_qubit_duration in grouped_by_shots:
                duration_sorted_by_qubit = fw_qubit_duration.sort_values("qubits")

                durations = duration_sorted_by_qubit.filter(regex=duration_perf_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                d = int(d)
                s = int(s)
                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"shots_{s}_depth_{d}_time" not in figures:
                    figures[f"shots_{s}_depth_{d}_time"] = go.Figure()

                scatter_viz(
                    fig=figures[f"shots_{s}_depth_{d}_time"],
                    name=f"{framework_name}",
                    main_color_sel=main_color_sel,
                    sec_color_sel=sec_color_sel,
                    x=duration_sorted_by_qubit["qubits"],
                    y=durations_mean,
                    y_min=durations_min,
                    y_max=durations_max,
                    x_title="# of Qubits",
                    log_x=False,
                    y_title=f"Time ({si_time})",
                    log_y=True,
                    plot_title=f"Duration per Framework over "
                    f"# of Qubits @ {s} Shots, Circuit Depth {d}",
                )

    return figures


def shots_time_viz(evaluations_combined: Dict, skip_frameworks: List):
    figures = {}

    # those two color sets are well suited as they correspond regarding
    # their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        if fw in skip_frameworks:
            continue

        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)
        framework_name = extract_framework_name_from_id(fw)

        grouped_by_depth = qubit_depth_duration.groupby("depth")

        for d, qubit_shots_duration in grouped_by_depth:
            grouped_by_qubits = qubit_shots_duration.groupby("qubits")

            for q, fw_shots_duration in grouped_by_qubits:
                duration_sorted_by_shots = fw_shots_duration.sort_values("shots")

                durations = duration_sorted_by_shots.filter(regex=duration_perf_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                q = int(q)
                d = int(d)
                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"qubits_{q}_depth_{d}_time" not in figures:
                    figures[f"qubits_{q}_depth_{d}_time"] = go.Figure()

                scatter_viz(
                    fig=figures[f"qubits_{q}_depth_{d}_time"],
                    name=f"{framework_name}",
                    main_color_sel=main_color_sel,
                    sec_color_sel=sec_color_sel,
                    x=duration_sorted_by_shots["shots"].astype(int),
                    y=durations_mean,
                    y_min=durations_min,
                    y_max=durations_max,
                    x_title="# of Shots",
                    log_x=True,
                    y_title=f"Time ({si_time})",
                    log_y=True,
                    plot_title=f"Duration per Framework over "
                    f"# of Shots @ {q} Qubits, Circuit Depth {d}",
                )

    return figures


def depth_time_viz(evaluations_combined: Dict, skip_frameworks: List):
    figures = {}

    # those two color sets are well suited as they correspond regarding
    # their color value but differ from their luminosity and saturation values
    main_colors_it = iter(design.qual_main)
    sec_colors_it = iter(design.qual_second)

    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    grouped_by_fw = evaluations_combined.groupby("framework")
    for fw, qubit_depth_duration in grouped_by_fw:
        if fw in skip_frameworks:
            continue

        main_color_sel = next(main_colors_it)
        sec_color_sel = rgb_to_rgba(next(sec_colors_it), 0.2)
        framework_name = extract_framework_name_from_id(fw)

        grouped_by_qubits = qubit_depth_duration.groupby("qubits")

        for q, depth_shots_duration in grouped_by_qubits:
            grouped_by_shots = depth_shots_duration.groupby("shots")

            for s, fw_depth_duration in grouped_by_shots:
                duration_sorted_by_depth = fw_depth_duration.sort_values("depth")

                durations = duration_sorted_by_depth.filter(regex=duration_perf_regex)
                durations *= factor_time

                durations_mean = durations.mean(axis=1)
                durations_max = durations.max(axis=1)
                durations_min = durations.min(axis=1)

                q = int(q)
                s = int(s)
                # image = []
                # for s, duration in grouped_by_shots_sorted_by_depth:
                #     image.append(duration['4'].to_numpy())

                if f"shots_{s}_qubits_{q}_time" not in figures:
                    figures[f"shots_{s}_qubits_{q}_time"] = go.Figure()

                scatter_viz(
                    fig=figures[f"shots_{s}_qubits_{q}_time"],
                    name=f"{framework_name}",
                    main_color_sel=main_color_sel,
                    sec_color_sel=sec_color_sel,
                    x=duration_sorted_by_depth["depth"].astype(int),
                    y=durations_mean,
                    y_min=durations_min,
                    y_max=durations_max,
                    x_title="Circuit Depth",
                    log_x=True,
                    y_title=f"Time ({si_time})",
                    log_y=True,
                    plot_title=f"Duration per Framework over "
                    f"Circuit Depth @ {s} Shots, {q} Qubits",
                )

    return figures


def qubits_measures_viz(evaluations_combined: Dict) -> Dict[str, go.Figure]:
    """Returns a dictionary of subplots, where the keys are of the form
    'shots_<n>_depth_<m>_measures', where <n> and <m> are the shot number and
    circuit depth, respectively. Each subplot shows the mean expressibility
    and mean entangling capability (over all qubits) of each framework for a
    given shot number and circuit depth.

    Args:
        evaluations_combined (Dict): A dataframe containing the evaluations.

    Returns:
        Dict[str, Figure]: A dictionary of subplots, where the keys are of the
            form 'shots_<n>_depth_<m>_measures', and the values are subplots
            objects.
    """

    figures = {}

    grouped_by_depth = evaluations_combined.groupby("depth")

    for d, qubit_shots_duration in grouped_by_depth:
        grouped_by_shots = qubit_shots_duration.groupby("shots")

        for s, fw_qubit_duration in grouped_by_shots:
            duration_sorted_by_qubit = fw_qubit_duration.sort_values("qubits")

            expressiblities = duration_sorted_by_qubit.filter(
                regex=expressibility_regex
            ).mean(axis=1)
            entangling_capabilities = duration_sorted_by_qubit.filter(
                regex=entangling_capability
            ).mean(axis=1)

            d = int(d)
            s = int(s)

            figures[f"shots_{s}_depth_{d}_measures"] = go.Figure()

            scatter_viz(
                fig=figures[f"shots_{s}_depth_{d}_measures"],
                name=f"Entangling Capability",
                main_color_sel=design.qual_main[0],
                sec_color_sel=rgb_to_rgba(design.qual_second[0], 0.2),
                x=duration_sorted_by_qubit["qubits"],
                y=entangling_capabilities,
                x_title="# of Qubits",
                log_x=False,
                y_title=f"Measures",
                log_y=False,
                plot_title=f"Measures per Framework over "
                f"# of Qubits @ {s} Shots, Circuit Depth {d}",
            )

            scatter_viz(
                fig=figures[f"shots_{s}_depth_{d}_measures"],
                name=f"Expressibility",
                main_color_sel=design.qual_main[1],
                sec_color_sel=rgb_to_rgba(design.qual_second[1], 0.2),
                x=duration_sorted_by_qubit["qubits"],
                y=expressiblities,
                x_title="# of Qubits",
                log_x=False,
                y_title=f"Measures",
                log_y=False,
                plot_title=f"Measures per Framework over "
                f"# of Qubits @ {s} Shots, Circuit Depth {d}",
            )

    return figures


def shots_measures_viz(evaluations_combined: Dict):
    """
    Visualizes measures based on shots and depth.

    Parameters:
    - evaluations_combined: a dictionary containing evaluations

    Returns:
    - figures: a dictionary containing the visualizations
    """
    figures = {}

    grouped_by_depth = evaluations_combined.groupby("depth")

    for d, qubit_shots_duration in grouped_by_depth:
        grouped_by_qubits = qubit_shots_duration.groupby("qubits")

        for q, fw_shots_duration in grouped_by_qubits:
            duration_sorted_by_shots = fw_shots_duration.sort_values("shots")

            expressiblities = duration_sorted_by_shots.filter(
                regex=expressibility_regex
            ).mean(axis=1)
            entangling_capabilities = duration_sorted_by_shots.filter(
                regex=entangling_capability
            ).mean(axis=1)

            q = int(q)
            d = int(d)

            figures[f"qubits_{q}_depth_{d}_measures"] = go.Figure()

            scatter_viz(
                fig=figures[f"qubits_{q}_depth_{d}_measures"],
                name=f"Entangling Capability",
                main_color_sel=design.qual_main[0],
                sec_color_sel=rgb_to_rgba(design.qual_second[0], 0.2),
                x=duration_sorted_by_shots["shots"].astype(int),
                y=entangling_capabilities,
                x_title="# of Shots",
                log_x=True,
                y_title=f"Measures",
                log_y=False,
                plot_title=f"Measures per Framework over "
                f"# of Shots @ {q} Qubits, Circuit Depth {d}",
            )

            scatter_viz(
                fig=figures[f"qubits_{q}_depth_{d}_measures"],
                name=f"Expressibility",
                main_color_sel=design.qual_main[1],
                sec_color_sel=rgb_to_rgba(design.qual_second[1], 0.2),
                x=duration_sorted_by_shots["shots"].astype(int),
                y=expressiblities,
                x_title="# of Shots",
                log_x=True,
                y_title=f"Measures",
                log_y=False,
                plot_title=f"Measures per Framework over "
                f"# of Shots @ {q} Qubits, Circuit Depth {d}",
            )

    return figures


def depth_measures_viz(evaluations_combined: Dict):
    """
    Generates visualizations of measures for different qubits and
    shots based on the provided evaluations.

    Parameters:
    - evaluations_combined (Dict): A dictionary containing
    evaluations combined by qubits and shots.

    Returns:
    - figures (Dict): A dictionary containing visualizations of
    measures for different qubits and shots.
    """
    figures = {}

    grouped_by_qubits = evaluations_combined.groupby("qubits")

    for q, depth_shots_duration in grouped_by_qubits:
        grouped_by_shots = depth_shots_duration.groupby("shots")

        for s, fw_depth_duration in grouped_by_shots:
            duration_sorted_by_depth = fw_depth_duration.sort_values("depth")

            expressiblities = duration_sorted_by_depth.filter(
                regex=expressibility_regex
            ).mean(axis=1)
            entangling_capabilities = duration_sorted_by_depth.filter(
                regex=entangling_capability
            ).mean(axis=1)

            q = int(q)
            s = int(s)

            figures[f"shots_{s}_qubits_{q}_measures"] = go.Figure()

            scatter_viz(
                fig=figures[f"shots_{s}_qubits_{q}_measures"],
                name=f"Entangling Capability",
                main_color_sel=design.qual_main[0],
                sec_color_sel=rgb_to_rgba(design.qual_second[0], 0.2),
                x=duration_sorted_by_depth["depth"].astype(int),
                y=entangling_capabilities,
                x_title="Circuit Depth",
                log_x=True,
                y_title=f"Measures",
                log_y=False,
                plot_title=f"Measures per Framework over "
                f"Circuit Depth @ {s} Shots, {q} Qubits",
            )

            scatter_viz(
                fig=figures[f"shots_{s}_qubits_{q}_measures"],
                name=f"Expressibility",
                main_color_sel=design.qual_main[1],
                sec_color_sel=rgb_to_rgba(design.qual_second[1], 0.2),
                x=duration_sorted_by_depth["depth"].astype(int),
                y=expressiblities,
                x_title="Circuit Depth",
                log_x=True,
                y_title=f"Measures",
                log_y=False,
                plot_title=f"Measures per Framework over "
                f"Circuit Depth @ {s} Shots, {q} Qubits",
            )

    return figures


def extract_tests(evaluations_combined: Dict):
    si_time, factor_time = get_time_scale(
        evaluations_combined.filter(regex=duration_perf_regex)
    )

    durations_perf = evaluations_combined.filter(regex=duration_perf_regex)

    durations_proc = evaluations_combined.filter(regex=duration_proc_regex)

    n_runs = durations_perf.shape[0]
    n_evals = durations_perf.shape[1]

    evals_perf_mean = durations_perf.mean(axis=0)
    evals_proc_mean = durations_proc.mean(axis=0)

    log = logging.getLogger(__name__)
    log.info(f"Received {n_evals} evaluations and {n_runs} runs")
    log.info(f"Evaluation mean across all runs (perf timing): {evals_perf_mean.mean()}")
    log.info(
        f"Evaluation deviation across all runs (perf timing): {evals_perf_mean.max() - evals_perf_mean.min()}"
    )
    log.info(f"Evaluation mean across all runs (proc timing): {evals_proc_mean.mean()}")
    log.info(
        f"Evaluation deviation across all runs (proc timing): {evals_proc_mean.max() - evals_proc_mean.min()}"
    )

    return {}


def export_selected(evaluations_combined, additional_figures, output_folder, **figures):
    max_qubits = evaluations_combined["qubits"].max()
    max_depth = evaluations_combined["depth"].max()
    max_shots = evaluations_combined["shots"].max()
    frameworks = evaluations_combined["framework"].unique()

    def export(fig, name, folder):
        # Disable warnings to prevent printing a box at the bottom left of the figure.
        # See this issue: https://github.com/plotly/plotly.py/issues/3469
        pio.full_figure_for_development(fig, warn=False)
        # Disable mathjax completely since above did not help.
        # See this issue: https://github.com/plotly/Kaleido/issues/122
        pio.kaleido.scope.mathjax = None

        fig.write_image(os.path.join(folder, f"{name}.pdf"), engine="kaleido")
        # set scale=3 to increase resolution in resulting plots
        fig.write_image(os.path.join(folder, f"{name}.png"), engine="kaleido", scale=3)

    sel = f"shots_{max_shots}_depth_{max_depth}_time"
    export(figures[sel], sel, output_folder)

    sel = f"shots_{max_shots}_qubits_{max_qubits}_time"
    export(figures[sel], sel, output_folder)

    sel = f"qubits_{max_qubits}_depth_{max_depth}_time"
    export(figures[sel], sel, output_folder)

    sel = f"shots_{max_shots}_depth_{max_depth}_measures"
    export(figures[sel], sel, output_folder)

    sel = f"shots_{max_shots}_qubits_{max_qubits}_measures"
    export(figures[sel], sel, output_folder)

    sel = f"qubits_{max_qubits}_depth_{max_depth}_measures"
    export(figures[sel], sel, output_folder)

    for fw in frameworks:
        sel = f"{fw}_qubits_{max_qubits}_time"
        export(figures[sel], sel, output_folder)

        sel = f"{fw}_depth_{max_depth}_time"
        export(figures[sel], sel, output_folder)

        sel = f"{fw}_shots_{max_shots}_time"
        export(figures[sel], sel, output_folder)

    for add_fig in additional_figures:
        export(figures[add_fig], add_fig, output_folder)

    return {}
