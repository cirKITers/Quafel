"""
This is a boilerplate pipeline 'visualization'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.visualization.nodes import (
    shots_depths_viz,
    shots_qubits_viz,
    depth_qubits_viz,
    qubits_time_viz,
    depth_time_viz,
    shots_time_viz,
    depth_measures_viz,
    qubits_measures_viz,
    shots_measures_viz,
    export_selected,
    extract_tests,
)


def create_pipeline(figures, **kwargs) -> dict:
    pl_visualize_evaluations = pipeline(
        [
            node(
                func=shots_depths_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("_fw" in s)
                            and ("depth_" in s)
                            and ("time" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"shots_depths_viz",
            ),
            node(
                func=shots_qubits_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("_fw" in s)
                            and ("qubits_" in s)
                            and ("time" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"shots_depth_viz",
            ),
            node(
                func=depth_qubits_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("_fw" in s)
                            and ("shots_" in s)
                            and ("time" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"depth_qubits_viz",
            ),
            node(
                func=qubits_time_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                    "skip_frameworks": "params:skip_frameworks",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("shots_" in s)
                            and ("depth_" in s)
                            and ("time" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"qubits_time_viz",
            ),
            node(
                func=shots_time_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                    "skip_frameworks": "params:skip_frameworks",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("qubits_" in s)
                            and ("depth_" in s)
                            and ("time" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"shots_time_viz",
            ),
            node(
                func=depth_time_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                    "skip_frameworks": "params:skip_frameworks",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("shots_" in s)
                            and ("qubits_" in s)
                            and ("time" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"depth_time_viz",
            ),
            node(
                func=depth_measures_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("shots_" in s)
                            and ("qubits_" in s)
                            and ("measures" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"depth_measures_viz",
            ),
            node(
                func=shots_measures_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("qubits_" in s)
                            and ("depth_" in s)
                            and ("measures" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"shots_measures_viz",
            ),
            node(
                func=qubits_measures_viz,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={
                    **{
                        f: f
                        for f in filter(
                            lambda s: ("shots_" in s)
                            and ("depth_" in s)
                            and ("measures" in s),
                            figures,
                        )
                    },
                },
                tags=["static"],
                name=f"qubits_measures_viz",
            ),
        ],
        inputs={
            "evaluations_combined": "evaluations_combined",
        },
        outputs={
            **{f: f for f in figures},
        },
        namespace="visualization",
    )

    pl_print_tests = pipeline(
        [
            node(
                func=extract_tests,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                },
                outputs={},
                tags=["static"],
                name=f"extract_tests",
            )
        ],
        inputs={
            "evaluations_combined": "evaluations_combined",
        },
        outputs={},
        namespace="visualization",
    )

    pl_export_visualizations = pipeline(
        [
            node(
                func=export_selected,
                inputs={
                    "evaluations_combined": "evaluations_combined",
                    "additional_figures": "params:additional_figures",
                    "output_folder": "params:output_folder",
                    **{f: f for f in figures},
                },
                outputs={},
                tags=["dynamic"],
                name=f"export_selected",
            )
        ],
        inputs={
            "evaluations_combined": "evaluations_combined",
            **{f: f for f in figures},
        },
        namespace="visualization",
    )

    return {
        "pl_visualize_evaluations": pl_visualize_evaluations + pl_export_visualizations,
        "pl_print_tests": pl_print_tests,
    }
