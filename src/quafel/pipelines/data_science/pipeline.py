"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from quafel.pipelines.data_science.nodes import (
    measure_execution_durations,
    combine_evaluations,
)


def create_pipeline(partitions, **kwargs) -> dict:
    pl_parallel_measure_execution_durations = pipeline(
        [
            *[
                node(
                    func=measure_execution_durations,
                    inputs={
                        "evaluations": "params:evaluations",
                        f"qasm_circuit_{i}": f"qasm_circuit_{i}",
                        f"n_shots_{i}": f"n_shots_{i}",
                        f"framework_id_{i}": f"framework_{i}",
                    },
                    outputs={
                        "execution_duration": f"execution_duration_{i}",
                        "execution_result": f"execution_result_{i}",
                    },
                    tags=["dynamic"],
                    name=f"measure_execution_durations_{i}",
                )
                for i in partitions
            ],
        ],
        inputs={
            **{f"qasm_circuit_{i}": f"qasm_circuit_{i}" for i in partitions},
            **{f"n_shots_{i}": f"n_shots_{i}" for i in partitions},
            **{f"framework_{i}": f"framework_{i}" for i in partitions},
        },
        outputs={
            **{
                f"execution_duration_{i}": f"execution_duration_{i}" for i in partitions
            },
            **{f"execution_result_{i}": f"execution_result_{i}" for i in partitions},
        },
        namespace="data_science",
    )

    pl_combine_evaluations = pipeline(
        [
            node(
                func=combine_evaluations,
                inputs={
                    "evaluation_partitions": "evaluation_partitions",
                    "execution_durations": "execution_durations",
                    "execution_results": "execution_results",
                    "measures": "measures",
                },
                outputs={
                    "evaluations_combined": "evaluations_combined",
                },
                tags=["static"],
                name=f"combine_evaluations",
            ),
        ],
        inputs={
            "evaluation_partitions": "evaluation_partitions",
            "execution_durations": "execution_durations",
            "execution_results": "execution_results",
            "measures": "measures",
        },
        outputs={
            "evaluations_combined": "evaluations_combined",
        },
        namespace="data_science",
    )

    return {
        "pl_parallel_measure_execution_durations": pl_parallel_measure_execution_durations,
        "pl_combine_evaluations": pl_combine_evaluations,
    }
