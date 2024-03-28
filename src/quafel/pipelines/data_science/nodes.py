from . import frameworks as fw
import time
import pandas as pd
from typing import Dict
import logging
import traceback
import re

log = logging.getLogger(__name__)


def measure_execution_durations(
    evaluations: int,
    **kwargs,
):
    ident = int(re.findall(r"\d+", list(kwargs.keys())[0])[-1])
    for key, value in kwargs.items():
        if key.startswith("framework_id"):
            framework_id = value
        elif key.startswith("qasm_circuit"):
            qasm_circuit = value
        elif key.startswith("n_shots"):
            n_shots = value

    try:
        framework = getattr(fw, framework_id)
    except AttributeError:
        raise AttributeError(
            f"Framework identifier does not match one of the existing frameworks. Existing frameworks are {fw}"
        )

    framework_instance = framework(qasm_circuit=qasm_circuit, n_shots=int(n_shots))

    execution_perf_durations = []
    execution_proc_durations = []
    execution_results = []

    for e in range(evaluations):
        try:
            start_perf = time.perf_counter()
            start_proc = time.process_time()
            framework_instance.execute()
            finish_perf = time.perf_counter()
            finish_proc = time.process_time()
            execution_perf_durations.append(finish_perf - start_perf)
            execution_proc_durations.append(finish_proc - start_proc)
        except Exception:
            log.error(
                f"Error executing framework {framework_id} for experiment id {ident}: \
                    Execution failed in evaluation {e}: \
                        {traceback.format_exc()}"
            )
            # mark the whole set invalid
            execution_results = [0 for _ in range(evaluations)]
            break

        execution_results.append(framework_instance.get_result())

    return {
        "execution_duration": pd.DataFrame(
            {
                f"perf_{ident}": execution_perf_durations,
                f"proc_{ident}": execution_proc_durations,
            }
        ),
        "execution_result": pd.DataFrame({ident: execution_results}),
    }


def combine_evaluations(
    evaluation_partitions: Dict,
    execution_durations: Dict,
    execution_results: Dict,
    measures: Dict,
):
    combine_all = pd.DataFrame()

    for (
        (partition_id, partition_load_func),
        (duration_id, duration_load_func),
        (result_id, result_load_func),
        (measure_id, measure_load_func),
    ) in zip(
        evaluation_partitions.items(),
        execution_durations.items(),
        execution_results.items(),
        measures.items(),
    ):
        assert (
            (partition_id == duration_id)
            and (partition_id == result_id)
            and (partition_id == measure_id)
        ), "Partition identifiers do not match duration and result identifiers."
        partition_data = partition_load_func()
        duration_data = duration_load_func()
        result_data = result_load_func()
        measure_data = measure_load_func()

        # TODO: unify somehow with the generation part
        partition_data.index = ["framework", "qubits", "depth", "shots"]

        duration_data_proc = duration_data.filter(regex="proc")
        duration_data_proc.index = [
            f"duration_proc_{i}" for i in range(len(duration_data))
        ]
        duration_data_proc.columns = [partition_id]

        duration_data_perf = duration_data.filter(regex="perf")
        duration_data_perf.index = [
            f"duration_perf_{i}" for i in range(len(duration_data))
        ]
        duration_data_perf.columns = [partition_id]

        duration_data = pd.concat([duration_data_perf, duration_data_proc])

        result_data.index = [f"result_{i}" for i in range(len(result_data))]

        measure_data = measure_data.T
        measure_data.columns = [partition_id]

        combined_partition_duration = pd.concat(
            [partition_data, duration_data, result_data, measure_data],
            ignore_index=False,
            axis=0,
        )

        combine_all = pd.concat(
            [combine_all, combined_partition_duration],
            ignore_index=False,
            axis=1,
        )

    combine_all = combine_all.transpose()
    # combine_all = combine_all.sort_values(by=[0, 1, 3]) # sort by framework and shots

    return {"evaluations_combined": combine_all}
