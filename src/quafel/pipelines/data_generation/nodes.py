from qiskit.circuit import ClassicalRegister, QuantumCircuit, CircuitInstruction
from qiskit.circuit import Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
import numpy as np

from typing import List
import pandas as pd


def log_circuit(qasm_circuit):
    return {"circuit_image": None}


def full_generate_random_qasm_circuits(evaluation_partitions, seed=100):
    qasm_circuits = {}
    n_shots = {}
    frameworks = {}

    for partition_id, partition_load_func in evaluation_partitions.items():
        partition_data = partition_load_func()
        partition_data.index = ["framework", "qubits", "depth", "shots"]

        framework = partition_data[partition_id]["framework"]
        qubits = int(partition_data[partition_id]["qubits"])
        depth = int(partition_data[partition_id]["depth"])
        shots = int(partition_data[partition_id]["shots"])

        qasm_circuits[f"qasm_circuit_{partition_id}"] = generate_random_qasm_circuit(
            qubits, depth, seed
        )["qasm_circuit"]
        n_shots[f"n_shots_{partition_id}"] = shots
        frameworks[f"framework_{partition_id}"] = framework

    return {
        **qasm_circuits,
        **n_shots,
        **frameworks,
    }


def part_generate_random_qasm_circuit(partition, seed=100):
    # TODO: improve this by accessing data by name
    framework = partition[partition.columns[0]][0]
    qubits = int(partition[partition.columns[0]][1])
    depth = int(partition[partition.columns[0]][2])
    shots = int(partition[partition.columns[0]][3])
    return {
        **generate_random_qasm_circuit(qubits, depth, seed),
        "n_shots": shots,
        "framework": framework,
    }


def _random_circuit(
    num_qubits,
    depth,
    max_operands=2,
    measure=True,
    conditional=False,
    reset=False,
    seed=None,
):
    """
    Code partly taken from Qiskit:
    qiskit/circuit/random/utils.py

    Args:
        num_qubits (_type_): _description_
        depth (_type_): _description_
        max_operands (int, optional): _description_. Defaults to 2.
        measure (bool, optional): _description_. Defaults to True.
        conditional (bool, optional): _description_. Defaults to False.
        reset (bool, optional): _description_. Defaults to False.
        seed (_type_, optional): _description_. Defaults to None.

    Raises:
        CircuitError: _description_

    Returns:
        _type_: _description_
    """
    if num_qubits == 0:
        return QuantumCircuit()
    if max_operands < 1 or max_operands > 4:
        raise CircuitError("max_operands must be between 1 and 4")
    max_operands = max_operands if num_qubits > max_operands else num_qubits

    gates_1q = [
        # (Gate class, number of qubits, number of parameters)
        (standard_gates.IGate, 1, 0),
        (standard_gates.XGate, 1, 0),
        (standard_gates.RZGate, 1, 1),
        (standard_gates.HGate, 1, 0),
        (standard_gates.RXGate, 1, 1),
        (standard_gates.RYGate, 1, 1),
        (standard_gates.SGate, 1, 0),
        (standard_gates.TGate, 1, 0),
        (standard_gates.U2Gate, 1, 2),
        (standard_gates.U3Gate, 1, 3),
        (standard_gates.YGate, 1, 0),
        (standard_gates.ZGate, 1, 0),
    ]
    if reset:
        gates_1q.append((Reset, 1, 0))
    gates_2q = [
        (standard_gates.CXGate, 2, 0),
        (standard_gates.CZGate, 2, 0),
        (standard_gates.SwapGate, 2, 0),
    ]
    gates_3q = [
        (standard_gates.CCXGate, 3, 0),
    ]

    # add gates to the overall set, depending on the number of operands
    gates = gates_1q.copy()
    if max_operands >= 2:
        gates.extend(gates_2q)
    if max_operands >= 3:
        gates.extend(gates_3q)
    gates = np.array(
        gates,
        dtype=[("class", object), ("num_qubits", np.int16), ("num_params", np.int32)],
    )

    # generate a numpy array, that we will use later, to fill "gaps"
    gates_1q = np.array(gates_1q, dtype=gates.dtype)

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomisation routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
        gate_specs = rng.choice(gates, size=len(qubits))
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int16)
        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        # This will, with favor multi qubit gates, but does not ensure >1 qubit gates
        # being used, especially with a low number of layers
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]
        slack = num_qubits - cumulative_qubits[max_index - 1]
        if slack:
            gate_specs = np.hstack((gate_specs, rng.choice(gates_1q, size=slack)))

        # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomises those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int16)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int16)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        if conditional and layer_number != 0:
            is_conditional = rng.random(size=len(gate_specs)) < 0.1
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0
            for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
                is_conditional,
            ):
                operation = gate(*parameters[p_start:p_end])
                if is_cond:
                    qc.measure(qc.qubits, cr)
                    # The condition values are required to be bigints, not Numpy's fixed-width type.
                    operation.condition = (cr, int(condition_values[c_ptr]))
                    c_ptr += 1
                qc._append(
                    CircuitInstruction(
                        operation=operation, qubits=qubits[q_start:q_end]
                    )
                )
        else:
            for gate, q_start, q_end, p_start, p_end in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
            ):
                operation = gate(*parameters[p_start:p_end])
                qc._append(
                    CircuitInstruction(
                        operation=operation, qubits=qubits[q_start:q_end]
                    )
                )

    if measure:
        qc.measure(qc.qubits, cr)

    return qc


def generate_random_qasm_circuit(qubits: int, depth: int, seed: int):
    qc = _random_circuit(qubits, depth, max_operands=2, measure=True, seed=seed)
    return {"qasm_circuit": qc.qasm()}


def generate_evaluation_matrix(
    min_qubits: int,
    max_qubits: int,
    qubits_increment: int,
    qubits_type: int,
    min_depth: int,
    max_depth: int,
    depth_increment: int,
    depth_type: int,
    min_shots: int,
    max_shots: int,
    shots_increment: int,
    shots_type: int,
    frameworks: List[str],
):
    def generate_ticks(min_t, max_t, inc_t, type_t="linear"):
        if type_t == "linear":
            ticks = [i for i in range(min_t, max_t + inc_t, inc_t)]
        elif "exp" in type_t:
            base = int(type_t.split("exp")[1])
            ticks = [base**i for i in range(min_t, max_t + inc_t, inc_t)]
        else:
            raise ValueError("Unknown base specified and type is not linear")

        return ticks

    qubits = generate_ticks(min_qubits, max_qubits, qubits_increment, qubits_type)
    depths = generate_ticks(min_depth, max_depth, depth_increment, depth_type)
    shots = generate_ticks(min_shots, max_shots, shots_increment, shots_type)

    frameworks = frameworks

    return {
        "evaluation_matrix": {
            "frameworks": frameworks,
            "qubits": qubits,
            "depths": depths,
            "shots": shots,
        }
    }


def generate_evaluation_partitions(evaluation_matrix, skip_combinations):
    partitions = {}
    idx = 0
    for f in evaluation_matrix["frameworks"]:
        if "qubits" in skip_combinations:
            q = max(evaluation_matrix["qubits"])
            for d in evaluation_matrix["depths"]:
                for s in evaluation_matrix["shots"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        elif "depth" in skip_combinations:
            d = max(evaluation_matrix["depth"])
            for q in evaluation_matrix["qubtis"]:
                for s in evaluation_matrix["shots"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        elif "shots" in skip_combinations:
            s = max(evaluation_matrix["shots"])
            for d in evaluation_matrix["depths"]:
                for q in evaluation_matrix["qubits"]:
                    partitions[f"{idx}"] = {
                        "framework": f,
                        "qubits": q,
                        "depth": d,
                        "shots": s,
                    }
                    idx += 1
        else:
            for q in evaluation_matrix["qubits"]:
                for d in evaluation_matrix["depths"]:
                    for s in evaluation_matrix["shots"]:
                        partitions[f"{idx}"] = {
                            "framework": f,
                            "qubits": q,
                            "depth": d,
                            "shots": s,
                        }
                        idx += 1

    eval_partitions = pd.DataFrame(partitions)

    return {"evaluation_partitions": eval_partitions}
