from deepparallel.core.qubo_builders import qubo_for_paths
from deepparallel.core.optimizer_client import OptimizerClient
from deepparallel.core.decision_ledger import DecisionRecord
from deepparallel.core.resources import snapshot_resources
import uuid

def choose_paths(props: dict) -> dict:
    res = snapshot_resources()
    Q, meta = qubo_for_paths(props)
    cap = meta["capacity"]; cap["budget"] = res["gpu_memory_mb"]

    client = OptimizerClient()
    trace_id = str(uuid.uuid4())
    sol = client.solve(Q.tolist(), k_eq=3, capacity=cap)

    rec = DecisionRecord.from_run(trace_id, Q.tolist(),
                                  {"props": props, "capacity": cap, "k_eq": 3},
                                  sol)
    # emit rec.to_json() to your event bus here
    names = meta["names"]
    sel_idx = [i for i, v in enumerate(sol["solution"]) if v==1]
    return {"trace_id": trace_id, "selected": [names[i] for i in sel_idx], "ledger": rec.to_json()}
