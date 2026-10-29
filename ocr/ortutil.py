"""ONNX Runtime session options for CPU deployment.

By default every ONNX session spawns one thread per core. With several worker processes in a container that
oversubscribes the CPU. Set ORT_INTRA_THREADS (threads inside one operator) and ORT_INTER_THREADS (parallel
operators) to cap it; 0 / unset keeps the ONNX Runtime default. ORT_ALLOW_SPINNING=0 stops idle ONNX threads from
busy-waiting after a call, which otherwise takes CPU from the OpenCV work that follows (unset keeps the default).
"""

import os


def session_options():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_THREADS", "0"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_THREADS", "0"))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if os.environ.get("ORT_ALLOW_SPINNING") in ("0", "1"):
        so.add_session_config_entry("session.intra_op.allow_spinning", os.environ["ORT_ALLOW_SPINNING"])
    return so
