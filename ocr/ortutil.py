"""ONNX Runtime session options for CPU deployment.

By default every ONNX session spawns one thread per core. With several worker processes in a container that
oversubscribes the CPU. Set ORT_INTRA_THREADS (threads inside one operator) and ORT_INTER_THREADS (parallel
operators) to cap it; 0 / unset keeps the ONNX Runtime default.
"""

import os


def session_options():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_INTRA_THREADS", "0"))
    so.inter_op_num_threads = int(os.environ.get("ORT_INTER_THREADS", "0"))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return so
