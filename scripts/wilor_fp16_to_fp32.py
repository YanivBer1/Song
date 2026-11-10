"""Convert an all-float16 ONNX graph to float32 (CPU onnxruntime runs fp16 slowly / not at all)."""
import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


def conv_tensor(t):
    if t.data_type == F16:
        a = numpy_helper.to_array(t).astype(np.float32)
        t.CopyFrom(numpy_helper.from_array(a, t.name))
        return 1
    return 0


def fix_graph(g):
    n = 0
    for t in g.initializer:
        n += conv_tensor(t)
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == F16:
            tt.elem_type = F32
    for node in g.node:
        for a in node.attribute:
            if a.name == "to" and node.op_type == "Cast" and a.i == F16:
                a.i = F32
            if a.type == onnx.AttributeProto.TENSOR:
                n += conv_tensor(a.t)
            if a.type == onnx.AttributeProto.GRAPH:
                n += fix_graph(a.g)
    return n


print("converted tensors:", fix_graph(m.graph))
ops = {}
for node in m.graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1
print(ops)
onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=dst.split("/")[-1] + ".data")
