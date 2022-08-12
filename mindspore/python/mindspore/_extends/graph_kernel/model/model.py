# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
"""GraphKernel cost model"""
from mindspore import log as logger


class Utils:
    """Model utils"""

    def __init__(self):
        pass

    @staticmethod
    def get_attr_type(attr):
        """Get attr type"""
        if isinstance(attr, bool):
            return 'bool'
        if isinstance(attr, str):
            return 'str'
        if isinstance(attr, int):
            return 'int'
        if isinstance(attr, float):
            return 'float'
        if isinstance(attr, (list, tuple)):
            if not attr:
                raise ValueError("attr is invalid: the length of attr is 0")
            if isinstance(attr[0], int):
                return 'listInt'
            if isinstance(attr[0], str):
                return 'listStr'
        raise ValueError("attr {} type {} is not in supported list ['bool', 'str', 'int', 'float', 'int' list, "
                         "'str' list]".format(attr, type(attr)))


class DataFormat:
    """DataFormat"""
    DEFAULT = "DefaultFormat"
    NC1KHKWHWC0 = "NC1KHKWHWC0"
    ND = "ND"
    NCHW = "NCHW"
    NHWC = "NHWC"
    HWCN = "HWCN"
    NC1HWC0 = "NC1HWC0"
    FRAC_Z = "FRACTAL_Z"
    FRAC_NZ = "FRACTAL_NZ"
    C1HWNCOC0 = "C1HWNCoC0"
    NC1HWC0_C04 = "NC1HWC0_C04"
    FRACTAL_Z_C04 = "FRACTAL_Z_C04"
    NDHWC = "NDHWC"

    def __init__(self):
        pass


class DataType:
    """Data Type"""
    FLOAT = "float"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT = "int"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT = "uint"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOL = "bool"

    def __init__(self):
        pass


class PrimLib:
    """Prim lib"""

    UNKNOWN = 0
    RESHAPE = 1
    ELEMWISE = 2
    BROADCAST = 3
    REDUCE = 4
    OPAQUE = 5

    def __init__(self):
        pass

    class Prim:
        """Prim"""

        def __init__(self, iter_type, calibrate=1, relation_func=None):
            self.iter_type = iter_type
            self.calibrate = calibrate
            self.relation_func = relation_func
            if relation_func is None:
                self.relation_func = lambda *x: self.default_relation_func[iter_type](self, *x)

        def default_reshape_relation(self, op, input_idx):
            """Process reshape relation"""
            axis_relation, elem_relation = self.unknown_relation(op, input_idx)
            elem_relation = [PrimLib.RESHAPE] * len(elem_relation)
            return axis_relation, elem_relation

        def default_elemwise_broadcast_relation(self, op, input_idx):
            """Process elemwise and broadcast relation"""
            out_shape = op.output.shape
            in_shape = op.inputs[input_idx].shape
            if len(out_shape) < len(in_shape):
                raise ValueError("For '{}', the input/output size is abnormal, as the length of output shape{} is less "
                                 "than the length of input shape{}".format(op.prim, out_shape, in_shape))
            axis_relation, elem_relation = [], []
            delta = len(out_shape) - len(in_shape)
            if delta > 0:
                for _ in range(0, delta):
                    axis_relation.append(None)
                    elem_relation.append(None)
            for i, _ in enumerate(in_shape):
                axis_relation.append(i)
                elem_relation.append(
                    PrimLib.ELEMWISE if out_shape[i + delta] == in_shape[i] else PrimLib.BROADCAST)
            return axis_relation, elem_relation

        def default_reduce_relation(self, op, input_idx):
            """Process reduce relation"""
            axis_relation, elem_relation = self.default_elemwise_broadcast_relation(op, input_idx)
            for i in op.attrs['reduce_axis']:
                elem_relation[i] = PrimLib.REDUCE
            return axis_relation, elem_relation

        def unknown_relation(self, op, input_idx):
            """Process unknown relation"""
            out_shape = op.output.shape
            in_shape = op.inputs[input_idx].shape
            all_relation = list(range(len(in_shape)))
            axis_relation = [all_relation for i in range(0, len(out_shape))]
            elem_relation = [PrimLib.UNKNOWN for i in range(0, len(out_shape))]
            return axis_relation, elem_relation

        default_relation_func = [
            unknown_relation,
            default_reshape_relation,
            default_elemwise_broadcast_relation,
            default_elemwise_broadcast_relation,
            default_reduce_relation,
            unknown_relation,
        ]

    primtives = {
        'Add': Prim(ELEMWISE),
        'Abs': Prim(ELEMWISE),
        'Neg': Prim(ELEMWISE),
        'Mul': Prim(ELEMWISE),
        'Sub': Prim(ELEMWISE),
        'Log': Prim(ELEMWISE),
        'IsNan': Prim(ELEMWISE),
        'IsInf': Prim(ELEMWISE),
        'IsFinite': Prim(ELEMWISE),
        'Exp': Prim(ELEMWISE),
        'Rsqrt': Prim(ELEMWISE),
        'Sqrt': Prim(ELEMWISE),
        'Div': Prim(ELEMWISE),
        'FloorDiv': Prim(ELEMWISE),
        'RealDiv': Prim(ELEMWISE),
        'Mod': Prim(ELEMWISE),
        'Floor': Prim(ELEMWISE),
        'FloorMod': Prim(ELEMWISE),
        'Erf': Prim(ELEMWISE),
        'Erfc': Prim(ELEMWISE),
        'Cast': Prim(ELEMWISE),
        'Pow': Prim(ELEMWISE),
        'Minimum': Prim(ELEMWISE),
        'Maximum': Prim(ELEMWISE),
        'Reciprocal': Prim(ELEMWISE),
        'Equal': Prim(ELEMWISE),
        'NotEqual': Prim(ELEMWISE),
        'Greater': Prim(ELEMWISE),
        'GreaterEqual': Prim(ELEMWISE),
        'Less': Prim(ELEMWISE),
        'LessEqual': Prim(ELEMWISE),
        'LogicalNot': Prim(ELEMWISE),
        'LogicalAnd': Prim(ELEMWISE),
        'LogicalOr': Prim(ELEMWISE),
        'Select': Prim(ELEMWISE, 8),
        'ReduceSum': Prim(REDUCE),
        'ReduceMax': Prim(REDUCE),
        'ReduceMin': Prim(REDUCE),
        'Argmax': Prim(REDUCE),
        'Argmin': Prim(REDUCE),
        'Assign': Prim(ELEMWISE),
        'Sign': Prim(ELEMWISE),
        'Sin': Prim(ELEMWISE),
        'Cos': Prim(ELEMWISE),
        'Asin': Prim(ELEMWISE),
        'ACos': Prim(ELEMWISE),
        'Tanh': Prim(ELEMWISE),
        'Asinh': Prim(ELEMWISE),
        'Acosh': Prim(ELEMWISE),
        'Reshape': Prim(RESHAPE),
        'Transpose': Prim(OPAQUE),
        'CumSum': Prim(OPAQUE),
        'Tile': Prim(BROADCAST),
        'BroadcastTo': Prim(BROADCAST),
        'StridedSlice': Prim(OPAQUE),
        'MatMul': Prim(OPAQUE),
        'TransData': Prim(OPAQUE),
        'BatchMatMul': Prim(OPAQUE),
        'UnPadAkg': Prim(OPAQUE),
        'PadAkg': Prim(OPAQUE),
        'Conv2D': Prim(OPAQUE),
        'CReal': Prim(ELEMWISE),
        'CImag': Prim(ELEMWISE),
        'Complex': Prim(ELEMWISE),
        'Atan': Prim(ELEMWISE),
        'Atan2': Prim(ELEMWISE),
        'Expm1': Prim(ELEMWISE),
        'TensorScatterAdd': Prim(OPAQUE),
        'Gather': Prim(OPAQUE),
        'GatherNd': Prim(OPAQUE),
        'CSRReduceSum': Prim(OPAQUE),
        'CSRMul': Prim(ELEMWISE),
        'CSRDiv': Prim(ELEMWISE),
        'CSRGather': Prim(OPAQUE),
        'UnsortedSegmentSum': Prim(OPAQUE),
        'StandardNormal': Prim(OPAQUE),
        'OneHot': Prim(OPAQUE),
        'ElemAny': Prim(OPAQUE),
        'Custom': Prim(OPAQUE),
    }

    default_primtive = Prim(UNKNOWN)

    @classmethod
    def get_prim(cls, op):
        """Get op primtive"""
        prim = cls.primtives.get(op.prim, None)
        if prim is None:
            logger.warning("primtive is not registered: {}".format(op.prim))
            prim = cls.default_primtive
        return prim

    @classmethod
    def input_relation(cls, op, input_idx):
        """Get op's input_relation according to input_idx"""
        return cls.get_prim(op).relation_func(op, input_idx)

    @classmethod
    def iter_type(cls, op):
        """Get op's iter type"""
        return cls.get_prim(op).iter_type

    @classmethod
    def is_reduce(cls, op):
        """Check whether op's iter type is reduce"""
        return cls.get_prim(op).iter_type == cls.REDUCE

    @classmethod
    def calibrate_iter_size(cls, op, iter_size):
        """Get calibrate_iter_size"""
        return cls.get_prim(op).calibrate * iter_size

    @classmethod
    def dtype_bytes(cls, dtype):
        """Get dtype bytes"""
        bits, unit = 1, 1
        for i in range(len(dtype) - 1, 0, -1):
            if dtype[i].isdecimal():
                bits += int(dtype[i]) * unit
                unit *= 10
            else:
                break
        return bits // 8

    @classmethod
    def inplace_reuse(cls, op, input_idx, start_axis=0):
        """Check whether op is inplace reuse"""
        if cls.dtype_bytes(op.output.dtype) > cls.dtype_bytes(op.inputs[input_idx].dtype):
            return False
        _, elem_relation = cls.get_prim(op).relation_func(op, input_idx)
        for i in range(start_axis, len(elem_relation)):
            if elem_relation[i] != cls.ELEMWISE:
                return False
        return True


class Tensor:
    """Tensor"""

    PARA_NONE = 0
    PARA_INPUT = 1
    PARA_OUTPUT = 2

    class Buddy:
        """Buddy"""

        def __init__(self, leader):
            self.members = [leader]

    def __init__(self, name, shape, dtype, data_format=DataFormat.DEFAULT, para_type=0):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data_format = data_format
        self.para_type = para_type
        self.op = None
        self.to_ops = []
        self.buddy = None

    def __str__(self):
        return self.name + str(list(self.shape))

    def __repr__(self):
        return "%s.%s%s" % (self.name, self.dtype, str(list(self.shape)))

    def get_size(self):
        """Get size"""
        size = PrimLib.dtype_bytes(self.dtype)
        for i in self.shape:
            size *= i
        return size

    def add_buddy(self, tensor):
        """Add buddy"""
        if self.buddy is None:
            self.buddy = self.Buddy(self)
        self.buddy.members.append(tensor)
        tensor.buddy = self.buddy


class Value:
    """Value"""

    def __init__(self, name, dtype, value, data_format=DataFormat.DEFAULT):
        self.name = name
        self.shape = [1]
        self.dtype = dtype
        self.value = value
        self.data_format = data_format

    def __str__(self):
        return self.name + str(list(self.shape))

    def __repr__(self):
        return "%s.%s%s" % (self.name, self.dtype, str(list(self.shape)))

    @staticmethod
    def get_size():
        """Get size"""
        return 1


class Operator:
    """Operator"""

    def __init__(self, primtive, inputs, output, attrs):
        self.prim = primtive
        self.inputs = inputs
        self.output = output
        self.attrs = attrs
        for t in inputs:
            t.to_ops.append(self)
        if output.op is None:
            output.op = self
        self.all_inputs = []  # include Tensor inputs and Value inputs.

    def __str__(self):
        args = ', '.join((str(t) for t in self.all_inputs))
        expr = "%s = %s.%s(%s) id:%s" % (
            str(self.output), self.prim, self.output.dtype, args, id(self))
        return expr if not self.attrs else '%s // %s' % (expr, str(self.attrs))

    def __repr__(self):
        return str(self)


class Graph:
    """Graph"""

    def __init__(self, name, ops, stitch_info=None, recompute_ops=None):
        self.name = name
        self.ops = ops  # in topo order, can not use set
        self.inputs = []
        self.outputs = []
        self.stitch_info = stitch_info
        self.recompute_ops = recompute_ops
        self.processor = ""

    def set_processor(self, processor):
        """Set processor"""
        self.processor = processor

    def add(self, ops):
        """Add ops"""
        if isinstance(ops, Operator):
            self.ops.append(ops)
        else:
            self.ops.extend(ops)

    def extract_subgraph(self, graph_name, tensor_names, difference=False):
        """Extract subgraph from this graph"""
        graph = Graph(graph_name, [])
        outputs = set(tensor_names)
        if difference:
            for op in self.ops:
                if op.output.name not in outputs:
                    graph.add(op)
        else:
            for op in self.ops:
                if op.output.name in outputs:
                    graph.add(op)
                    outputs.remove(op.output.name)
            for name in outputs:
                raise ValueError("Invalid input tensor : {}, can not find it in graph".format(name))
        return graph

    def deduce_parameters(self):
        """Deduce parameters"""
        inputs, outputs = [], []
        for op in self.ops:
            for t in op.inputs:
                if t not in inputs and t.op not in self.ops:
                    inputs.append(t)
            if op.output in outputs:
                continue
            if op.output.para_type == Tensor.PARA_OUTPUT or not op.output.to_ops:
                outputs.append(op.output)
                continue
            if any((succ not in self.ops for succ in op.output.to_ops)):
                outputs.append(op.output)
        if self.inputs:
            inputs = self.inputs

        if self.outputs:
            outputs = self.outputs
        return inputs, outputs

    def __str__(self):
        inputs, outputs = self.deduce_parameters()
        para_str = ', '.join((repr(t) for t in inputs))
        out_str = ', '.join((repr(t) for t in outputs))
        lines = []
        lines.append("%s(%s) -> %s {" % (self.name, para_str, out_str))
        if self.stitch_info:
            if self.stitch_info.stitch_ops:
                lines.append('  stitch -> ' + str(self.stitch_info.stitch_ops))
            if self.stitch_info.stitch_atomic_ops:
                lines.append('  stitch_atomic_ops-> ' + str(self.stitch_info.stitch_atomic_ops))

        for op in self.ops:
            lines.append('  ' + str(op))
        lines.append('}')
        return '\n'.join(lines)

    def __repr__(self):
        return str(self)

    def dump(self):
        """Dump Graph to json"""
        attr_name = {'reduce_axis': 'axis'}
        inputs, outputs = self.deduce_parameters()
        input_desc, output_desc, op_desc = [], [], []
        for t in inputs:
            input_desc.append([{'data_type': t.dtype, 'shape': t.shape,
                                'tensor_name': t.name, 'format': t.data_format}])
        for t in outputs:
            output_desc.append({'data_type': t.dtype, 'shape': t.shape,
                                'tensor_name': t.name, 'format': t.data_format})
        for op in self.ops:
            attrs, in_desc = [], []
            for a in op.attrs:
                name = attr_name.get(a, a)
                attrs.append(
                    {'name': name, 'value': op.attrs[a], 'data_type': Utils.get_attr_type(op.attrs[a])})
            for t in op.all_inputs:
                if isinstance(t, Tensor):
                    in_desc.append([{'data_type': t.dtype, 'name': '', 'shape': t.shape,
                                     'tensor_name': t.name, 'format': t.data_format}])
                else:
                    in_desc.append([{'data_type': t.dtype, 'value': t.value, 'name': '', 'shape': t.shape,
                                     'tensor_name': t.name, 'format': t.data_format}])
            out_desc = [{'data_type': op.output.dtype, 'name': '', 'shape': op.output.shape,
                         'tensor_name': op.output.name, 'format': op.output.data_format}]
            op_desc.append({'attr': attrs, 'impl_path': '',
                            'input_desc': in_desc, 'name': op.prim, 'output_desc': out_desc})

        graph_desc = {'composite': True, 'composite_graph': '', 'id': 0,
                      'input_desc': input_desc, 'op': self.name, 'op_desc': op_desc, 'output_desc': output_desc,
                      'platform': 'AKG', 'process': self.processor}

        if self.stitch_info and self.stitch_info.stitch_ops:
            buffer_stitch = {'stitch_op': list(self.stitch_info.stitch_ops)}
            if self.stitch_info.stitch_atomic_ops:
                buffer_stitch['stitch_atomic_op'] = list(self.stitch_info.stitch_atomic_ops)
            graph_desc['buffer_stitch'] = buffer_stitch

        return graph_desc


class GraphVisitor:
    """Graph visitor"""

    def __init__(self, forward=True):
        self.forward = forward

    def visit_graph(self, graph):
        """Visit graph"""
        if self.forward:
            for op in graph.ops:
                self.visit(op)
        else:
            for i in range(len(graph.ops)-1, -1, -1):
                self.visit(graph.ops[i])


class AlignShape(GraphVisitor):
    """Align shape"""

    def __init__(self):
        super(AlignShape, self).__init__()

    @staticmethod
    def visit(op):
        """Visit op node"""
        prim = PrimLib.get_prim(op)
        if prim.iter_type in (PrimLib.ELEMWISE, PrimLib.BROADCAST, PrimLib.REDUCE):
            out_dim = len(op.output.shape)
            align_dim = out_dim
            for t in op.inputs:
                if len(t.shape) > align_dim:
                    align_dim = len(t.shape)
            if align_dim > out_dim:
                op.output.shape = [1] * (align_dim - out_dim) + op.output.shape


class GraphKernelUnsupportedException(Exception):
    """"GraphKernel Unsupported Exception"""

    def __init__(self, message):
        super(GraphKernelUnsupportedException, self).__init__()
        self.message = message
