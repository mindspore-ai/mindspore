# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""GraphKernel model builder"""

import copy
from . import op_infer
from .model import Tensor, Value, Operator, Graph, AlignShape, AddControlBuddy


class GraphBuilder:
    """Graph builder"""
    class GraphWrapper:
        """Graph wrapper"""

        def __init__(self, name):
            self.graph = Graph(name, [])

        def set_input(self, *para):
            for t in para:
                t.para_type = Tensor.PARA_INPUT
                self.graph.inputs.append(t)

        def set_output(self, *para):
            for t in para:
                t.para_type = Tensor.PARA_OUTPUT
                self.graph.outputs.append(t)

    def __init__(self):
        self.graphs = []
        self.current = None
        self.name_id = 0

    def _alloc_tensor_name(self):
        tid = self.name_id
        self.name_id += 1
        return "t%d" % (tid)

    def graph_scope(self, name):
        """The graph scope to be processed"""
        class GraphScope:
            def __init__(self, gb):
                self.gb = gb

            def __enter__(self):
                return self.gb.current

            def __exit__(self, ptype, value, trace):
                self.gb.graphs.append(self.gb.current.graph)
                self.gb.current = None

        assert self.current is None
        self.current = self.GraphWrapper(name)
        return GraphScope(self)

    def tensor(self, shape, dtype, data_format="DefaultFormat", name=None, para_type=Tensor.PARA_NONE):
        """Create a new Tensor"""
        if name in (None, ''):
            name = self._alloc_tensor_name()
        if not shape:
            shape = [1]
        return Tensor(name, shape, dtype, data_format, para_type=para_type)

    def value(self, dtype, value, name=None):
        """Create a new Value"""
        if name in (None, ''):
            name = self._alloc_tensor_name()

        v = Value(name, dtype, value)
        return v

    def op(self, prim, output, inputs, attrs=None):
        """Insert an operator into graph"""
        if attrs is None:
            attrs = {}
        if isinstance(inputs, Tensor):
            inputs = [inputs]
        tensor_inputs = [t for t in inputs if isinstance(t, Tensor)]
        node = Operator(prim, tensor_inputs, output, attrs)
        node.all_inputs = inputs
        self.current.graph.add(node)

    def emit(self, prim, inputs, name=None, attrs=None):
        """Emit a new operation"""
        if attrs is None:
            attrs = {}
        if isinstance(inputs, (Tensor, Value)):
            inputs = [inputs]
        tensor_inputs = [t for t in inputs if isinstance(t, (Tensor, Value))]
        out_shape, out_dtype, out_format = op_infer.infer(prim, tensor_inputs, attrs)
        output = self.tensor(out_shape, out_dtype, out_format, name)
        self.op(prim, output, inputs, attrs)
        return output

    def get(self):
        return self.graphs


class CompositeGraph:
    """Composite Graph"""

    def __init__(self):
        self.graph = None
        self.desc = None
        self.tensors = {}  # name : Tensor

    def refine(self):
        """Refine Graph"""
        AlignShape().visit_graph(self.graph)
        AddControlBuddy().visit_graph(self.graph)

    def load(self, desc):
        """Load Graph from json"""
        def _attr_of(op, inputs, output):
            def _get_axis_while_none(input_shape, output_shape):
                red_axis = []
                if len(output_shape) == len(input_shape):
                    for i, s in enumerate(output_shape):
                        if s == 1 and input_shape[i] > 1:
                            red_axis.append(i)
                else:
                    red_axis = list(range(len(output_shape)))
                return red_axis

            attr = {}
            if op['name'] not in ('ReduceSum', 'ReduceMax', 'ReduceMin'):
                return attr
            for a in op['attr']:
                if a['name'] == 'axis':
                    red_axis, dim_size = [], len(inputs[0].shape)
                    if not a['value']:
                        red_axis = _get_axis_while_none(inputs[0].shape, output.shape)
                    else:
                        if isinstance(a['value'], int):
                            a['value'] = [a['value']]
                        for i in a['value']:
                            red_axis.append(i if i >= 0 else dim_size + i)
                    attr['reduce_axis'] = red_axis
                    break
            return attr

        builder = GraphBuilder()
        with builder.graph_scope(desc['op']):
            for in_desc in desc['input_desc'] if desc['input_desc'] is not None else []:
                name, shape, dtype, data_format = in_desc[0]['tensor_name'], in_desc[
                    0]['shape'], in_desc[0]['data_type'], in_desc[0]['format']
                self.tensors[name] = builder.tensor(
                    shape, dtype, data_format, name=name, para_type=Tensor.PARA_INPUT)
            for out_desc in desc['output_desc']:
                name, shape, dtype, data_format = out_desc['tensor_name'], out_desc[
                    'shape'], out_desc['data_type'], out_desc['format']
                self.tensors[name] = builder.tensor(
                    shape, dtype, data_format, name=name, para_type=Tensor.PARA_OUTPUT)
            cur_fusion = None
            for op in desc['op_desc']:
                inputs = [self.tensors[d['tensor_name']] for x in op['input_desc'] for d in x if 'value' not in d]
                out_desc = op['output_desc']
                name, shape, dtype, data_format = out_desc[0]['tensor_name'], out_desc[
                    0]['shape'], out_desc[0]['data_type'], out_desc[0]['format']
                if op['name'] == 'InplaceAssign':
                    inputs[0].add_buddy(inputs[1])
                    inputs[1].para_type = Tensor.PARA_OUTPUT
                    output = inputs[2]
                    self.tensors[name] = output
                else:
                    output = self.tensors.get(name, None)
                    if not output:
                        output = builder.tensor(
                            shape, dtype, data_format, name=name)
                        self.tensors[name] = output
                    builder.op(op['name'], output, inputs,
                               attrs=_attr_of(op, inputs, output))
                if 'fusion' in op:
                    if cur_fusion is None:
                        cur_fusion = output
                    else:
                        cur_fusion.add_buddy(output)
                        if op['fusion'].endswith('_end'):
                            cur_fusion = None
        self.graph = builder.get()[0]
        self.desc = desc

    def add_stitch_info(self, subgraph, desc):
        if subgraph.stitch_info and subgraph.stitch_info.stitch_ops:
            buffer_stitch = {'stitch_op': list(subgraph.stitch_info.stitch_ops)}
            if subgraph.stitch_info.stitch_atomic_ops:
                buffer_stitch['stitch_atomic_op'] = list(subgraph.stitch_info.stitch_atomic_ops)
            desc['buffer_stitch'] = buffer_stitch
        return desc

    def dump(self, subgraph):
        """Dump Graph to json"""
        desc = {}
        inputs, outputs = subgraph.deduce_parameters()
        graph_ops = set(subgraph.ops)
        inplace_assign = {}  # y_name, output_name
        inplace_assign_z = None
        for op in self.desc['op_desc']:
            if op['name'] == 'InplaceAssign':
                inplace_assign[op['input_desc'][1][0]['tensor_name']] = op['output_desc'][0]['tensor_name']
        if inplace_assign:
            for t in outputs:
                if t.name not in inplace_assign:
                    inplace_assign_z = t
        for key in self.desc:
            if key == 'input_desc':
                desc[key] = [
                    [{'data_type': t.dtype, 'shape': t.shape, 'tensor_name': t.name}] for t in inputs]
            elif key == 'output_desc':
                out_desc = []
                for t in outputs:
                    if t.name in inplace_assign:
                        z = inplace_assign_z if inplace_assign_z is not None else self.tensors[t.name]
                        out_desc.append(
                            {'data_type': z.dtype, 'shape': z.shape, 'tensor_name': inplace_assign[t.name]})
                    else:
                        out_desc.append(
                            {'data_type': t.dtype, 'shape': t.shape, 'tensor_name': t.name})
                desc[key] = out_desc
            elif key == 'op_desc':
                op_desc = []
                for d in self.desc[key]:
                    if d['name'] == 'InplaceAssign':
                        y = d['input_desc'][1][0]['tensor_name']
                        if self.tensors[y].op in graph_ops:
                            z, fake = (inplace_assign_z, False) if inplace_assign_z is not None else (
                                self.tensors[y], True)
                            inplace_desc = copy.deepcopy(d)
                            inplace_desc['attr'] = {'name': 'fake_output', 'value': fake}
                            z_desc, out_desc = inplace_desc['input_desc'][2][0], inplace_desc['output_desc'][0]
                            z_desc['shape'] = z.shape
                            z_desc['data_type'] = z.dtype
                            z_desc['tensor_name'] = z.name
                            out_desc['shape'] = z.shape
                            out_desc['data_type'] = z.dtype
                            op_desc.append(inplace_desc)
                    else:
                        op = self.tensors[d['output_desc'][0]['tensor_name']].op
                        if op in graph_ops:
                            op_desc.append(d)
                desc[key] = op_desc
            elif key == 'op':
                desc[key] = subgraph.name
            else:
                desc[key] = self.desc[key]

        desc = self.add_stitch_info(subgraph, desc)
        return desc


def load_composite(desc):
    """Load composite kernel"""
    composite = CompositeGraph()
    composite.load(desc)
    composite.refine()
    return composite
