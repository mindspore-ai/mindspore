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
"""GraphKernel model builder"""
from .model import Tensor, Value, Operator, Graph, AlignShape


class GraphBuilder:
    """Graph builder"""
    class GraphWrapper:
        """Graph wrapper"""

        def __init__(self, name):
            self.graph = Graph(name, [])

        def set_input(self, *para):
            """set input to graph inputs"""
            for t in para:
                t.para_type = Tensor.PARA_INPUT
                self.graph.inputs.append(t)

        def set_output(self, *para):
            """set output to graph inputs"""
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
            """Graph Scope"""

            def __init__(self, gb):
                self.gb = gb

            def __enter__(self):
                return self.gb.current

            def __exit__(self, ptype, value, trace):
                self.gb.graphs.append(self.gb.current.graph)
                self.gb.current = None

        if self.current is not None:
            raise ValueError("self.current is not None!")
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

    def get(self):
        """Get graphs"""
        return self.graphs


class CompositeGraph:
    """Composite Graph"""

    def __init__(self):
        self.graph = None
        self.desc = None
        self.tensors = {}  # name : Tensor

    @staticmethod
    def add_stitch_info(subgraph, desc):
        """add stitch info to desc"""
        if subgraph.stitch_info and subgraph.stitch_info.stitch_ops:
            buffer_stitch = {'stitch_op': list(subgraph.stitch_info.stitch_ops)}
            if subgraph.stitch_info.stitch_atomic_ops:
                buffer_stitch['stitch_atomic_op'] = list(subgraph.stitch_info.stitch_atomic_ops)
            desc['buffer_stitch'] = buffer_stitch
        return desc

    @staticmethod
    def add_recompute_ops(subgraph, desc):
        """add recompute ops to desc"""
        if subgraph.recompute_ops:
            desc['recompute_ops'] = [op.output.name for op in subgraph.recompute_ops]
        return desc

    def refine(self):
        """Refine Graph"""
        AlignShape().visit_graph(self.graph)

    def load(self, desc):
        """Load Graph from json"""
        def _attr_of(op):
            if not op['attr']:
                return dict()
            attr = {}
            for a in op['attr']:
                if a['name'] == 'axis' and op['name'] in ('ReduceSum', 'ReduceMax', 'ReduceMin', 'Argmax', 'Argmin'):
                    attr['reduce_axis'] = a['value']
                else:
                    attr[a['name']] = a['value']
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
            for op in desc['op_desc']:
                inputs = [self.tensors.get(d['tensor_name'], None) for x in op['input_desc']
                          for d in x if 'value' not in d]
                if op['name'] in ('ReduceSum', 'ReduceMax', 'ReduceMin'):
                    axis = op['input_desc'][1][0]['value']
                    if isinstance(axis, int):
                        axis = [axis]
                    if not op['attr']:
                        attr = [{'name': 'axis', 'dtype': 'listInt', 'value': axis}]
                        op['attr'] = attr
                    else:
                        op['attr'].append({'name': 'axis', 'dtype': 'listInt', 'value': axis})
                out_desc = op['output_desc']
                name, shape, dtype, data_format = out_desc[0]['tensor_name'], out_desc[
                    0]['shape'], out_desc[0]['data_type'], out_desc[0]['format']
                output = self.tensors.get(name, None)
                if not output:
                    output = builder.tensor(shape, dtype, data_format, name=name)
                    self.tensors[name] = output
                builder.op(op['name'], output, inputs, attrs=_attr_of(op))
        self.graph = builder.get()[0]
        self.desc = desc


    def dump(self, subgraph):
        """Dump Graph to json"""
        desc = {}
        inputs, outputs = subgraph.deduce_parameters()
        graph_ops = set(subgraph.ops)

        def dump_output(t):
            return {'data_type': t.dtype, 'shape': t.shape, 'tensor_name': t.name}

        def dump_op_desc(d):
            op = self.tensors[d['output_desc'][0]['tensor_name']].op
            if op in graph_ops or op in subgraph.recompute_ops:
                return d
            return None

        for key in self.desc.keys():
            if key == 'input_desc':
                desc[key] = [[{'data_type': t.dtype, 'shape': t.shape, 'tensor_name': t.name}] for t in inputs]
            elif key == 'output_desc':
                desc[key] = list(map(dump_output, outputs))
            elif key == 'op_desc':
                op_desc = map(dump_op_desc, self.desc[key])
                desc[key] = [d for d in op_desc if d is not None]
            elif key == 'op':
                desc[key] = subgraph.name
            else:
                desc[key] = self.desc[key]

        desc = self.add_stitch_info(subgraph, desc)
        desc = self.add_recompute_ops(subgraph, desc)
        return desc


def load_composite(desc):
    """Load composite kernel"""
    composite = CompositeGraph()
    composite.load(desc)
    composite.refine()
    return composite
