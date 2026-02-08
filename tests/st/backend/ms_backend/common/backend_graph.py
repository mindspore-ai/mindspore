# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Backend graph mock for backend test.
"""
from mindspore._c_expression import BackendGraphMock_
from mindspore import Parameter
class BackendGraph(BackendGraphMock_):
    def __init__(self):
        BackendGraphMock_.__init__(self)

    def __repr__(self):
        return self.__str__()

    def __call__(self, *args):
        return super().__call__(args)

    def add_parameter(self, para_type, shape=()):
        '''
        Add a parameter to the backend graph. If weight is added, the para_type should be a Parameter,
        otherwise pass in type and shape.
        '''
        if isinstance(para_type, Parameter):
            return self.add_weight_parameter_(para_type)
        return self.add_parameter_(para_type, shape)

    def add_cnode(self, *args):
        return self.add_cnode_(args)

    def add_valuenode(self, value):
        return self.add_valuenode_(value)

    def add_subgraph(self, graph):
        return self.add_subgraph_(graph)

    def add_return(self, return_input):
        return self.add_return_(return_input)

    def set_abstract(self, *args):
        '''
        Two usage methods:
        1. set_abstract(src_node, dst_node)
        Assign the abstract property of the src node to the dst node.
        2. set_abstract(node, type, shape)
        Set the abstract property represented by type and shape to the node.
        '''
        return self.set_abstract_(args)

    def set_target(self, node, target):
        return self.set_target_(node, target)

    def set_cell_reuse(self):
        return self.set_cell_reuse_()

    def set_input(self, cnode, index, input_node):
        return self.set_input_(cnode, index, input_node)

    def infer(self):
        '''
        Generate abstract by Renormalize, and constant folding may occur in graph.
        '''
        return self.infer_()

    def native_infer(self):
        '''
        Generate abstract by call infer shape and type to avoid constant folding.
        Control flow is unsupported in this.
        '''
        return self.native_infer_()

    def skip_infer(self):
        '''
        No need to generate abstract automatically.
        '''
        return self.skip_infer_()

    def set_jit_config(self, config):
        return self.set_jit_config_(config)

    def compile(self):
        return self.compile_()
