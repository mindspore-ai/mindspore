# Copyright 2023 Huawei Technologies Co., Ltd
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
# ============================================================================
"""onnx graph modifier"""
import os
from abc import ABC

import onnx

from mslite_bench.graphs.graph_modifier import ABCGraphModifier


class OnnxModifier(ABCGraphModifier, ABC):
    """ modifier for onnx model"""
    def __init__(self, model_path):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{model_path} does not exist!')
        onnx.checker.check_model(model_path)
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self._check_model()
        self.model_input_names = self._get_input_names()
        self.model_output_names = [model_output.name for model_output in self.model.graph.output]
        self.black_node_type_list = {
            'Pad', 'Div', 'Const', 'Shape', 'ConstOfShape', 'Slice', 'Cast', 'Gather',
            'Reshape', 'Unsqueeze', 'Mul', 'RandomNormalLike', 'Exp', 'InstanceNormalization'
            'Where', 'Equal', 'Greater', 'Clip', 'Range', 'IsInf', 'IsNaN', 'Less', 'Loop'
            'Not', 'Or', 'Xor', 'And', 'BitwiseNot', 'BitwiseAnd', 'BitwiseXor', 'BitwiseOr'
            'BitwiseNot', 'BatchNormalization', 'Constant'
        }

    def extract_model(self,
                      save_path,
                      input_names=None,
                      output_names=None):
        """ extract sub model based on input and output tensor names"""
        if input_names is None:
            input_names = self.model_input_names

        if output_names is None:
            output_names = self.model_output_names
        elif len(output_names) == 1 and output_names[0].lower() == 'mslite_bench_all':
            output_names = list(set(self._all_node_names()) - set(self.model_input_names))
        else:
            valid_node_names = set(self._all_node_names())
            invalid_out_names = [name for name in output_names if name not in valid_node_names]
            if invalid_out_names:
                output_names = list(set(output_names) - set(invalid_out_names))
                self.logger.warning('Output nodes %s are not supported for '
                                    'accuracy compare', invalid_out_names)
            if not output_names:
                raise ValueError('Shall input valid output names, but it is empty or all invalid')

        if not isinstance(input_names, list) or not isinstance(output_names, list):
            raise ValueError("input and output nodes name shall be a list")

        try:
            onnx.utils.extract_model(self.model_path,
                                     save_path,
                                     input_names,
                                     output_names,
                                     check_model=False)
        except KeyError as e:
            self.logger.error('Extract sub model failed, this tensor name is not in graph.value_info: %s', e)
            raise
        return output_names

    def _all_node_names(self):
        """return all node names in network"""
        def is_in_black_node_list(node):
            return node.op_type in self.black_node_type_list

        output_names = [node.output for node in self.model.graph.node if not is_in_black_node_list(node)]
        ret_names = []
        for out_name in output_names:
            for name in out_name:
                ret_names.append(name)
        return ret_names

    def _sorted_blocks(self):
        """get sorted blocks based on feed-foward network"""
        return []

    def _get_input_names(self):
        """get all input nodes"""
        all_input_node_names = {item.name for item in self.model.graph.input}
        input_initializer_node_names = {item.name for item in self.model.graph.initializer}

        return list(all_input_node_names - input_initializer_node_names)

    def _check_model(self):
        """check model whether has value info"""
        graph = onnx.shape_inference.infer_shapes(self.model).graph
        if not graph.value_info:
            self.logger.error('Model value info is empty, this model is not supported!')
            raise ValueError('model value info is empty')
